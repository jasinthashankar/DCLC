"""InfoNCE Contrastive Loss and Boundary-Aware Negative Sampling for DCLC.

Implements the contrastive learning formulation from Section 3.2 (Eq. 14, Page 9)
and training settings from Section 4.1 (Page 13):
  - Temperature tau = 0.07.
  - Up to 15 negative samples per central chunk.
  - Boundary-aware negative exclusion:
      1. Candidates within the anchor's local context window W_i = [C_{i-K}, ..., C_{i+K}].
      2. Candidates sharing the exact same content fingerprint (SHA-256).
      3. Candidates with strong proximity in the initial feature space (cosine similarity >= threshold).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """InfoNCE loss for DCLC contrastive representation learning (Equation 14).

    Formula:
        L_i = -log( exp(sim(h_i, h_i^+) / tau) / ( exp(sim(h_i, h_i^+) / tau) + sum_{z in N_i} exp(sim(h_i, z) / tau) ) )

    where sim(u, v) is cosine similarity (dot product for normalized embeddings),
    tau is the temperature parameter (0.07 in paper Section 4.1),
    h_i^+ is the positive representation, and N_i is the set of valid negative samples.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        negatives_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.

        Args:
            anchor: Tensor of shape (B, D), L2-normalized embedding h_i.
            positive: Tensor of shape (B, D), positive sample embedding h_i^+.
            negatives: Tensor of shape (B, K_neg, D), negative sample embeddings.
            negatives_mask: Optional boolean Tensor of shape (B, K_neg),
                            True where negative is valid, False where masked/padded.

        Returns:
            Scalar loss Tensor (or per-sample loss if reduction='none').
        """
        if anchor.dim() != 2 or positive.dim() != 2:
            raise ValueError("anchor and positive must be 2D tensors of shape (B, D)")
        if negatives.dim() != 3:
            raise ValueError("negatives must be a 3D tensor of shape (B, K_neg, D)")

        b, d = anchor.shape
        b_neg, k_neg, d_neg = negatives.shape

        if b != b_neg or b != positive.shape[0]:
            raise ValueError(f"Batch dimension mismatch: anchor {b}, positive {positive.shape[0]}, negatives {b_neg}")
        if d != positive.shape[1] or d != d_neg:
            raise ValueError(f"Embedding dimension mismatch: anchor {d}, positive {positive.shape[1]}, negatives {d_neg}")

        # Compute cosine similarities (embeddings are normalized, so dot product = cosine similarity)
        # Positive similarity: shape (B, 1)
        pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature

        # Negative similarities: shape (B, K_neg)
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature

        # Determine which anchors have at least one valid negative
        if negatives_mask is not None:
            if negatives_mask.shape != (b, k_neg):
                raise ValueError(
                    f"negatives_mask shape {negatives_mask.shape} does not match negatives (B, K_neg)=({b}, {k_neg})"
                )
            # Mask invalid negatives with large negative number so exp(sim) -> 0
            neg_sim = neg_sim.masked_fill(~negatives_mask, -1e9)
            has_valid_negatives = negatives_mask.sum(dim=-1) > 0
        else:
            has_valid_negatives = (
                torch.ones(b, dtype=torch.bool, device=anchor.device)
                if k_neg > 0
                else torch.zeros(b, dtype=torch.bool, device=anchor.device)
            )

        logits = torch.cat([pos_sim, neg_sim], dim=1)  # shape (B, 1 + K_neg)
        targets = torch.zeros(b, dtype=torch.long, device=anchor.device)

        # Compute per-sample cross-entropy
        loss_per_sample = F.cross_entropy(logits, targets, reduction="none")

        num_usable = has_valid_negatives.sum()
        if num_usable == 0:
            logger.warning(
                "Batch contains 0 anchors with valid negative samples. "
                "Returning zero loss without gradient."
            )
            if self.reduction == "none":
                return torch.zeros(b, device=anchor.device)
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)

        # Exclude anchors with zero valid negatives from the loss average
        if self.reduction == "mean":
            loss = (loss_per_sample * has_valid_negatives.float()).sum() / num_usable.float()
        elif self.reduction == "sum":
            loss = (loss_per_sample * has_valid_negatives.float()).sum()
        elif self.reduction == "none":
            loss = loss_per_sample * has_valid_negatives.float()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        return loss



class BoundaryAwareNegativeSampler:
    """Boundary-aware negative sampling for DCLC contrastive learning.

    Paper Section 3.2 (Page 9):
      "To avoid false-negative pairs, the negative sample set N_i is constructed
      using a boundary-aware constrained sampling strategy:
        1. Context window W_i = [C_{i-K}, ..., C_{i+K}] from the same sequence.
        2. Same fingerprints as C_i.
        3. Strong proximity to C_i in the initial feature space are filtered out."
    """

    def __init__(
        self,
        file_ids: np.ndarray,
        chunk_indices: np.ndarray,
        fingerprints: list[str],
        initial_features: np.ndarray,
        context_radius: int = 4,
        proximity_threshold: float = 0.85,
        max_negatives: int = 15,
        seed: Optional[int] = 42,
    ):
        """Initialize the sampler with dataset metadata.

        Args:
            file_ids: 1D array of length N with file identifier for each chunk.
            chunk_indices: 1D array of length N with chunk sequence index.
            fingerprints: List of length N with SHA-256 fingerprint strings.
            initial_features: Array of shape (N, D) containing initial feature vectors.
            context_radius: Radius K of local context window W_i (paper default K=4).
            proximity_threshold: Cosine similarity cutoff in initial feature space.
            max_negatives: Maximum negatives to sample per anchor (paper default 15).
            seed: Random seed for reproducible negative sampling.
        """
        self.num_chunks = len(file_ids)
        if len(chunk_indices) != self.num_chunks or len(fingerprints) != self.num_chunks:
            raise ValueError("All metadata lists/arrays must have the same length.")
        if len(initial_features) != self.num_chunks:
            raise ValueError("initial_features must have same length as metadata.")

        self.file_ids = np.asarray(file_ids, dtype=np.int32)
        self.chunk_indices = np.asarray(chunk_indices, dtype=np.int32)
        self.fingerprints = list(fingerprints)

        # Normalize initial features for fast cosine similarity computation
        norms = np.linalg.norm(initial_features, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normalized_initial_feats = (initial_features / norms).astype(np.float32)

        self.context_radius = context_radius
        self.proximity_threshold = proximity_threshold
        self.max_negatives = max_negatives
        self.rng = np.random.default_rng(seed)

    def get_valid_negative_indices(self, anchor_idx: int) -> tuple[np.ndarray, dict[str, int]]:
        """Filter the chunk corpus to select valid negative sample indices for anchor_idx.

        Returns:
            Tuple of (sampled_negative_indices, diagnostics_dict).
        """
        if not (0 <= anchor_idx < self.num_chunks):
            raise IndexError(f"anchor_idx {anchor_idx} out of range [0, {self.num_chunks})")

        anchor_fid = self.file_ids[anchor_idx]
        anchor_cidx = self.chunk_indices[anchor_idx]
        anchor_fp = self.fingerprints[anchor_idx]
        anchor_feat = self.normalized_initial_feats[anchor_idx]

        # Candidate mask excluding anchor itself
        candidate_mask = np.ones(self.num_chunks, dtype=bool)
        candidate_mask[anchor_idx] = False

        # 1. Local context window exclusion (Rule 1)
        # Same file and |chunk_indices[j] - chunk_indices[anchor]| <= context_radius
        is_same_file = self.file_ids == anchor_fid
        is_in_window = is_same_file & (np.abs(self.chunk_indices - anchor_cidx) <= self.context_radius) & candidate_mask

        # 2. Same fingerprint exclusion (Rule 2)
        is_same_fp = np.array([fp == anchor_fp for fp in self.fingerprints], dtype=bool) & candidate_mask

        # 3. Strong proximity in initial feature space (Rule 3)
        # Cosine similarity between anchor initial feature and candidate initial features
        sims = self.normalized_initial_feats @ anchor_feat
        is_proximate = (sims >= self.proximity_threshold) & candidate_mask

        # Exclude anchor itself and all violated rules
        is_invalid = (~candidate_mask) | is_in_window | is_same_fp | is_proximate

        valid_indices = np.where(~is_invalid)[0]

        diagnostics = {
            "total_candidates": self.num_chunks - 1,
            "excluded_context_window": int(np.sum(is_in_window)),
            "excluded_same_fingerprint": int(np.sum(is_same_fp)),
            "excluded_proximity": int(np.sum(is_proximate)),
            "total_valid_negatives": len(valid_indices),
        }


        if len(valid_indices) == 0:
            return np.empty(0, dtype=np.int32), diagnostics

        # Sample up to max_negatives
        if len(valid_indices) > self.max_negatives:
            chosen = self.rng.choice(valid_indices, size=self.max_negatives, replace=False)
        else:
            chosen = valid_indices

        return chosen.astype(np.int32), diagnostics
