"""Version-Aware Contrastive Training for DCLC (Phase 4B).

Trains the DCLC Transformer Context Encoder using ground-truth version-resemblance
positive pairs and boundary-aware negative sampling strictly within the training split.

Key Guardrails:
1. Split Isolation:
   - Training uses ONLY training families (000-069).
   - Validation uses ONLY validation families (070-084) for model selection and early stopping.
   - Test families (085-099) are never touched during training or validation.
2. Ground-Truth Positive Pairs:
   - Positive pairs are constructed from verified cross-version resemblance pairs (Jaccard >= 0.50).
   - Exact duplicates (SHA-256 matches) are tracked as a separate category.
   - Ambiguous boundary shifts (0.10 <= Jaccard < 0.50) are strictly excluded from both positive
     training pairs and negative candidate sets.
3. Boundary-Aware Negative Sampling:
   - Negative samples enforce:
     * Context window exclusion (|i - j| <= K in same file)
     * SHA-256 fingerprint collision exclusion
     * Initial feature cosine proximity exclusion (sim >= 0.85)
     * Version-aware lineage exclusion (Jaccard >= 0.10 across revisions)
4. Checkpoint Isolation:
   - Saves strictly to checkpoints/dclc_encoder_version_aware.pt, preserving
     checkpoints/dclc_encoder.pt and checkpoints/dclc_encoder_corrected.pt.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.loss import InfoNCELoss
from src.model import DCLCTransformerContextEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VersionAwareNegativeSampler:
    """Boundary-aware and version-aware negative sampler restricted to a specific split."""

    def __init__(
        self,
        chunk_indices_in_split: list[int],
        manifest_chunks: list[dict[str, Any]],
        initial_features: np.ndarray,
        context_radius: int = 4,
        proximity_threshold: float = 0.85,
        max_negatives: int = 15,
        seed: int = 42,
    ):
        self.chunk_ids = np.array(chunk_indices_in_split, dtype=np.int32)
        self.num_chunks = len(self.chunk_ids)
        self.context_radius = context_radius
        self.proximity_threshold = proximity_threshold
        self.max_negatives = max_negatives
        self.rng = np.random.default_rng(seed)

        # Build lookup tables for chunks in split
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        self.family_ids = np.zeros(self.num_chunks, dtype=np.int32)
        self.file_ids = np.zeros(self.num_chunks, dtype=np.int32)
        self.file_chunk_indices = np.zeros(self.num_chunks, dtype=np.int32)
        self.fingerprints = []
        self.base_record_sets: list[set[int]] = []

        for idx, cid in enumerate(self.chunk_ids):
            meta = manifest_chunks[cid]
            self.family_ids[idx] = meta["family_id"]
            self.file_ids[idx] = meta["file_id"]
            self.file_chunk_indices[idx] = meta["chunk_index"]
            self.fingerprints.append(meta["sha256"])
            self.base_record_sets.append(set(meta["base_record_ids"]))

        # Normalized initial features for fast proximity check
        split_feats = initial_features[self.chunk_ids]
        norms = np.linalg.norm(split_feats, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normed_features = (split_feats / norms).astype(np.float32)

    def get_valid_negative_indices(self, anchor_global_id: int) -> np.ndarray:
        """Sample valid negative chunk global indices for a given anchor chunk."""
        if anchor_global_id not in self.id_to_idx:
            raise KeyError(f"Anchor {anchor_global_id} not in split chunks.")

        a_idx = self.id_to_idx[anchor_global_id]
        a_fid = self.family_ids[a_idx]
        a_file = self.file_ids[a_idx]
        a_cidx = self.file_chunk_indices[a_idx]
        a_fp = self.fingerprints[a_idx]
        a_records = self.base_record_sets[a_idx]
        a_feat = self.normed_features[a_idx]

        # 1. Candidate mask excluding anchor itself
        valid_mask = np.ones(self.num_chunks, dtype=bool)
        valid_mask[a_idx] = False

        # 2. Rule 1: Context window exclusion in same file
        is_same_file = self.file_ids == a_file
        in_window = is_same_file & (np.abs(self.file_chunk_indices - a_cidx) <= self.context_radius)
        valid_mask[in_window] = False

        # 3. Rule 2: SHA-256 fingerprint collision
        for idx in np.where(valid_mask)[0]:
            if self.fingerprints[idx] == a_fp:
                valid_mask[idx] = False

        # 4. Rule 3: Initial feature proximity (cosine >= proximity_threshold)
        sims = self.normed_features @ a_feat
        proximate = sims >= self.proximity_threshold
        valid_mask[proximate] = False

        # 5. Rule 4: Multi-version lineage overlap (Jaccard >= 0.10)
        same_fam_indices = np.where(valid_mask & (self.family_ids == a_fid))[0]
        if a_records:
            for idx in same_fam_indices:
                cand_records = self.base_record_sets[idx]
                if cand_records:
                    j = len(a_records & cand_records) / len(a_records | cand_records)
                    if j >= 0.10:
                        valid_mask[idx] = False

        valid_candidate_indices = np.where(valid_mask)[0]
        if len(valid_candidate_indices) == 0:
            return np.empty(0, dtype=np.int32)

        if len(valid_candidate_indices) > self.max_negatives:
            chosen_local = self.rng.choice(valid_candidate_indices, size=self.max_negatives, replace=False)
        else:
            chosen_local = valid_candidate_indices

        return self.chunk_ids[chosen_local]


class VersionResemblancePairDataset(Dataset):
    """PyTorch Dataset yielding (anchor_seq, positive_seq, anchor_global_id)."""

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        context_sequences: np.ndarray,
    ):
        self.pairs = pairs
        self.context_sequences = context_sequences

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        anc_id, pos_id = self.pairs[idx]
        anc_seq = torch.from_numpy(self.context_sequences[anc_id]).float()
        pos_seq = torch.from_numpy(self.context_sequences[pos_id]).float()
        return anc_seq, pos_seq, anc_id


def build_split_pairs(
    manifest_chunks: list[dict[str, Any]],
    split_name: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    """Extract version-resemblance pairs, exact-duplicate pairs, and chunk IDs for a split.

    Returns:
        resemblance_pairs: list of (anc_id, pos_id) with Jaccard >= 0.50 (unordered, non-duplicated).
        exact_dup_pairs: list of (anc_id, pos_id) with identical SHA-256.
        split_chunk_ids: list of global chunk IDs in this split.
    """
    chunks_in_split = [c for c in manifest_chunks if c["split"] == split_name]
    split_chunk_ids = [c["global_chunk_id"] for c in chunks_in_split]

    fam_to_chunks: dict[int, list[dict[str, Any]]] = {}
    for c in chunks_in_split:
        fam_to_chunks.setdefault(c["family_id"], []).append(c)

    resemblance_pairs: list[tuple[int, int]] = []
    exact_dup_pairs: list[tuple[int, int]] = []

    for fid, f_chunks in fam_to_chunks.items():
        n = len(f_chunks)
        for i in range(n):
            for j in range(i + 1, n):
                ca = f_chunks[i]
                cb = f_chunks[j]
                if ca["file_id"] == cb["file_id"]:
                    continue  # Only cross-version pairs

                id_a = ca["global_chunk_id"]
                id_b = cb["global_chunk_id"]

                if ca["sha256"] == cb["sha256"]:
                    exact_dup_pairs.append((id_a, id_b))
                    continue

                sa = set(ca["base_record_ids"])
                sb = set(cb["base_record_ids"])
                if not (sa or sb):
                    continue

                jacc = len(sa & sb) / len(sa | sb)
                if jacc >= 0.50:
                    resemblance_pairs.append((id_a, id_b))

    return resemblance_pairs, exact_dup_pairs, split_chunk_ids


def train_version_aware_model(
    manifest_path: str | Path = "data/multi_version/processed/multiversion_manifest.json",
    data_npz_path: str | Path = "data/multi_version/processed/multiversion_train_data.npz",
    output_checkpoint_path: str | Path = "checkpoints/dclc_encoder_version_aware.pt",
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    output_dim: int = 128,
    temperature: float = 0.07,
    batch_size: int = 64,
    num_epochs: int = 10,
    patience: int = 3,
    lr: float = 3e-4,
    stop_gradient_negatives: bool = False,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train the version-aware DCLC encoder and save checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load data
    logger.info(f"Loading multi-version dataset from {manifest_path} and {data_npz_path}...")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_chunks = manifest["chunks"]

    npz_data = np.load(data_npz_path)
    context_sequences = npz_data["context_sequences"].astype(np.float32)
    initial_features = npz_data["initial_features"].astype(np.float32)

    # 2. Extract train and val pairs
    train_pairs, train_dups, train_cids = build_split_pairs(manifest_chunks, "train")
    val_pairs, val_dups, val_cids = build_split_pairs(manifest_chunks, "val")

    logger.info(f"Train split: {len(train_cids)} chunks, {len(train_pairs)} resemblance pairs, {len(train_dups)} exact duplicates.")
    logger.info(f"Val split:   {len(val_cids)} chunks, {len(val_pairs)} resemblance pairs, {len(val_dups)} exact duplicates.")

    train_sampler = VersionAwareNegativeSampler(
        chunk_indices_in_split=train_cids,
        manifest_chunks=manifest_chunks,
        initial_features=initial_features,
        context_radius=4,
        proximity_threshold=0.85,
        max_negatives=15,
        seed=seed,
    )

    val_sampler = VersionAwareNegativeSampler(
        chunk_indices_in_split=val_cids,
        manifest_chunks=manifest_chunks,
        initial_features=initial_features,
        context_radius=4,
        proximity_threshold=0.85,
        max_negatives=15,
        seed=seed + 100,
    )

    train_loader = DataLoader(
        VersionResemblancePairDataset(train_pairs, context_sequences),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        VersionResemblancePairDataset(val_pairs, context_sequences),
        batch_size=batch_size,
        shuffle=False,
    )

    # 3. Model, Loss, Optimizer
    model = DCLCTransformerContextEncoder(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_dim=output_dim,
        normalize_output=True,
        input_norm="layernorm",
    ).to(device)

    loss_fn = InfoNCELoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    output_path = Path(output_checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    training_history = []

    logger.info(
        f"Starting version-aware training: epochs={num_epochs}, patience={patience}, "
        f"batch_size={batch_size}, lr={lr}, stop_gradient_negatives={stop_gradient_negatives}."
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_losses = []

        for anc_seqs, pos_seqs, anc_ids in train_loader:
            anc_seqs = anc_seqs.to(device)
            pos_seqs = pos_seqs.to(device)
            b = len(anc_ids)

            optimizer.zero_grad()
            anc_embs = model(anc_seqs)
            pos_embs = model(pos_seqs)

            # Sample negatives for each anchor in batch and vectorize them together
            neg_seq_list = []
            mask_list = []

            for b_i in range(b):
                anc_id = int(anc_ids[b_i])
                neg_cids = train_sampler.get_valid_negative_indices(anc_id)
                k_actual = len(neg_cids)

                if k_actual > 0:
                    neg_seq = context_sequences[neg_cids]
                    if k_actual < 15:
                        pad = np.zeros((15 - k_actual, 9, 128), dtype=np.float32)
                        neg_seq = np.concatenate([neg_seq, pad], axis=0)
                        mask = np.concatenate([
                            np.ones(k_actual, dtype=bool),
                            np.zeros(15 - k_actual, dtype=bool),
                        ])
                    else:
                        mask = np.ones(15, dtype=bool)
                else:
                    neg_seq = np.zeros((15, 9, 128), dtype=np.float32)
                    mask = np.zeros(15, dtype=bool)

                neg_seq_list.append(neg_seq)
                mask_list.append(mask)

            # Batched negative forward pass: ONE forward call for all negatives in the batch
            batch_neg_seqs = torch.from_numpy(np.stack(neg_seq_list)).float().to(device) # (B, 15, 9, 128)
            batch_masks = torch.from_numpy(np.stack(mask_list)).bool().to(device)       # (B, 15)

            if stop_gradient_negatives:
                with torch.no_grad():
                    batch_negs = model(batch_neg_seqs.view(b * 15, 9, 128)).view(b, 15, output_dim)
            else:
                batch_negs = model(batch_neg_seqs.view(b * 15, 9, 128)).view(b, 15, output_dim)

            if batch_masks.sum().item() == 0:
                continue

            loss = loss_fn(anc_embs, pos_embs, batch_negs, negatives_mask=batch_masks)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        mean_train_loss = float(np.mean(epoch_train_losses)) if epoch_train_losses else 0.0

        # Validation loop
        model.eval()
        val_losses = []
        pos_sims_epoch = []
        neg_sims_epoch = []

        with torch.no_grad():
            for val_anc, val_pos, val_anc_ids in val_loader:
                val_anc = val_anc.to(device)
                val_pos = val_pos.to(device)
                b_val = len(val_anc_ids)

                val_anc_embs = model(val_anc)
                val_pos_embs = model(val_pos)

                pos_sim = (val_anc_embs * val_pos_embs).sum(dim=-1)
                pos_sims_epoch.extend(pos_sim.cpu().numpy().tolist())

                val_neg_seq_list = []
                val_mask_list = []

                for b_i in range(b_val):
                    v_id = int(val_anc_ids[b_i])
                    neg_cids = val_sampler.get_valid_negative_indices(v_id)
                    k_actual = len(neg_cids)

                    if k_actual > 0:
                        neg_seq = context_sequences[neg_cids]
                        if k_actual < 15:
                            pad = np.zeros((15 - k_actual, 9, 128), dtype=np.float32)
                            neg_seq = np.concatenate([neg_seq, pad], axis=0)
                            mask = np.concatenate([
                                np.ones(k_actual, dtype=bool),
                                np.zeros(15 - k_actual, dtype=bool),
                            ])
                        else:
                            mask = np.ones(15, dtype=bool)
                    else:
                        neg_seq = np.zeros((15, 9, 128), dtype=np.float32)
                        mask = np.zeros(15, dtype=bool)

                    val_neg_seq_list.append(neg_seq)
                    val_mask_list.append(mask)

                batch_val_neg_seqs = torch.from_numpy(np.stack(val_neg_seq_list)).float().to(device)
                batch_val_masks = torch.from_numpy(np.stack(val_mask_list)).bool().to(device)

                batch_val_negs = model(batch_val_neg_seqs.view(b_val * 15, 9, 128)).view(b_val, 15, output_dim)

                # Record negative similarities
                anc_expanded = val_anc_embs.unsqueeze(1) # (B, 1, D)
                neg_sims = (anc_expanded * batch_val_negs).sum(dim=-1) # (B, 15)
                valid_negs = neg_sims[batch_val_masks]
                if len(valid_negs) > 0:
                    neg_sims_epoch.extend(valid_negs.cpu().numpy().tolist())

                if batch_val_masks.sum().item() > 0:
                    v_loss = loss_fn(val_anc_embs, val_pos_embs, batch_val_negs, negatives_mask=batch_val_masks)
                    val_losses.append(v_loss.item())

        mean_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        mean_pos_sim = float(np.mean(pos_sims_epoch)) if pos_sims_epoch else 0.0
        mean_neg_sim = float(np.mean(neg_sims_epoch)) if neg_sims_epoch else 0.0

        epoch_summary = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "val_mean_pos_sim": mean_pos_sim,
            "val_mean_neg_sim": mean_neg_sim,
            "val_sim_margin": mean_pos_sim - mean_neg_sim,
        }
        training_history.append(epoch_summary)

        logger.info(
            f"Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {mean_train_loss:.4f} | "
            f"Val Loss: {mean_val_loss:.4f} | Val Pos Sim: {mean_pos_sim:.4f} | "
            f"Val Neg Sim: {mean_neg_sim:.4f} | Margin: {mean_pos_sim - mean_neg_sim:+.4f}"
        )

        # Early stopping and checkpoint selection on validation loss
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
            logger.info(f"  -> New best validation loss {mean_val_loss:.4f}. Saved checkpoint to {output_path}")
        else:
            patience_counter += 1
            logger.info(f"  -> Val loss did not improve ({patience_counter}/{patience})")
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}.")
                break

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(output_path),
        "history": training_history,
        "hyperparameters": {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "temperature": temperature,
            "lr": lr,
            "batch_size": batch_size,
            "stop_gradient_negatives": stop_gradient_negatives,
            "seed": seed,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train version-aware DCLC encoder.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--stop-gradient-negatives", action="store_true", default=False)
    args = parser.parse_args()

    train_version_aware_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stop_gradient_negatives=args.stop_gradient_negatives,
    )
