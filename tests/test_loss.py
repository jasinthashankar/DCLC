"""Tests for DCLC InfoNCE Loss and Boundary-Aware Negative Sampling."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.loss import InfoNCELoss, BoundaryAwareNegativeSampler


def test_infonce_loss_finite_value():
    loss_fn = InfoNCELoss(temperature=0.07)

    # Batch of 3 samples with embedding dimension 128, 5 negatives each
    anchor = F.normalize(torch.randn(3, 128), p=2, dim=-1)
    positive = F.normalize(torch.randn(3, 128), p=2, dim=-1)
    negatives = F.normalize(torch.randn(3, 5, 128), p=2, dim=-1)

    loss = loss_fn(anchor, positive, negatives)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_infonce_loss_prefers_closer_positives():
    loss_fn = InfoNCELoss(temperature=0.07)

    anchor = F.normalize(torch.randn(2, 64), p=2, dim=-1)
    negatives = F.normalize(torch.randn(2, 4, 64), p=2, dim=-1)

    # High positive similarity (identical to anchor)
    good_positive = anchor.clone()
    # Low positive similarity (opposite of anchor)
    bad_positive = -anchor.clone()

    low_loss = loss_fn(anchor, good_positive, negatives)
    high_loss = loss_fn(anchor, bad_positive, negatives)

    assert low_loss.item() < high_loss.item()


def test_infonce_loss_with_mask():
    loss_fn = InfoNCELoss(temperature=0.07)

    anchor = F.normalize(torch.randn(2, 32), p=2, dim=-1)
    positive = F.normalize(torch.randn(2, 32), p=2, dim=-1)
    negatives = F.normalize(torch.randn(2, 4, 32), p=2, dim=-1)

    # Mask out the last 2 negatives (only 2 valid negatives)
    mask = torch.tensor([
        [True, True, False, False],
        [True, True, True, False],
    ], dtype=torch.bool)

    loss = loss_fn(anchor, positive, negatives, negatives_mask=mask)
    assert torch.isfinite(loss)


def test_infonce_loss_anchor_with_zero_negatives_excluded_from_mean():
    loss_fn = InfoNCELoss(temperature=0.07)

    # Batch of 2 anchors: anchor 0 has valid negatives, anchor 1 has 0 valid negatives
    anchor = F.normalize(torch.randn(2, 32), p=2, dim=-1)
    positive = F.normalize(torch.randn(2, 32), p=2, dim=-1)
    negatives = F.normalize(torch.randn(2, 4, 32), p=2, dim=-1)

    # Mask: sample 0 has 2 valid negatives, sample 1 has 0 valid negatives
    mask = torch.tensor([
        [True, True, False, False],
        [False, False, False, False],
    ], dtype=torch.bool)

    combined_loss = loss_fn(anchor, positive, negatives, negatives_mask=mask)

    # Calculate loss for anchor 0 alone
    anchor0_loss = loss_fn(anchor[:1], positive[:1], negatives[:1, :2])

    # The combined loss should equal anchor 0's loss, because anchor 1 must be excluded from the mean
    torch.testing.assert_close(combined_loss, anchor0_loss)


def test_infonce_loss_batch_with_no_usable_anchors():
    loss_fn = InfoNCELoss(temperature=0.07)

    # All anchors in batch have 0 valid negatives
    anchor = F.normalize(torch.randn(2, 16), p=2, dim=-1)
    positive = F.normalize(torch.randn(2, 16), p=2, dim=-1)
    negatives = F.normalize(torch.randn(2, 4, 16), p=2, dim=-1)

    mask = torch.zeros(2, 4, dtype=torch.bool)

    loss = loss_fn(anchor, positive, negatives, negatives_mask=mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_infonce_invalid_arguments():
    with pytest.raises(ValueError, match="Temperature must be positive"):
        InfoNCELoss(temperature=0.0)


    loss_fn = InfoNCELoss(temperature=0.07)
    with pytest.raises(ValueError, match="Batch dimension mismatch"):
        loss_fn(
            torch.randn(2, 16),
            torch.randn(3, 16),
            torch.randn(2, 4, 16),
        )


def test_negative_sampler_context_window_exclusion():
    # 10 chunks from file 0, indices 0...9
    file_ids = np.zeros(10, dtype=np.int32)
    chunk_indices = np.arange(10, dtype=np.int32)
    fingerprints = [f"unique_hash_{i}" for i in range(10)]
    # Orthogonal initial features so proximity threshold does not trigger
    initial_features = np.eye(10, dtype=np.float32)

    sampler = BoundaryAwareNegativeSampler(
        file_ids=file_ids,
        chunk_indices=chunk_indices,
        fingerprints=fingerprints,
        initial_features=initial_features,
        context_radius=2,  # window of +/- 2 chunks
        proximity_threshold=0.85,
        max_negatives=15,
    )

    # Anchor at index 4: window is [2, 3, 4, 5, 6]
    sampled, diag = sampler.get_valid_negative_indices(anchor_idx=4)

    # Excluded in window among other chunks should be 4 chunks (2, 3, 5, 6)
    assert diag["excluded_context_window"] == 4
    # Valid indices should only be from {0, 1, 7, 8, 9}
    for idx in sampled:
        assert idx not in {2, 3, 4, 5, 6}



def test_negative_sampler_same_fingerprint_exclusion():
    # 2 files, but chunk 0 in file 0 and chunk 0 in file 1 share same fingerprint
    file_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    chunk_indices = np.array([0, 1, 0, 1], dtype=np.int32)
    fingerprints = ["dup_hash", "hash_b", "dup_hash", "hash_d"]
    initial_features = np.eye(4, dtype=np.float32)

    sampler = BoundaryAwareNegativeSampler(
        file_ids=file_ids,
        chunk_indices=chunk_indices,
        fingerprints=fingerprints,
        initial_features=initial_features,
        context_radius=0,
        proximity_threshold=0.85,
        max_negatives=5,
    )

    # Anchor is chunk 0 (fingerprint "dup_hash")
    sampled, diag = sampler.get_valid_negative_indices(anchor_idx=0)

    # Chunk 2 has the same fingerprint "dup_hash", so it MUST be excluded
    assert 2 not in sampled
    assert diag["excluded_same_fingerprint"] >= 1


def test_negative_sampler_proximity_exclusion():
    # 2 files, different fingerprints, but chunk 1 is highly similar to anchor
    file_ids = np.array([0, 1], dtype=np.int32)
    chunk_indices = np.array([0, 0], dtype=np.int32)
    fingerprints = ["hash_a", "hash_b"]

    # Candidate has cosine similarity 0.95 with anchor (exceeds 0.85 threshold)
    feat_0 = np.array([1.0, 0.0], dtype=np.float32)
    feat_1 = np.array([0.95, 0.3122], dtype=np.float32)
    initial_features = np.stack([feat_0, feat_1])

    sampler = BoundaryAwareNegativeSampler(
        file_ids=file_ids,
        chunk_indices=chunk_indices,
        fingerprints=fingerprints,
        initial_features=initial_features,
        context_radius=0,
        proximity_threshold=0.85,
    )

    sampled, diag = sampler.get_valid_negative_indices(anchor_idx=0)
    # Since candidate is proximate (0.95 >= 0.85), it should be excluded
    assert len(sampled) == 0
    assert diag["excluded_proximity"] == 1


def test_negative_sampler_insufficient_negatives_handled():
    # Only 1 chunk total in dataset: anchor itself
    file_ids = np.array([0], dtype=np.int32)
    chunk_indices = np.array([0], dtype=np.int32)
    fingerprints = ["single_hash"]
    initial_features = np.array([[1.0, 2.0]], dtype=np.float32)

    sampler = BoundaryAwareNegativeSampler(
        file_ids=file_ids,
        chunk_indices=chunk_indices,
        fingerprints=fingerprints,
        initial_features=initial_features,
        context_radius=2,
    )

    sampled, diag = sampler.get_valid_negative_indices(anchor_idx=0)
    assert len(sampled) == 0
    assert diag["total_valid_negatives"] == 0
