"""Tests for Version-Aware Contrastive Training and Evaluation (Phase 4B)."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.evaluate import load_model_and_checkpoint
from src.version_aware_trainer import (
    VersionAwareNegativeSampler,
    build_split_pairs,
    VersionResemblancePairDataset,
)
from src.multiversion_evaluator import evaluate_test_split


def test_version_aware_negative_sampler_exclusions():
    """Verify that VersionAwareNegativeSampler strictly excludes context window, SHA-256, initial feature proximity, and lineage overlap."""
    manifest_chunks = [
        # Anchor chunk 0 (Family 1, File 1, chunk 0)
        {"family_id": 1, "file_id": 1, "chunk_index": 0, "sha256": "hash_anchor", "base_record_ids": [10, 11]},
        # Chunk 1: in context window (|0 - 2| <= 4 in file 1) -> MUST BE EXCLUDED
        {"family_id": 1, "file_id": 1, "chunk_index": 2, "sha256": "hash_win", "base_record_ids": [99]},
        # Chunk 2: different file, same SHA-256 -> MUST BE EXCLUDED
        {"family_id": 1, "file_id": 2, "chunk_index": 0, "sha256": "hash_anchor", "base_record_ids": [10, 11]},
        # Chunk 3: different file, shares lineage J=2/3 >= 0.10 -> MUST BE EXCLUDED
        {"family_id": 1, "file_id": 2, "chunk_index": 5, "sha256": "hash_resem", "base_record_ids": [10, 11, 12]},
        # Chunk 4: different family, initial feature cosine >= 0.85 -> MUST BE EXCLUDED
        {"family_id": 2, "file_id": 3, "chunk_index": 0, "sha256": "hash_prox", "base_record_ids": [100]},
        # Chunk 5: different family, orthogonal feature -> VALID NEGATIVE
        {"family_id": 2, "file_id": 3, "chunk_index": 1, "sha256": "hash_valid1", "base_record_ids": [200]},
        # Chunk 6: same family, different file, zero lineage overlap -> VALID NEGATIVE
        {"family_id": 1, "file_id": 2, "chunk_index": 10, "sha256": "hash_valid2", "base_record_ids": [300]},
    ]

    initial_features = np.zeros((7, 128), dtype=np.float32)
    initial_features[0, 0] = 1.0  # Anchor along dim 0
    initial_features[1, 1] = 1.0
    initial_features[2, 0] = 1.0
    initial_features[3, 1] = 1.0
    initial_features[4, 0] = 0.99  # Proximity cosine ~ 0.99 >= 0.85
    initial_features[5, 5] = 1.0   # Orthogonal (cosine = 0.0)
    initial_features[6, 6] = 1.0   # Orthogonal (cosine = 0.0)

    sampler = VersionAwareNegativeSampler(
        chunk_indices_in_split=list(range(7)),
        manifest_chunks=manifest_chunks,
        initial_features=initial_features,
        context_radius=4,
        proximity_threshold=0.85,
        max_negatives=15,
        seed=42,
    )

    neg_indices = sampler.get_valid_negative_indices(anchor_global_id=0)

    # Valid candidates should only be chunk 5 and chunk 6
    assert set(neg_indices) == {5, 6}
    assert 1 not in neg_indices  # Excluded by context window
    assert 2 not in neg_indices  # Excluded by SHA-256 match
    assert 3 not in neg_indices  # Excluded by version lineage overlap
    assert 4 not in neg_indices  # Excluded by initial feature proximity


def test_build_split_pairs_manifest_partitioning():
    manifest_path = Path("data/multi_version/processed/multiversion_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Multi-version manifest not found.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest["chunks"]

    train_resem, train_dups, train_cids = build_split_pairs(chunks, "train")
    val_resem, val_dups, val_cids = build_split_pairs(chunks, "val")
    test_resem, test_dups, test_cids = build_split_pairs(chunks, "test")

    # Verify counts match audit exactly
    assert len(train_cids) == 1938
    assert len(val_cids) == 399
    assert len(test_cids) == 464

    assert len(train_resem) == 2278
    assert len(val_resem) == 466
    assert len(test_resem) == 542

    assert len(train_dups) == 75
    assert len(val_dups) == 14
    assert len(test_dups) == 26

    # Verify complete split isolation: no chunk in train pairs belongs to val or test
    train_cids_set = set(train_cids)
    val_cids_set = set(val_cids)
    test_cids_set = set(test_cids)

    for a, b in train_resem + train_dups:
        assert a in train_cids_set and b in train_cids_set
        assert a not in val_cids_set and b not in val_cids_set
        assert a not in test_cids_set and b not in test_cids_set


def test_version_resemblance_dataset_shapes():
    context_sequences = np.random.randn(10, 9, 128).astype(np.float32)
    pairs = [(0, 1), (2, 3), (4, 5)]
    ds = VersionResemblancePairDataset(pairs, context_sequences)

    assert len(ds) == 3
    anc_seq, pos_seq, anc_id = ds[0]
    assert anc_seq.shape == (9, 128)
    assert pos_seq.shape == (9, 128)
    assert anc_id == 0
    assert isinstance(anc_seq, torch.Tensor)
    assert isinstance(pos_seq, torch.Tensor)
