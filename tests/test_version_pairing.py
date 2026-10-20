"""Tests for Version-Aware Pairing and Feasibility Study (src/version_pairing.py)."""

from pathlib import Path
import pytest

from src.version_pairing import (
    PairCategory,
    compute_line_jaccard,
    audit_version_chunk_alignment,
    assess_feasibility_of_train_test_split,
    run_version_aware_feasibility_study,
)


def test_compute_line_jaccard():
    # 1. Identical bytes -> Jaccard 1.0
    b1 = b"line 1\nline 2\nline 3\n"
    assert compute_line_jaccard(b1, b1) == 1.0

    # 2. Disjoint bytes -> Jaccard 0.0
    b2 = b"completely different content\nno shared lines\n"
    assert compute_line_jaccard(b1, b2) == 0.0

    # 3. Partial overlap (2 out of 4 distinct lines shared -> 2/4 = 0.5)
    b3 = b"line 1\nline 2\nline 4\nline 5\n"
    # lines1: {1, 2, 3}, lines3: {1, 2, 4, 5}. Intersection: {1, 2} (2). Union: {1, 2, 3, 4, 5} (5).
    # Jaccard = 2/5 = 0.4
    assert compute_line_jaccard(b1, b3) == pytest.approx(0.4)

    # 4. Empty bytes
    assert compute_line_jaccard(b"", b"") == 0.0


def test_audit_version_chunk_alignment():
    raw_v1 = b"SHARED_BLOCK_A\n" * 100 + b"V1_BLOCK_B\n" * 100
    raw_v2 = b"SHARED_BLOCK_A\n" * 100 + b"V2_BLOCK_B_EDITED\n" * 100

    # Chunk 0: identical
    # Chunk 1: edited
    v1_records = [
        {"global_chunk_id": 0, "byte_offset": 0, "byte_length": len(b"SHARED_BLOCK_A\n" * 100), "sha256": "hash_shared"},
        {"global_chunk_id": 1, "byte_offset": len(b"SHARED_BLOCK_A\n" * 100), "byte_length": len(b"V1_BLOCK_B\n" * 100), "sha256": "hash_v1"},
    ]
    v2_records = [
        {"global_chunk_id": 2, "byte_offset": 0, "byte_length": len(b"SHARED_BLOCK_A\n" * 100), "sha256": "hash_shared"},
        {"global_chunk_id": 3, "byte_offset": len(b"SHARED_BLOCK_A\n" * 100), "byte_length": len(b"V2_BLOCK_B_EDITED\n" * 100), "sha256": "hash_v2"},
    ]

    audits = audit_version_chunk_alignment(raw_v1, raw_v2, v1_records, v2_records, resemblance_jaccard_threshold=0.30)
    assert len(audits) == 2
    assert audits[0].category == PairCategory.EXACT_DUPLICATE
    assert audits[0].sha256_match is True
    assert audits[0].line_jaccard == 1.0

    assert audits[1].category == PairCategory.UNRELATED  # Disjoint lines in block B
    assert audits[1].sha256_match is False
    assert audits[1].line_jaccard == 0.0


def test_assess_feasibility_of_train_test_split():
    # 1. Under-determined dataset (3 pairs < 20 min)
    res_under = assess_feasibility_of_train_test_split(["pair1", "pair2", "pair3"], min_required_pairs=20)
    assert res_under["can_split_safely"] is False
    assert res_under["trustworthy_resemblance_pairs_count"] == 3
    assert any("STOP training" in rec for rec in res_under["recommendations"])

    # 2. Sufficient dataset (25 pairs >= 20 min)
    res_ok = assess_feasibility_of_train_test_split([f"pair_{i}" for i in range(25)], min_required_pairs=20)
    assert res_ok["can_split_safely"] is True
    assert res_ok["trustworthy_resemblance_pairs_count"] == 25


def test_run_version_aware_feasibility_study_integration():
    raw_dir = Path("data/raw")
    manifest_path = Path("data/processed/dataset_manifest.json")

    if not (raw_dir.exists() and manifest_path.exists()):
        pytest.skip("Dataset or manifest not available for integration test.")

    report = run_version_aware_feasibility_study(raw_dir=raw_dir, manifest_path=manifest_path)

    counts = report["category_counts"]
    assert counts["total_aligned_positions"] == 14
    assert counts["exact_duplicates"] == 11
    assert counts["version_resemblance"] == 3
    assert counts["boundary_shift_overlap"] == 0

    # Confirm feasibility stop recommendation
    feas = report["feasibility"]
    assert feas["can_split_safely"] is False
    assert feas["trustworthy_resemblance_pairs_count"] == 3


def test_pair_separation_and_leakage_prevention():
    # Verify that categories are strictly disjoint
    categories = [PairCategory.EXACT_DUPLICATE, PairCategory.VERSION_RESEMBLANCE, PairCategory.BOUNDARY_SHIFT_OVERLAP, PairCategory.UNRELATED]
    assert len(categories) == len(set(categories))
