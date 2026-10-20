"""Tests for Controlled Multi-Version Dataset Generator (Phase 4A)."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.multiversion_generator import (
    PairCategory,
    DatasetSplit,
    generate_base_family_records,
    mutate_family_version,
    map_records_to_chunks,
    classify_chunk_pair,
    generate_multiversion_dataset,
)


def test_generator_reproducibility():
    # Same seed and family ID must produce byte-for-byte identical base records
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    records1 = generate_base_family_records(family_id=5, num_records=50, rng=rng1)
    records2 = generate_base_family_records(family_id=5, num_records=50, rng=rng2)

    assert len(records1) == len(records2) == 50
    for r1, r2 in zip(records1, records2):
        assert r1.text == r2.text
        assert r1.base_record_id == r2.base_record_id
        assert r1.mutation_type == r2.mutation_type


def test_ground_truth_lineage_and_mutations():
    rng = np.random.default_rng(12345)
    base = generate_base_family_records(family_id=1, num_records=100, rng=rng)

    # Apply 15% mutation
    rev = mutate_family_version(base, family_id=1, version_idx=1, mutation_level=0.15, rng=rng)

    # Base IDs in revision
    rev_base_ids = [r.base_record_id for r in rev]
    unchanged = [r for r in rev if r.mutation_type == "original"]
    modified = [r for r in rev if r.mutation_type == "modified"]
    inserted = [r for r in rev if r.mutation_type == "inserted"]

    # Mutation counts should be close to 15 total mutations (40% mod, 30% del, 30% ins)
    assert len(modified) > 0, "Expected modified records"
    assert len(inserted) > 0, "Expected inserted records"
    assert len(unchanged) > 0, "Expected unchanged records"

    # All inserted records must have base_record_id == -1
    for ins in inserted:
        assert ins.base_record_id == -1

    # All modified and original records must have valid base_record_ids
    for m in modified:
        assert 0 <= m.base_record_id < 100
    for u in unchanged:
        assert 0 <= u.base_record_id < 100


def test_boundary_shift_classification_and_exclusion():
    # Case 1: Exact duplicate
    c_a = {"sha256": "hash_match", "base_record_ids": [1, 2, 3]}
    c_b = {"sha256": "hash_match", "base_record_ids": [1, 2, 3]}
    cat, j, trust, _ = classify_chunk_pair(c_a, c_b, family_a=0, family_b=0)
    assert cat == PairCategory.EXACT_DUPLICATE
    assert trust is True
    assert j == 1.0

    # Case 2: Version resemblance (J >= 0.50)
    c_c = {"sha256": "hash_diff1", "base_record_ids": [1, 2, 3, 4]}
    c_d = {"sha256": "hash_diff2", "base_record_ids": [1, 2, 3, 5]}
    # Jaccard = 3 / 5 = 0.60
    cat, j, trust, _ = classify_chunk_pair(c_c, c_d, family_a=0, family_b=0)
    assert cat == PairCategory.VERSION_RESEMBLANCE
    assert trust is True
    assert j == pytest.approx(0.60)

    # Case 3: Ambiguous boundary shift (0.10 <= J < 0.50) -> EXCLUDED
    c_e = {"sha256": "hash_diff3", "base_record_ids": [1, 10, 11, 12]}
    c_f = {"sha256": "hash_diff4", "base_record_ids": [1, 20, 21, 22]}
    # Jaccard = 1 / 7 = 0.1428
    cat, j, trust, _ = classify_chunk_pair(c_e, c_f, family_a=0, family_b=0)
    assert cat == PairCategory.AMBIGUOUS_BOUNDARY_SHIFT
    assert trust is False  # Must be excluded from supervised evaluation!

    # Case 4: Unrelated cross-family
    cat, j, trust, _ = classify_chunk_pair(c_a, c_b, family_a=0, family_b=1)
    assert cat == PairCategory.UNRELATED_CROSS_FAMILY
    assert trust is True
    assert j == 0.0


def test_family_split_isolation_and_no_leakage(tmp_path: Path):
    # Run a miniature generator test with 10 families
    summary = generate_multiversion_dataset(
        output_dir=tmp_path / "mv_test",
        num_families=10,
        records_per_base=50,
        seed=123,
    )

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    families = manifest["families"]
    chunks = manifest["chunks"]

    train_fids = {f["family_id"] for f in families if f["split"] == DatasetSplit.TRAIN.value}
    val_fids = {f["family_id"] for f in families if f["split"] == DatasetSplit.VAL.value}
    test_fids = {f["family_id"] for f in families if f["split"] == DatasetSplit.TEST.value}

    # Strict partition checks
    assert len(train_fids & val_fids) == 0, "Train and Val families overlap!"
    assert len(train_fids & test_fids) == 0, "Train and Test families overlap!"
    assert len(val_fids & test_fids) == 0, "Val and Test families overlap!"
    assert len(train_fids) + len(val_fids) + len(test_fids) == 10

    # Chunks must strictly inherit family split
    for c in chunks:
        fid = c["family_id"]
        if fid in train_fids:
            assert c["split"] == DatasetSplit.TRAIN.value
        elif fid in val_fids:
            assert c["split"] == DatasetSplit.VAL.value
        elif fid in test_fids:
            assert c["split"] == DatasetSplit.TEST.value


def test_no_duplicate_base_files_across_splits():
    manifest_path = Path("data/multi_version/processed/multiversion_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Full multiversion manifest not generated yet.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta = manifest["metadata"]

    assert meta["duplicate_base_files_across_splits"] == 0
    assert meta["total_families"] == 100
    assert meta["total_files"] == 400


def test_generated_dataset_archive_integrity():
    npz_path = Path("data/multi_version/processed/multiversion_train_data.npz")
    manifest_path = Path("data/multi_version/processed/multiversion_manifest.json")

    if not (npz_path.exists() and manifest_path.exists()):
        pytest.skip("Generated dataset not found.")

    with np.load(npz_path) as data:
        assert "context_sequences" in data
        assert "initial_features" in data
        assert "family_ids" in data
        assert "file_ids" in data
        assert "chunk_indices" in data

        cs = data["context_sequences"]
        feats = data["initial_features"]
        fids = data["family_ids"]

        assert cs.shape == (2801, 9, 128)
        assert feats.shape == (2801, 128)
        assert len(fids) == 2801
        assert cs.dtype == np.float32
        assert feats.dtype == np.float32
        assert np.isfinite(cs).all()
        assert np.isfinite(feats).all()


def test_inserted_records_sentinel_exclusion_and_lineage_regression():
    """Regression test ensuring base_record_id = -1 is excluded and cannot inflate Jaccard."""
    from src.multiversion_generator import Record

    # Chunk with base records 10, 11 and an inserted record (-1)
    rec_a = [
        Record(family_id=1, base_record_id=10, text="A" * 1024, mutation_type="original"),
        Record(family_id=1, base_record_id=11, text="B" * 1024, mutation_type="original"),
        Record(family_id=1, base_record_id=-1, text="INS_X" * 1024, mutation_type="inserted"),
    ]
    # Chunk with base records 10, 12 and a DIFFERENT inserted record (-1)
    rec_b = [
        Record(family_id=1, base_record_id=10, text="A" * 1024, mutation_type="original"),
        Record(family_id=1, base_record_id=12, text="C" * 1024, mutation_type="original"),
        Record(family_id=1, base_record_id=-1, text="INS_Y" * 1024, mutation_type="inserted"),
    ]

    _, chunks_a = map_records_to_chunks(rec_a, min_size=256, avg_size=1024, max_size=4096)
    _, chunks_b = map_records_to_chunks(rec_b, min_size=256, avg_size=1024, max_size=4096)

    # 1. Negative sentinels must NOT appear in base_record_ids
    for c in chunks_a + chunks_b:
        assert -1 not in c["base_record_ids"]
        assert all(rid >= 0 for rid in c["base_record_ids"])

    # 2. Synthetic chunks with only inserted records
    pure_ins_1 = {"sha256": "h_ins1", "base_record_ids": []}
    pure_ins_2 = {"sha256": "h_ins2", "base_record_ids": []}
    cat, j, trust, _ = classify_chunk_pair(pure_ins_1, pure_ins_2, family_a=1, family_b=1)
    assert cat == PairCategory.UNRELATED_SAME_FAMILY
    assert j == 0.0
    assert trust is True

    # 3. If -1 were falsely included, Jaccard between {10, -1} and {11, -1} would be 1/3 (0.33, boundary shift).
    # With -1 excluded, set_a={10}, set_b={11}, intersection=0 -> Jaccard=0.0 (unrelated same family).
    chunk_with_sentinel_excluded_1 = {"sha256": "h1", "base_record_ids": [10]}
    chunk_with_sentinel_excluded_2 = {"sha256": "h2", "base_record_ids": [11]}
    cat, j, trust, _ = classify_chunk_pair(chunk_with_sentinel_excluded_1, chunk_with_sentinel_excluded_2, family_a=1, family_b=1)
    assert cat == PairCategory.UNRELATED_SAME_FAMILY
    assert j == 0.0

    # 4. Modified records retain their original base_record_id
    rec_mod = [
        Record(family_id=1, base_record_id=42, text="MODIFIED" * 512, mutation_type="modified"),
    ]
    _, chunks_mod = map_records_to_chunks(rec_mod, min_size=256, avg_size=1024, max_size=4096)
    assert 42 in chunks_mod[0]["base_record_ids"]
    assert "modified" in chunks_mod[0]["mutation_types"]

