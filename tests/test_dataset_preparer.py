"""Tests for the DCLC dataset preparation pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset_preparer import (
    generate_sample_raw_data,
    process_file_chunks,
    prepare_dataset,
    load_prepared_dataset,
)


def test_generate_sample_raw_data(tmp_path: Path):
    created = generate_sample_raw_data(tmp_path)
    assert len(created) == 4
    filenames = {p.name for p in created}
    assert "sample_doc_v1.txt" in filenames
    assert "sample_doc_v2.txt" in filenames
    assert "sample_doc_identical.txt" in filenames
    assert "sample_unrelated.txt" in filenames

    # Verify files have non-zero content
    for p in created:
        assert p.stat().st_size > 0

    # Verify identical file matches v1 exactly
    v1_bytes = (tmp_path / "sample_doc_v1.txt").read_bytes()
    ident_bytes = (tmp_path / "sample_doc_identical.txt").read_bytes()
    assert v1_bytes == ident_bytes

    # Verify v2 is different from v1
    v2_bytes = (tmp_path / "sample_doc_v2.txt").read_bytes()
    assert v1_bytes != v2_bytes


def test_process_file_chunks_empty():
    res = process_file_chunks(
        data=b"",
        file_id=0,
        filename="empty.txt",
        feature_dim=128,
        context_radius=4,
    )
    assert res["chunk_count"] == 0
    assert res["context_sequences"].shape == (0, 9, 128)
    assert res["initial_features"].shape == (0, 128)
    assert len(res["fingerprints"]) == 0


def test_process_file_chunks_dimensions(tmp_path: Path):
    data = b"DCLC data preparation unit test chunking " * 1000
    res = process_file_chunks(
        data=data,
        file_id=1,
        filename="test.txt",
        min_size=512,
        avg_size=1024,
        max_size=2048,
        feature_dim=64,
        context_radius=4,
    )
    assert res["chunk_count"] > 0
    num_chunks = res["chunk_count"]

    # Context sequence length should be 2 * radius + 1 = 9
    assert res["context_sequences"].shape == (num_chunks, 9, 64)
    assert res["initial_features"].shape == (num_chunks, 64)
    assert res["chunk_indices"].shape == (num_chunks,)
    assert res["byte_offsets"].shape == (num_chunks,)
    assert res["byte_lengths"].shape == (num_chunks,)
    assert len(res["fingerprints"]) == num_chunks


def test_prepare_and_load_dataset(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    # Generate sample raw files
    generate_sample_raw_data(input_dir)

    # Run dataset preparation with small CDC sizes for testing
    summary = prepare_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        min_size=1024,
        avg_size=2048,
        max_size=4096,
        feature_dim=32,
        context_radius=3,
    )

    assert Path(summary["npz_path"]).exists()
    assert Path(summary["manifest_path"]).exists()
    assert summary["total_files"] == 4
    assert summary["total_chunks"] > 0

    # Load dataset back
    loaded = load_prepared_dataset(summary["npz_path"], summary["manifest_path"])

    assert "context_sequences" in loaded
    assert "initial_features" in loaded
    assert "file_ids" in loaded
    assert "chunk_indices" in loaded
    assert "byte_offsets" in loaded
    assert "byte_lengths" in loaded
    assert "manifest" in loaded

    num_chunks = summary["total_chunks"]
    # Check shape: (N, 2*3+1, 32) = (N, 7, 32)
    assert loaded["context_sequences"].shape == (num_chunks, 7, 32)
    assert loaded["initial_features"].shape == (num_chunks, 32)
    assert loaded["file_ids"].shape == (num_chunks,)
    assert loaded["chunk_indices"].shape == (num_chunks,)
    assert loaded["byte_offsets"].shape == (num_chunks,)
    assert loaded["byte_lengths"].shape == (num_chunks,)

    # Verify dtypes
    assert loaded["context_sequences"].dtype == np.float32
    assert loaded["initial_features"].dtype == np.float32
    assert loaded["file_ids"].dtype == np.int32
    assert loaded["chunk_indices"].dtype == np.int32


def test_file_tracking_and_chunk_order(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()

    # Create two files with distinct contents
    (input_dir / "file_a.txt").write_bytes(b"File A test content " * 300)
    (input_dir / "file_b.txt").write_bytes(b"File B different test content " * 400)

    summary = prepare_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        min_size=256,
        avg_size=512,
        max_size=1024,
        feature_dim=16,
        context_radius=2,
    )

    loaded = load_prepared_dataset(summary["npz_path"], summary["manifest_path"])
    file_ids = loaded["file_ids"]
    chunk_indices = loaded["chunk_indices"]

    # File IDs should be 0 and 1
    unique_file_ids = sorted(list(set(file_ids)))
    assert unique_file_ids == [0, 1]

    # For each file, chunk_indices should start at 0 and strictly increase
    for fid in unique_file_ids:
        indices_for_file = chunk_indices[file_ids == fid]
        expected = np.arange(len(indices_for_file), dtype=np.int32)
        np.testing.assert_array_equal(indices_for_file, expected)


def test_identical_chunks_fingerprints_match(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()

    payload = b"Consistent repeatable payload block " * 200
    (input_dir / "copy_1.txt").write_bytes(payload)
    (input_dir / "copy_2.txt").write_bytes(payload)

    summary = prepare_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        min_size=256,
        avg_size=512,
        max_size=1024,
        feature_dim=16,
        context_radius=2,
    )

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    chunks_copy_1 = [c for c in manifest["chunks"] if c["filename"] == "copy_1.txt"]
    chunks_copy_2 = [c for c in manifest["chunks"] if c["filename"] == "copy_2.txt"]

    assert len(chunks_copy_1) == len(chunks_copy_2)
    # Every chunk fingerprint should match
    for c1, c2 in zip(chunks_copy_1, chunks_copy_2):
        assert c1["sha256"] == c2["sha256"]
        assert c1["byte_length"] == c2["byte_length"]


def test_empty_directory_error(tmp_path: Path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="No files found"):
        prepare_dataset(input_dir=empty_dir, output_dir=out_dir)


def test_nonexistent_directory_error(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist"
    out_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError):
        prepare_dataset(input_dir=non_existent, output_dir=out_dir)
