"""Data preparation pipeline for DCLC training.

Inspects raw data files, chunks them using FastCDC, extracts initial
feature vectors (Algorithm 1), builds context sequences, and saves the
prepared arrays into Colab-friendly NumPy .npz and JSON manifest formats.

Research note:
The DCLC paper (Section 3.2, Section 4.1) specifies context radius K=4
(window of 2K+1=9 chunks) and feature dimension M=128.
The exact sub-chunk count K_sub and shingle parameter N in Algorithm 1,
as well as the underlying LSH family and shingle hash function, are not
fully specified in the paper. Those choices remain documented as provisional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.context_builder import build_context_sequence
from src.fastcdc_chunker import fastcdc_chunks
from src.initial_feature_extractor import extract_initial_feature


def generate_sample_raw_data(target_dir: Path | str) -> list[Path]:
    """Create a small set of synthetic text files for similarity testing.

    Includes:
      - sample_doc_v1.txt: Base version (~160 KB).
      - sample_doc_v2.txt: Slightly edited version (modified lines & insertions).
      - sample_doc_identical.txt: Bit-for-bit identical replica of v1.
      - sample_unrelated.txt: Unrelated content with completely different patterns.

    Returns:
        List of Paths to the created files.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Base repetitive pattern representing structured system logs / source code
    base_blocks = [
        f"LOG_RECORD_{i:05d}: SYSTEM_EVENT=AUTH_OK user=user_{i % 50} "
        f"session={hashlib.md5(str(i).encode()).hexdigest()} "
        f"latency={(i * 17) % 250}ms status=SUCCESS description="
        f"'Storage cluster operation acknowledged successfully for block {i}'\n"
        for i in range(1200)
    ]
    v1_content = "".join(base_blocks).encode("utf-8")

    # Version 2: Modified with insertions, deletions, and replacements
    v2_blocks = list(base_blocks)
    # Insert modified entries in the middle
    for j in range(200, 240):
        v2_blocks[j] = (
            f"LOG_RECORD_{j:05d}: SYSTEM_EVENT=AUTH_FAIL user=attacker_{j} "
            f"session=UNKNOWN latency=999ms status=FAILED description="
            f"'Security alert: unauthorized access attempt caught in block {j}'\n"
        )
    # Insert some additional lines
    v2_blocks.insert(600, "NEW_FEATURE_FLAG: DCLC_DYNAMIC_CACHE_ENABLED=TRUE\n" * 50)
    v2_content = "".join(v2_blocks).encode("utf-8")

    # Identical replica
    identical_content = v1_content

    # Unrelated content: different vocabulary, random-like scientific data
    unrelated_blocks = [
        f"GENOME_SEQ_{k:05d}: locus=chr{k % 22 + 1}:{(k * 31337) % 1000000} "
        f"ref=A alt=G score={np.sin(k):.6f} annotation=NON_CODING_VARIANT\n"
        for k in range(1500)
    ]
    unrelated_content = "".join(unrelated_blocks).encode("utf-8")

    files = [
        ("sample_doc_v1.txt", v1_content),
        ("sample_doc_v2.txt", v2_content),
        ("sample_doc_identical.txt", identical_content),
        ("sample_unrelated.txt", unrelated_content),
    ]

    created_paths = []
    for filename, content in files:
        file_path = target_path / filename
        file_path.write_bytes(content)
        created_paths.append(file_path)

    return created_paths


def process_file_chunks(
    data: bytes,
    file_id: int,
    filename: str,
    min_size: int = 8192,
    avg_size: int = 16384,
    max_size: int = 32768,
    k_sub: int = 16,
    n_shingle: int = 5,
    feature_dim: int = 128,
    context_radius: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Split bytes via FastCDC, compute initial features, and build context sequences.

    Args:
        data: Raw file bytes.
        file_id: Integer identifier for tracking file origin.
        filename: Human-readable name of the file.
        min_size: Minimum FastCDC chunk size in bytes.
        avg_size: Average FastCDC chunk size in bytes.
        max_size: Maximum FastCDC chunk size in bytes.
        k_sub: Number of sub-chunks per chunk in Algorithm 1.
        n_shingle: Shingle window parameter N in Algorithm 1.
        feature_dim: Initial feature dimension M.
        context_radius: Number of surrounding chunks on each side (K_context in paper).
        seed: Random seed for deterministic LSH projections.

    Returns:
        Dictionary containing extracted chunk arrays and chunk metadata.
    """
    if not data:
        return {
            "file_id": file_id,
            "filename": filename,
            "chunk_count": 0,
            "context_sequences": np.empty((0, 2 * context_radius + 1, feature_dim), dtype=np.float32),
            "initial_features": np.empty((0, feature_dim), dtype=np.float32),
            "chunk_indices": np.empty((0,), dtype=np.int32),
            "byte_offsets": np.empty((0,), dtype=np.int64),
            "byte_lengths": np.empty((0,), dtype=np.int32),
            "fingerprints": [],
        }

    raw_chunks = list(fastcdc_chunks(data, min_size=min_size, avg_size=avg_size, max_size=max_size))
    chunk_count = len(raw_chunks)

    initial_features_list = []
    byte_offsets = []
    byte_lengths = []
    fingerprints = []

    current_offset = 0
    for chunk in raw_chunks:
        c_len = len(chunk)
        byte_offsets.append(current_offset)
        byte_lengths.append(c_len)
        current_offset += c_len

        # Cryptographic fingerprint for deduplication / negative sampling
        fp = hashlib.sha256(chunk).hexdigest()
        fingerprints.append(fp)

        # Initial feature vector e_i (Algorithm 1)
        e_i = extract_initial_feature(
            chunk,
            k=k_sub,
            n=n_shingle,
            m=feature_dim,
            seed=seed,
        )
        initial_features_list.append(e_i)

    # Build context sequences: each chunk gets a window of 2 * radius + 1 features
    context_sequences = []
    for idx in range(chunk_count):
        ctx_seq = build_context_sequence(
            initial_features_list,
            center_index=idx,
            radius=context_radius,
        )
        context_sequences.append(ctx_seq)

    return {
        "file_id": file_id,
        "filename": filename,
        "chunk_count": chunk_count,
        "context_sequences": np.stack(context_sequences).astype(np.float32)
        if context_sequences
        else np.empty((0, 2 * context_radius + 1, feature_dim), dtype=np.float32),
        "initial_features": np.stack(initial_features_list).astype(np.float32)
        if initial_features_list
        else np.empty((0, feature_dim), dtype=np.float32),
        "chunk_indices": np.arange(chunk_count, dtype=np.int32),
        "byte_offsets": np.array(byte_offsets, dtype=np.int64),
        "byte_lengths": np.array(byte_lengths, dtype=np.int32),
        "fingerprints": fingerprints,
    }


def prepare_dataset(
    input_dir: Path | str,
    output_dir: Path | str,
    min_size: int = 8192,
    avg_size: int = 16384,
    max_size: int = 32768,
    k_sub: int = 16,
    n_shingle: int = 5,
    feature_dim: int = 128,
    context_radius: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Process all files in input_dir and save prepared training data.

    Saves:
      - {output_dir}/dclc_train_data.npz: Compressed arrays for training.
      - {output_dir}/dataset_manifest.json: JSON metadata, file tracking, fingerprints.

    Returns:
        Summary dictionary of the prepared dataset.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    # Gather files deterministically
    file_paths = sorted([p for p in input_path.iterdir() if p.is_file()])
    if not file_paths:
        raise ValueError(f"No files found in input directory: {input_path}")

    all_context_seqs = []
    all_initial_feats = []
    all_file_ids = []
    all_chunk_indices = []
    all_byte_offsets = []
    all_byte_lengths = []

    files_manifest = []
    chunk_manifest = []
    global_chunk_counter = 0

    for file_id, fpath in enumerate(file_paths):
        raw_bytes = fpath.read_bytes()
        res = process_file_chunks(
            raw_bytes,
            file_id=file_id,
            filename=fpath.name,
            min_size=min_size,
            avg_size=avg_size,
            max_size=max_size,
            k_sub=k_sub,
            n_shingle=n_shingle,
            feature_dim=feature_dim,
            context_radius=context_radius,
            seed=seed,
        )

        n_chunks = res["chunk_count"]
        files_manifest.append({
            "file_id": file_id,
            "filename": fpath.name,
            "size_bytes": len(raw_bytes),
            "chunk_count": n_chunks,
        })

        if n_chunks > 0:
            all_context_seqs.append(res["context_sequences"])
            all_initial_feats.append(res["initial_features"])
            all_file_ids.append(np.full(n_chunks, file_id, dtype=np.int32))
            all_chunk_indices.append(res["chunk_indices"])
            all_byte_offsets.append(res["byte_offsets"])
            all_byte_lengths.append(res["byte_lengths"])

            for idx in range(n_chunks):
                chunk_manifest.append({
                    "global_chunk_id": global_chunk_counter,
                    "file_id": file_id,
                    "filename": fpath.name,
                    "chunk_index": int(res["chunk_indices"][idx]),
                    "byte_offset": int(res["byte_offsets"][idx]),
                    "byte_length": int(res["byte_lengths"][idx]),
                    "sha256": res["fingerprints"][idx],
                })
                global_chunk_counter += 1

    # Concatenate all chunk arrays
    if all_context_seqs:
        merged_context_seqs = np.concatenate(all_context_seqs, axis=0)
        merged_initial_feats = np.concatenate(all_initial_feats, axis=0)
        merged_file_ids = np.concatenate(all_file_ids, axis=0)
        merged_chunk_indices = np.concatenate(all_chunk_indices, axis=0)
        merged_byte_offsets = np.concatenate(all_byte_offsets, axis=0)
        merged_byte_lengths = np.concatenate(all_byte_lengths, axis=0)
    else:
        merged_context_seqs = np.empty((0, 2 * context_radius + 1, feature_dim), dtype=np.float32)
        merged_initial_feats = np.empty((0, feature_dim), dtype=np.float32)
        merged_file_ids = np.empty((0,), dtype=np.int32)
        merged_chunk_indices = np.empty((0,), dtype=np.int32)
        merged_byte_offsets = np.empty((0,), dtype=np.int64)
        merged_byte_lengths = np.empty((0,), dtype=np.int32)

    # Save to compressed NumPy archive
    npz_path = output_path / "dclc_train_data.npz"
    np.savez_compressed(
        npz_path,
        context_sequences=merged_context_seqs,
        initial_features=merged_initial_feats,
        file_ids=merged_file_ids,
        chunk_indices=merged_chunk_indices,
        byte_offsets=merged_byte_offsets,
        byte_lengths=merged_byte_lengths,
    )

    # Save human-readable manifest
    manifest_path = output_path / "dataset_manifest.json"
    manifest_data = {
        "metadata": {
            "total_files": len(files_manifest),
            "total_chunks": int(merged_context_seqs.shape[0]),
            "feature_dim": feature_dim,
            "context_radius": context_radius,
            "context_sequence_len": 2 * context_radius + 1,
            "fastcdc": {
                "min_size": min_size,
                "avg_size": avg_size,
                "max_size": max_size,
            },
            "algorithm1_params": {
                "k_sub": k_sub,
                "n_shingle": n_shingle,
                "lsh_choice": "Random projection (provisional)",
                "shingle_hash_choice": "SHAKE-256 (provisional)",
            },
        },
        "files": files_manifest,
        "chunks": chunk_manifest,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return {
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "total_files": len(files_manifest),
        "total_chunks": int(merged_context_seqs.shape[0]),
        "context_sequences_shape": merged_context_seqs.shape,
        "initial_features_shape": merged_initial_feats.shape,
    }


def load_prepared_dataset(
    npz_path: Path | str,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load prepared training data archive and optional manifest into memory.

    Convenient loader for Google Colab and training notebooks.
    """
    npz_file = Path(npz_path)
    if not npz_file.exists():
        raise FileNotFoundError(f"Prepared data archive not found: {npz_file}")

    with np.load(npz_file) as loaded:
        dataset = {
            "context_sequences": loaded["context_sequences"],
            "initial_features": loaded["initial_features"],
            "file_ids": loaded["file_ids"],
            "chunk_indices": loaded["chunk_indices"],
            "byte_offsets": loaded["byte_offsets"],
            "byte_lengths": loaded["byte_lengths"],
        }

    if manifest_path is not None:
        m_file = Path(manifest_path)
        if m_file.exists():
            dataset["manifest"] = json.loads(m_file.read_text(encoding="utf-8"))

    return dataset


def main() -> None:
    """CLI entrypoint for DCLC dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare DCLC training dataset from raw files.")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Path to raw files directory")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to save processed data")
    parser.add_argument("--generate-samples", action="store_true", help="Generate synthetic sample files first")
    parser.add_argument("--min-size", type=int, default=8192, help="FastCDC minimum chunk size")
    parser.add_argument("--avg-size", type=int, default=16384, help="FastCDC average chunk size")
    parser.add_argument("--max-size", type=int, default=32768, help="FastCDC maximum chunk size")
    parser.add_argument("--k-sub", type=int, default=16, help="Algorithm 1 sub-chunks count")
    parser.add_argument("--n-shingle", type=int, default=5, help="Algorithm 1 shingle window size")
    parser.add_argument("--dim", type=int, default=128, help="Feature dimension M")
    parser.add_argument("--radius", type=int, default=4, help="Context radius K (paper K=4)")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if args.generate_samples or not input_path.exists() or not any(input_path.iterdir()):
        print(f"Generating synthetic sample files in: {input_path}")
        created = generate_sample_raw_data(input_path)
        print(f"Created {len(created)} sample files.")

    print(f"Processing raw files from '{input_path}' into '{output_path}'...")
    summary = prepare_dataset(
        input_dir=input_path,
        output_dir=output_path,
        min_size=args.min_size,
        avg_size=args.avg_size,
        max_size=args.max_size,
        k_sub=args.k_sub,
        n_shingle=args.n_shingle,
        feature_dim=args.dim,
        context_radius=args.radius,
    )

    print("Data preparation complete!")
    print(f"  Total files: {summary['total_files']}")
    print(f"  Total chunks: {summary['total_chunks']}")
    print(f"  Context sequences shape: {summary['context_sequences_shape']}")
    print(f"  Initial features shape: {summary['initial_features_shape']}")
    print(f"  Saved archive: {summary['npz_path']}")
    print(f"  Saved manifest: {summary['manifest_path']}")


if __name__ == "__main__":
    main()
