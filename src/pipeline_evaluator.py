"""End-to-End Pipeline Evaluator for DCLC Deduplication and Delta Encoding (Phase 4C).

Evaluates and compares four storage architectures on the held-out Test split (families 085-099):
1. Raw Storage (no deduplication, no compression).
2. Standalone Compression (per-chunk zlib compression).
3. Exact SHA-256 Deduplication + Standalone Compression.
4. Full DCLC Pipeline: Exact Hash Deduplication + Label-Free Candidate Indexing
   + Frozen Neural Similarity Scoring (threshold 0.70) + Verified Delta Encoding.

Candidate Indexers Compared:
- Stream Locality (window W=32)
- Random-Projection LSH (L=8, B=6 on 128-dim initial features)
- Embedding Cosine k-NN (top_k=20 on 128-dim Transformer embeddings)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from src.candidate_indexer import (
    BaseCandidateIndexer,
    EmbeddingCosineKNNIndexer,
    HierarchicalCacheANNIndexer,
    RandomProjectionLSHIndexer,
    StreamLocalityIndexer,
)
from src.delta_encoder import (
    compress_standalone,
    decode_delta,
    decode_mrd_group,
    decode_mrd_single,
    evaluate_and_encode_delta,
    evaluate_and_encode_mrd,
)
from src.evaluate import load_model_and_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_chunk_raw_bytes(
    manifest_chunk: dict[str, Any],
    manifest_files: dict[int, dict[str, Any]],
    raw_data_dir: Path | str = "data/multi_version/raw",
) -> bytes:
    """Read the exact byte slice of a chunk from its source raw file."""
    file_meta = manifest_files[manifest_chunk["file_id"]]
    family_id = file_meta["family_id"]
    filename = file_meta["filename"]

    file_path = Path(raw_data_dir) / f"family_{family_id:03d}" / filename
    with open(file_path, "rb") as f:
        f.seek(manifest_chunk["byte_offset"])
        chunk_bytes = f.read(manifest_chunk["byte_length"])

    return chunk_bytes


def run_pipeline_evaluation(
    manifest_path: str | Path = "data/multi_version/processed/multiversion_manifest.json",
    data_npz_path: str | Path = "data/multi_version/processed/multiversion_train_data.npz",
    checkpoint_path: str | Path = "checkpoints/dclc_encoder_version_aware.pt",
    resemblance_threshold: float = 0.70,
    device: str = "cpu",
    enable_mrd: bool = False,
    mrd_window_size: int = 3,
    mrd_max_refs: int = 3,
) -> dict[str, Any]:
    """Run end-to-end comparative evaluation across the 464 test chunks."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    files_map = {f["file_id"]: f for f in manifest["files"]}
    all_chunks = manifest["chunks"]
    test_chunks = [c for c in all_chunks if c["split"] == "test"]
    n_test = len(test_chunks)

    test_cids = [c["global_chunk_id"] for c in test_chunks]

    logger.info(f"Loaded {n_test} test chunks (families 085-099).")

    # Load arrays and model
    npz_data = np.load(data_npz_path)
    test_seqs = npz_data["context_sequences"][test_cids].astype(np.float32)
    test_feats = npz_data["initial_features"][test_cids].astype(np.float32)

    model = load_model_and_checkpoint(checkpoint_path, device=device)
    with torch.no_grad():
        test_embs = model(torch.from_numpy(test_seqs).float().to(device)).cpu().numpy()

    # Pre-load all raw bytes for test chunks
    test_raw_bytes = [load_chunk_raw_bytes(c, files_map) for c in test_chunks]

    # Pre-compute ground-truth resemblance pairs for candidate recall accounting
    ground_truth_resem_pairs: set[tuple[int, int]] = set()
    exact_duplicate_target_pairs: set[tuple[int, int]] = set()

    for i in range(n_test):
        ca = test_chunks[i]
        sa = set(ca["base_record_ids"])
        for j in range(i + 1, n_test):
            cb = test_chunks[j]
            if ca["family_id"] != cb["family_id"] or ca["file_id"] == cb["file_id"]:
                continue
            if ca["sha256"] == cb["sha256"]:
                exact_duplicate_target_pairs.add((i, j))
                exact_duplicate_target_pairs.add((j, i))
                continue
            sb = set(cb["base_record_ids"])
            if (sa or sb) and len(sa & sb) / len(sa | sb) >= 0.50:
                ground_truth_resem_pairs.add((i, j))
                ground_truth_resem_pairs.add((j, i))

    total_resem_targets = len(ground_truth_resem_pairs) // 2
    logger.info(f"Test resemblance target pairs: {total_resem_targets}")

    # =========================================================================
    # SYSTEM 0: Raw Storage (No compression, no dedup)
    # =========================================================================
    total_raw_bytes = sum(len(b) for b in test_raw_bytes)

    # =========================================================================
    # SYSTEM 1: Independent Per-Chunk Compression (No dedup)
    # =========================================================================
    sys1_compressed_chunks = [compress_standalone(b) for b in test_raw_bytes]
    total_sys1_bytes = sum(len(c) for c in sys1_compressed_chunks)

    # =========================================================================
    # SYSTEM 2: Exact SHA-256 Deduplication + Standalone Compression
    # =========================================================================
    seen_fingerprints_sys2: dict[str, int] = {}
    sys2_stored_bytes = 0
    sys2_exact_dups_count = 0
    pointer_overhead_sys2 = 0

    for idx, (chunk_meta, raw_b, comp_b) in enumerate(zip(test_chunks, test_raw_bytes, sys1_compressed_chunks)):
        fp = chunk_meta["sha256"]
        if fp in seen_fingerprints_sys2:
            # Exact duplicate found: store 32-byte pointer reference
            sys2_exact_dups_count += 1
            pointer_overhead_sys2 += 32
        else:
            seen_fingerprints_sys2[fp] = idx
            sys2_stored_bytes += len(comp_b)

    total_sys2_stored = sys2_stored_bytes + pointer_overhead_sys2

    # =========================================================================
    # SYSTEM 3: Full DCLC Pipeline (Exact Dedup + Candidate Index + Neural Scoring + Delta)
    # Compare 3 candidate generation strategies
    # =========================================================================
    indexers: dict[str, BaseCandidateIndexer] = {
        "Stream_Locality_W32": StreamLocalityIndexer(window_size=32),
        "RP_LSH_L8_B6": RandomProjectionLSHIndexer(dim=128, num_tables=8, num_bits=6, seed=42),
        "Embedding_KNN_K20": EmbeddingCosineKNNIndexer(top_k=20),
        # Phase A: hierarchical cache + ANN (Section 3.3)
        "Hierarchical_Cache64_ANN20": HierarchicalCacheANNIndexer(
            cache_capacity=64,
            similarity_threshold=0.70,
            top_k=20,
        ),
    }

    dclc_results: dict[str, Any] = {}

    for idx_name, indexer in indexers.items():
        indexer.reset()
        stored_raw_bytes: dict[int, bytes] = {}
        stored_embeddings: dict[int, np.ndarray] = {}
        exact_hash_table: dict[str, int] = {}

        dclc_stored_bytes = 0
        exact_dups_count = 0
        deltas_accepted_count = 0
        deltas_rejected_count = 0
        standalone_stored_count = 0
        reconstruction_verified_count = 0

        total_candidates_examined = 0
        max_candidate_pool = 0
        recalled_resemblance_pairs: set[tuple[int, int]] = set()

        time_query_total = 0.0
        time_scoring_total = 0.0
        time_delta_total = 0.0
        start_eval_time = time.perf_counter()

        for i in range(n_test):
            target_meta = test_chunks[i]
            target_bytes = test_raw_bytes[i]
            target_comp = sys1_compressed_chunks[i]
            target_feat = test_feats[i]
            target_emb = test_embs[i]
            fp = target_meta["sha256"]

            # Step 1: Exact Duplicate Detection (O(1) hash check)
            if fp in exact_hash_table:
                exact_dups_count += 1
                dclc_stored_bytes += 32  # 32-byte hash pointer overhead
                continue

            # Step 2: Non-identical chunk -> Query candidate index
            t_q0 = time.perf_counter()
            candidate_ids = indexer.query_candidates(
                query_id=i,
                initial_feature=target_feat,
                embedding=target_emb,
                sha256=fp,
            )
            time_query_total += (time.perf_counter() - t_q0)

            total_candidates_examined += len(candidate_ids)
            max_candidate_pool = max(max_candidate_pool, len(candidate_ids))

            # Record recall of ground truth pairs
            for c_id in candidate_ids:
                pair = tuple(sorted((i, c_id)))
                if (i, c_id) in ground_truth_resem_pairs:
                    recalled_resemblance_pairs.add(pair)

            # Step 3: Neural Similarity Scoring on Candidates
            t_s0 = time.perf_counter()
            best_cand_id = None
            best_sim = -1.0

            for c_id in candidate_ids:
                if c_id not in stored_embeddings:
                    continue
                cand_emb = stored_embeddings[c_id]
                sim = float(target_emb @ cand_emb)
                if sim >= resemblance_threshold and sim > best_sim:
                    best_sim = sim
                    best_cand_id = c_id
            time_scoring_total += (time.perf_counter() - t_s0)

            # Step 4: Delta Encoding Verification
            delta_applied = False
            if best_cand_id is not None:
                t_d0 = time.perf_counter()
                base_bytes = stored_raw_bytes[best_cand_id]
                delta_res = evaluate_and_encode_delta(base_bytes, target_bytes, compression_level=6)

                if delta_res.is_accepted:
                    # Assert byte-for-byte correctness immediately
                    reconstructed = decode_delta(base_bytes, delta_res.delta_container)
                    assert reconstructed == target_bytes, "FATAL: Delta reconstruction mismatch!"
                    reconstruction_verified_count += 1

                    deltas_accepted_count += 1
                    dclc_stored_bytes += delta_res.net_stored_size
                    delta_applied = True
                else:
                    deltas_rejected_count += 1
                time_delta_total += (time.perf_counter() - t_d0)

            if not delta_applied:
                # Store standalone compressed
                standalone_stored_count += 1
                dclc_stored_bytes += len(target_comp)

            # Register chunk into storage indices
            stored_raw_bytes[i] = target_bytes
            stored_embeddings[i] = target_emb
            exact_hash_table[fp] = i
            indexer.add_chunk(i, target_feat, target_emb, fp)

        total_runtime_pipeline = time.perf_counter() - start_eval_time

        recalled_unique_count = len(recalled_resemblance_pairs)
        cand_recall = (
            float(recalled_unique_count / total_resem_targets) if total_resem_targets > 0 else 0.0
        )
        avg_candidate_pool = float(total_candidates_examined / n_test)
        total_possible_comparisons = n_test * (n_test - 1) // 2
        reduction_rate = float(
            1.0 - (total_candidates_examined / total_possible_comparisons)
        )

        dclc_savings_bytes = total_raw_bytes - dclc_stored_bytes
        dclc_savings_pct = float(dclc_savings_bytes / total_raw_bytes * 100)
        compression_ratio = float(total_raw_bytes / dclc_stored_bytes) if dclc_stored_bytes > 0 else 0.0

        dclc_results[idx_name] = {
            "stored_bytes": dclc_stored_bytes,
            "savings_bytes": dclc_savings_bytes,
            "savings_pct": dclc_savings_pct,
            "compression_ratio": compression_ratio,
            "candidate_recall": cand_recall,
            "recalled_pairs_count": recalled_unique_count,
            "total_resem_targets": total_resem_targets,
            "avg_candidate_pool": avg_candidate_pool,
            "max_candidate_pool": max_candidate_pool,
            "comparison_reduction_rate": reduction_rate * 100,
            "exact_duplicates_count": exact_dups_count,
            "deltas_accepted": deltas_accepted_count,
            "deltas_rejected": deltas_rejected_count,
            "standalone_stored": standalone_stored_count,
            "reconstruction_verified_count": reconstruction_verified_count,
            "reconstruction_success_rate": 100.0 if deltas_accepted_count == reconstruction_verified_count else 0.0,
            "timing_benchmarks": {
                "total_pipeline_time_seconds": total_runtime_pipeline,
                "candidate_query_time_seconds": time_query_total,
                "scoring_time_seconds": time_scoring_total,
                "delta_time_seconds": time_delta_total,
                "query_latency_ms_per_chunk": (time_query_total / n_test) * 1000.0,
                "scoring_latency_ms_per_chunk": (time_scoring_total / n_test) * 1000.0,
                "delta_latency_ms_per_chunk": (time_delta_total / n_test) * 1000.0,
            },
        }

    final_results = {
        "dataset_scope": {
            "split": "test",
            "families": "085-099",
            "total_chunks": n_test,
            "total_logical_raw_bytes": total_raw_bytes,
            "exact_duplicate_targets": len(exact_duplicate_target_pairs) // 2,
            "resemblance_targets": total_resem_targets,
        },
        "system_0_raw": {
            "stored_bytes": total_raw_bytes,
            "savings_pct": 0.0,
            "compression_ratio": 1.0,
        },
        "system_1_standalone_compression": {
            "stored_bytes": total_sys1_bytes,
            "savings_bytes": total_raw_bytes - total_sys1_bytes,
            "savings_pct": float((total_raw_bytes - total_sys1_bytes) / total_raw_bytes * 100),
            "compression_ratio": float(total_raw_bytes / total_sys1_bytes),
        },
        "system_2_exact_dedup": {
            "stored_bytes": total_sys2_stored,
            "savings_bytes": total_raw_bytes - total_sys2_stored,
            "savings_pct": float((total_raw_bytes - total_sys2_stored) / total_raw_bytes * 100),
            "compression_ratio": float(total_raw_bytes / total_sys2_stored),
            "exact_duplicates_eliminated": sys2_exact_dups_count,
        },
        "system_3_full_dclc": dclc_results,
    }

    if enable_mrd:
        logger.info("Running Multi-Reference Delta (MRD) evaluation...")
        final_results["system_4_dclc_mrd"] = run_mrd_evaluation(
            manifest_path=manifest_path,
            data_npz_path=data_npz_path,
            checkpoint_path=checkpoint_path,
            resemblance_threshold=resemblance_threshold,
            window_size=mrd_window_size,
            max_refs=mrd_max_refs,
            device=device,
        )

    return final_results


def run_mrd_evaluation(
    manifest_path: str | Path = "data/multi_version/processed/multiversion_manifest.json",
    data_npz_path: str | Path = "data/multi_version/processed/multiversion_train_data.npz",
    checkpoint_path: str | Path = "checkpoints/dclc_encoder_version_aware.pt",
    resemblance_threshold: float = 0.70,
    window_size: int = 3,
    max_refs: int = 3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare Single-Reference (SR) vs Multi-Reference Delta (MRD) on identical test data.

    Evaluates:
      - Compression Ratio (DCR)
      - Stored bytes & Savings %
      - Encoding time (seconds)
      - Group restore throughput (MG) in MB/s
      - Single-target restore throughput (MS) in MB/s
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    files_map = {f["file_id"]: f for f in manifest["files"]}
    all_chunks = manifest["chunks"]
    test_chunks = [c for c in all_chunks if c["split"] == "test"]
    n_test = len(test_chunks)
    test_cids = [c["global_chunk_id"] for c in test_chunks]

    npz_data = np.load(data_npz_path)
    test_seqs = npz_data["context_sequences"][test_cids].astype(np.float32)
    test_feats = npz_data["initial_features"][test_cids].astype(np.float32)

    model = load_model_and_checkpoint(checkpoint_path, device=device)
    with torch.no_grad():
        test_embs = model(torch.from_numpy(test_seqs).float().to(device)).cpu().numpy()

    test_raw_bytes = [load_chunk_raw_bytes(c, files_map) for c in test_chunks]
    total_raw_bytes = sum(len(b) for b in test_raw_bytes)

    # -------------------------------------------------------------------------
    # Evaluation A: Single-Reference (SR) Baseline
    # -------------------------------------------------------------------------
    sr_indexer = HierarchicalCacheANNIndexer(
        cache_capacity=64,
        similarity_threshold=resemblance_threshold,
        top_k=5,
    )
    sr_stored_raw: dict[int, bytes] = {}
    sr_exact_hash: dict[str, int] = {}
    sr_stored_bytes = 0
    sr_accepted = 0
    sr_rejected = 0

    sr_encode_time = 0.0
    sr_restore_time = 0.0
    sr_restored_bytes = 0

    for i in range(n_test):
        t_meta = test_chunks[i]
        t_bytes = test_raw_bytes[i]
        t_feat = test_feats[i]
        t_emb = test_embs[i]
        fp = t_meta["sha256"]

        if fp in sr_exact_hash:
            sr_stored_bytes += 32
            continue

        cands = sr_indexer.query_candidates(
            query_id=i, initial_feature=t_feat, embedding=t_emb, sha256=fp
        )

        delta_done = False
        if cands:
            best_cid = cands[0]
            if best_cid in sr_stored_raw:
                b_bytes = sr_stored_raw[best_cid]
                t0 = time.perf_counter()
                d_res = evaluate_and_encode_delta(b_bytes, t_bytes, compression_level=6)
                sr_encode_time += (time.perf_counter() - t0)

                if d_res.is_accepted and d_res.delta_container:
                    sr_accepted += 1
                    sr_stored_bytes += d_res.net_stored_size
                    delta_done = True

                    # Measure SR restore throughput
                    t_r0 = time.perf_counter()
                    rec = decode_delta(b_bytes, d_res.delta_container)
                    sr_restore_time += (time.perf_counter() - t_r0)
                    sr_restored_bytes += len(rec)
                    assert rec == t_bytes
                else:
                    sr_rejected += 1

        if not delta_done:
            c_comp = compress_standalone(t_bytes, level=6)
            sr_stored_bytes += len(c_comp)

        sr_stored_raw[i] = t_bytes
        sr_exact_hash[fp] = i
        sr_indexer.add_chunk(i, t_feat, t_emb, fp)

    sr_restore_mb_s = (
        (sr_restored_bytes / (1024 * 1024)) / sr_restore_time if sr_restore_time > 0 else 0.0
    )
    sr_cr = float(total_raw_bytes / sr_stored_bytes) if sr_stored_bytes > 0 else 0.0

    # -------------------------------------------------------------------------
    # Evaluation B: Multi-Reference Delta (MRD) Mode
    # -------------------------------------------------------------------------
    mrd_indexer = HierarchicalCacheANNIndexer(
        cache_capacity=64,
        similarity_threshold=resemblance_threshold,
        top_k=max_refs * 2,
    )
    mrd_stored_raw: dict[int, bytes] = {}
    mrd_exact_hash: dict[str, int] = {}
    mrd_stored_bytes = 0
    mrd_groups_accepted = 0
    mrd_groups_rejected = 0

    mrd_encode_time = 0.0
    mg_restore_time = 0.0
    mg_restored_bytes = 0
    ms_restore_time = 0.0
    ms_restored_bytes = 0

    idx = 0
    while idx < n_test:
        window_end = min(idx + window_size, n_test)
        window_indices = list(range(idx, window_end))
        target_group = [test_raw_bytes[j] for j in window_indices]
        group_raw_len = sum(len(c) for c in target_group)

        # Check for candidates for each target in window
        ref_cand_set: list[int] = []
        for j in window_indices:
            cands = mrd_indexer.query_candidates(
                query_id=j,
                initial_feature=test_feats[j],
                embedding=test_embs[j],
                sha256=test_chunks[j]["sha256"],
            )
            for c in cands:
                if c in mrd_stored_raw and c not in ref_cand_set:
                    ref_cand_set.append(c)
                if len(ref_cand_set) >= max_refs:
                    break
            if len(ref_cand_set) >= max_refs:
                break

        mrd_done = False
        if ref_cand_set:
            ref_group = [mrd_stored_raw[c] for c in ref_cand_set]
            t_m0 = time.perf_counter()
            mrd_res = evaluate_and_encode_mrd(ref_group, target_group, compression_level=6)
            mrd_encode_time += (time.perf_counter() - t_m0)

            if mrd_res.is_accepted and mrd_res.mrd_container:
                mrd_groups_accepted += 1
                mrd_stored_bytes += mrd_res.net_stored_size
                mrd_done = True

                # 1. Benchmark MG: Group-level restore
                t_mg0 = time.perf_counter()
                rec_group = decode_mrd_group(ref_group, mrd_res.mrd_container)
                mg_restore_time += (time.perf_counter() - t_mg0)
                mg_restored_bytes += sum(len(c) for c in rec_group)
                assert rec_group == target_group

                # 2. Benchmark MS: Single-target restore
                for sub_idx in range(len(target_group)):
                    t_ms0 = time.perf_counter()
                    rec_single = decode_mrd_single(ref_group, mrd_res.mrd_container, sub_idx)
                    ms_restore_time += (time.perf_counter() - t_ms0)
                    ms_restored_bytes += len(rec_single)
                    assert rec_single == target_group[sub_idx]
            else:
                mrd_groups_rejected += 1

        if not mrd_done:
            for j in window_indices:
                c_bytes = test_raw_bytes[j]
                fp = test_chunks[j]["sha256"]
                if fp in mrd_exact_hash:
                    mrd_stored_bytes += 32
                else:
                    c_comp = compress_standalone(c_bytes, level=6)
                    mrd_stored_bytes += len(c_comp)

        # Register all window chunks into storage and indexer
        for j in window_indices:
            mrd_stored_raw[j] = test_raw_bytes[j]
            mrd_exact_hash[test_chunks[j]["sha256"]] = j
            mrd_indexer.add_chunk(j, test_feats[j], test_embs[j], test_chunks[j]["sha256"])

        idx = window_end

    mg_restore_mb_s = (
        (mg_restored_bytes / (1024 * 1024)) / mg_restore_time if mg_restore_time > 0 else 0.0
    )
    ms_restore_mb_s = (
        (ms_restored_bytes / (1024 * 1024)) / ms_restore_time if ms_restore_time > 0 else 0.0
    )
    mrd_cr = float(total_raw_bytes / mrd_stored_bytes) if mrd_stored_bytes > 0 else 0.0

    mg_vs_sr_gain = (
        ((mg_restore_mb_s - sr_restore_mb_s) / sr_restore_mb_s * 100.0)
        if sr_restore_mb_s > 0
        else 0.0
    )
    ms_vs_sr_diff = (
        ((ms_restore_mb_s - sr_restore_mb_s) / sr_restore_mb_s * 100.0)
        if sr_restore_mb_s > 0
        else 0.0
    )

    return {
        "parameters": {
            "window_size": window_size,
            "max_refs": max_refs,
            "resemblance_threshold": resemblance_threshold,
        },
        "single_reference_sr": {
            "stored_bytes": sr_stored_bytes,
            "compression_ratio": round(sr_cr, 3),
            "deltas_accepted": sr_accepted,
            "deltas_rejected": sr_rejected,
            "encoding_time_seconds": round(sr_encode_time, 4),
            "restoration_throughput_mb_s": round(sr_restore_mb_s, 2),
        },
        "multi_reference_mrd": {
            "stored_bytes": mrd_stored_bytes,
            "compression_ratio": round(mrd_cr, 3),
            "mrd_groups_accepted": mrd_groups_accepted,
            "mrd_groups_rejected": mrd_groups_rejected,
            "encoding_time_seconds": round(mrd_encode_time, 4),
            "mg_group_restore_throughput_mb_s": round(mg_restore_mb_s, 2),
            "ms_single_restore_throughput_mb_s": round(ms_restore_mb_s, 2),
        },
        "restoration_throughput_comparison": {
            "mg_group_restore_vs_sr_pct": round(mg_vs_sr_gain, 2),
            "ms_single_restore_vs_sr_pct": round(ms_vs_sr_diff, 2),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end DCLC storage pipeline evaluation.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dclc_encoder_version_aware.pt")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--mrd", action="store_true", help="Enable Multi-Reference Delta evaluation.")
    args = parser.parse_args()

    results = run_pipeline_evaluation(
        checkpoint_path=args.checkpoint,
        resemblance_threshold=args.threshold,
        enable_mrd=args.mrd,
    )
    print(json.dumps(results, indent=2))
