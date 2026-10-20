"""
scripts/measure_precision_modes.py
-----------------------------------
Generates empirical precision comparison metrics across FP32, FP16, and INT8 (SQ8)
for the DCLC Phase 7D evidence report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.rag.config import get_embedding_precision, validate_precision
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.version_mapper import VersionRegistry

def run_precision_benchmark(corpus_size: int = 500, n_queries: int = 100):
    print(f"--- Running Phase 7D Precision Benchmark (Corpus: {corpus_size} chunks, Queries: {n_queries}) ---")

    # Generate synthetic text chunks
    np.random.seed(42)
    sentences = [
        "Distributed database replication protocols guarantee strong consistency across multiple cloud regions.",
        "Neural network weight quantization reduces cache thrashing and memory bus bandwidth saturation.",
        "FastCDC variable-size content-defined chunking calculates Rabin rolling fingerprints dynamically.",
        "Content-addressable storage keys version deltas via cryptographic SHA-256 block identifiers.",
        "Hybrid retrieval combines sparse BM25 inverted lexical term frequencies with dense dot product similarity.",
        "Strict version isolation filters retrieval candidate sets before evidence packets reach the generator.",
        "Hardware accelerated matrix multiplication benefits from half precision IEEE float16 tensor cores.",
        "Symmetric 8-bit scalar quantization maps coordinate intervals to signed integer ranges between -127 and 127.",
        "Garbage collection overhead and memory fragmentation degrade tail retrieval latency under high concurrency.",
        "Grounding verification cross-references generated citation offsets against original source bytes."
    ]

    # Create dummy registry
    reg = VersionRegistry()
    tmp_dir = Path("data/bench_precision_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for i in range(corpus_size):
        s = sentences[i % len(sentences)]
        chunks.append(f"Chunk {i}: {s} Additional padding context tokens {i * 7}.")

    full_text = "\n\n".join(chunks)
    p = tmp_dir / "bench_doc.txt"
    p.write_text(full_text, encoding="utf-8")
    v = reg.ingest_document(p, min_size=64, avg_size=256, max_size=1024)

    c_idx = ChunkVersionIndex()
    c_idx.build(reg)

    unique_chunks = len(c_idx.all_chunk_ids())
    print(f"Indexed {unique_chunks} unique chunks.")

    queries = [
        "What is the memory bandwidth saturation for neural network quantization?",
        "How does FastCDC calculate Rabin rolling fingerprints dynamically?",
        "Explain hybrid retrieval combining BM25 and dense dot product similarity.",
        "What guarantees strong consistency across multiple cloud regions?",
        "How are cryptographic SHA-256 block identifiers used in version deltas?"
    ] * (n_queries // 5)

    results = {}
    stores = {}

    for prec in ["float32", "float16", "int8"]:
        store = EmbeddingStore(precision=prec)
        t0 = time.perf_counter()
        store.build(c_idx, batch_size=32)
        build_time = time.perf_counter() - t0

        stores[prec] = store

        # Latency measurements
        query_latencies = []
        scores_by_query = []
        top1_by_query = []

        for q in queries:
            q_vec = store.query_vector(q)
            t_start = time.perf_counter()
            scores = store.cosine_scores(q_vec)
            lat_us = (time.perf_counter() - t_start) * 1_000_000
            query_latencies.append(lat_us)

            top1_cid = max(scores.items(), key=lambda x: x[1])[0]
            top1_by_query.append(top1_cid)
            scores_by_query.append(scores)

        p50 = float(np.percentile(query_latencies, 50))
        p95 = float(np.percentile(query_latencies, 95))

        results[prec] = {
            "precision": prec,
            "raw_vector_bytes": store.raw_vector_bytes,
            "metadata_bytes": store.metadata_bytes,
            "storage_bytes": store.storage_bytes,
            "compression_ratio": round(store.compression_ratio_vs_fp32, 2),
            "build_time_s": round(build_time, 3),
            "p50_query_latency_us": round(p50, 1),
            "p95_query_latency_us": round(p95, 1),
            "top1_list": top1_by_query,
            "scores_list": scores_by_query,
        }

    # Compare quality vs FP32
    fp32_top1 = results["float32"]["top1_list"]
    fp32_scores = results["float32"]["scores_list"]

    for prec in ["float32", "float16", "int8"]:
        top1_matches = sum(1 for a, b in zip(fp32_top1, results[prec]["top1_list"]) if a == b)
        top1_agreement = top1_matches / len(fp32_top1)

        mae_list = []
        for s_fp32, s_target in zip(fp32_scores, results[prec]["scores_list"]):
            for cid in s_fp32:
                mae_list.append(abs(s_fp32[cid] - s_target[cid]))

        results[prec]["top1_agreement_pct"] = round(top1_agreement * 100.0, 2)
        results[prec]["score_mae_vs_fp32"] = float(np.mean(mae_list))

        # Cleanup bulky raw lists from saved dict
        del results[prec]["top1_list"]
        del results[prec]["scores_list"]

    # Print summary
    print("\n--- Phase 7D Empirical Comparison ---")
    print(f"{'Precision':<10} | {'Storage (KB)':<12} | {'Ratio':<6} | {'Top-1 Aggr':<10} | {'Score MAE':<10} | {'p50 Lat (µs)':<12} | {'p95 Lat (µs)':<12}")
    print("-" * 88)
    for prec, r in results.items():
        kb = r['storage_bytes'] / 1024
        print(f"{prec:<10} | {kb:<12.1f} | {r['compression_ratio']:<6.2f}x | {r['top1_agreement_pct']:<9.1f}% | {r['score_mae_vs_fp32']:<10.6f} | {r['p50_query_latency_us']:<12.1f} | {r['p95_query_latency_us']:<12.1f}")

    out_file = Path("results/phase7d/precision_comparison.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_file}")

if __name__ == "__main__":
    run_precision_benchmark()
