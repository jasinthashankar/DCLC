"""
scripts/benchmark_phase7_baseline.py
------------------------------------
Reproducible Phase 7A Baseline Benchmark Script.

Measures:
1. Document & chunk inventory (documents, versions, unique chunks, total slots).
2. Numerical storage breakdown (dtype, dimension, raw vector payload bytes).
3. Deep memory profiling of all index structures and metadata:
   - EmbeddingStore (raw matrix + metadata)
   - ChunkVersionIndex (reverse mapping + canonical text)
   - BM25Index (vocabulary, IDF, document lengths)
   - VersionRegistry (raw bytes, ChunkRecords, DocumentVersions)
4. Retrieval effectiveness (Precision@k, Recall@k, Version Isolation Violations).
5. Multi-iteration latency profiling (mean, median, p90, p95 across 50 runs).
6. Controlled side-by-side comparison between Standard Baseline and DCLC-Aware Hybrid.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch

from src.rag.version_mapper import VersionRegistry
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.baseline_retriever import BaselineRetriever
from src.rag.retrieval.evaluation_harness import EvaluationHarness, EvalQuery


def deep_getsizeof(o: Any, seen: Set[int] | None = None) -> int:
    """
    Recursively calculates the exact deep memory size of any Python object in bytes.
    Avoids double counting via an object ID set.
    """
    if seen is None:
        seen = set()
    obj_id = id(o)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(o)

    if isinstance(o, np.ndarray):
        # sys.getsizeof on ndarray gives array wrapper overhead, nbytes gives data buffer
        return size + o.nbytes
    elif isinstance(o, (str, bytes, int, float, bool)) or o is None:
        return size
    elif isinstance(o, dict):
        for k, v in o.items():
            size += deep_getsizeof(k, seen)
            size += deep_getsizeof(v, seen)
    elif isinstance(o, (list, tuple, set, frozenset)):
        for item in o:
            size += deep_getsizeof(item, seen)
    elif is_dataclass(o):
        for f in fields(o):
            val = getattr(o, f.name, None)
            size += deep_getsizeof(val, seen)
    elif hasattr(o, "__dict__"):
        size += deep_getsizeof(o.__dict__, seen)

    return size


def run_latency_profile(retriever: Any, queries: List[EvalQuery], repetitions: int = 50, alpha: float = 0.5) -> Dict[str, float]:
    """
    Measures retrieval latency over multiple repetitions after warm-up.
    """
    # Warm-up run
    for q in queries:
        retriever.retrieve(
            query=q.query_text,
            top_k=5,
            alpha=alpha,
            doc_id=q.scope_doc_id,
            version_id=q.scope_version_id,
        )

    latencies_ms: List[float] = []
    for _ in range(repetitions):
        for q in queries:
            t0 = time.perf_counter()
            retriever.retrieve(
                query=q.query_text,
                top_k=5,
                alpha=alpha,
                doc_id=q.scope_doc_id,
                version_id=q.scope_version_id,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "total_queries_measured": len(latencies_ms),
    }


def main() -> None:
    corpus_dir = Path("data/rag_eval")
    queries_file = corpus_dir / "queries.json"

    with open(queries_file, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    _SMALL_PARAMS = dict(min_size=64, avg_size=256, max_size=1024)

    print("=" * 80)
    print("DCLC-RAG Phase 7A: Rigorous Baseline Retrieval & Memory Audit")
    print("=" * 80)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Corpus Directory: {corpus_dir.resolve()}")
    print("-" * 80)

    # 1. Ingest Documents into VersionRegistry
    registry = VersionRegistry()
    doc_file_to_version = {}
    doc_filenames = [
        "system_spec_v1.txt",
        "system_spec_v2.txt",
        "system_spec_v3.txt",
        "network_policy.md",
        "backup_service.py",
    ]

    for fname in doc_filenames:
        fpath = corpus_dir / fname
        v = registry.ingest_document(fpath, **_SMALL_PARAMS)
        doc_file_to_version[fname] = v

    # 2. Build DCLC-Aware Structures
    chunk_index = ChunkVersionIndex()
    chunk_index.build(registry)

    bm25_dclc = BM25Index()
    bm25_dclc.build_dedup_aware(chunk_index)

    store_dclc = EmbeddingStore()
    store_dclc.build(chunk_index, batch_size=16)

    dclc_retriever = HybridRetriever(bm25_dclc, store_dclc, chunk_index)

    # 3. Build Standard Baseline Structures
    baseline = BaselineRetriever()
    baseline.build(registry, batch_size=16)

    # 4. Prepare Evaluation Queries
    eval_queries = []
    for qd in queries_data:
        v = doc_file_to_version.get(qd.get("doc_filename"))
        eval_queries.append(
            EvalQuery(
                query_id=qd["query_id"],
                query_text=qd["query_text"],
                scope_doc_id=v.doc_id if v else None,
                scope_version_id=v.version_id if v else None,
                expected_answer_keywords=qd.get("expected_answer_keywords", []),
                prohibited_keywords=qd.get("prohibited_keywords", []),
            )
        )

    # 5. Measure Deep Memory
    gc.collect()

    # DCLC-Aware components
    dclc_raw_matrix_bytes = store_dclc._matrix.nbytes if store_dclc._matrix is not None else 0
    dclc_store_deep_bytes = deep_getsizeof(store_dclc)
    dclc_chunk_idx_deep_bytes = deep_getsizeof(chunk_index)
    dclc_bm25_deep_bytes = deep_getsizeof(bm25_dclc)
    dclc_retriever_total_bytes = deep_getsizeof(dclc_retriever)

    # Baseline components
    base_raw_matrix_bytes = baseline.store._matrix.nbytes if baseline.store._matrix is not None else 0
    base_store_deep_bytes = deep_getsizeof(baseline.store)
    base_slots_deep_bytes = deep_getsizeof(baseline._slots)
    base_bm25_deep_bytes = deep_getsizeof(baseline.bm25)
    base_retriever_total_bytes = deep_getsizeof(baseline)

    registry_deep_bytes = deep_getsizeof(registry)

    # 6. Evaluation Harness Metrics
    harness = EvaluationHarness(ks=[1, 3, 5])
    dclc_report = harness.evaluate_dclc(dclc_retriever, eval_queries, alpha=0.5)
    base_report = harness.evaluate_baseline(baseline, eval_queries, alpha=0.5)

    # 7. Multi-Repetition Latency Profiling (50 repetitions = 500 query evaluations per system)
    print("Running 50-repetition latency profiling (500 queries per system)...")
    dclc_lat = run_latency_profile(dclc_retriever, eval_queries, repetitions=50, alpha=0.5)
    base_lat = run_latency_profile(baseline, eval_queries, repetitions=50, alpha=0.5)

    # 8. Produce Detailed Report Output
    print("\n" + "=" * 80)
    print("DETAILED PHASE 7A BASELINE AUDIT RESULTS")
    print("=" * 80)
    print(f"{'Metric':<40} | {'Standard Baseline':<18} | {'DCLC-Aware Hybrid':<18}")
    print("-" * 80)
    print(f"{'Logical Documents Ingested':<40} | {len(registry.list_doc_ids()):<18} | {len(registry.list_doc_ids()):<18}")
    print(f"{'Document Versions':<40} | {len(doc_filenames):<18} | {len(doc_filenames):<18}")
    print(f"{'Total Version Chunks (Slots)':<40} | {baseline.total_slots:<18} | {chunk_index.total_slots():<18}")
    print(f"{'Unique Chunks Embedded':<40} | {base_report.unique_vectors:<18} | {dclc_report.unique_vectors:<18}")
    print(f"{'Deduplication Ratio':<40} | {'0.00%':<18} | {f'{chunk_index.dedup_ratio()*100:.2f}%':<18}")
    print(f"{'Embedding Dimension':<40} | {baseline.store.embedding_dim:<18} | {store_dclc.embedding_dim:<18}")
    print(f"{'Numerical Data Type':<40} | {str(baseline.store._matrix.dtype):<18} | {str(store_dclc._matrix.dtype):<18}")
    print(f"{'Raw Vector Matrix Memory':<40} | {f'{base_raw_matrix_bytes:,} B':<18} | {f'{dclc_raw_matrix_bytes:,} B':<18}")
    print(f"{'EmbeddingStore Deep Memory':<40} | {f'{base_store_deep_bytes:,} B':<18} | {f'{dclc_store_deep_bytes:,} B':<18}")
    print(f"{'BM25 Index Deep Memory':<40} | {f'{base_bm25_deep_bytes:,} B':<18} | {f'{dclc_bm25_deep_bytes:,} B':<18}")
    print(f"{'Index Provenance/Slot Mapping':<40} | {f'{base_slots_deep_bytes:,} B':<18} | {f'{dclc_chunk_idx_deep_bytes:,} B':<18}")
    print(f"{'Total Retriever Object Footprint':<40} | {f'{base_retriever_total_bytes:,} B':<18} | {f'{dclc_retriever_total_bytes:,} B':<18}")
    print(f"{'Shared Registry Memory (Source Texts)':<40} | {f'{registry_deep_bytes:,} B':<18} | {f'{registry_deep_bytes:,} B':<18}")
    print("-" * 80)
    print(f"{'Precision@1':<40} | {f'{base_report.precision_at_k[1]:.4f}':<18} | {f'{dclc_report.precision_at_k[1]:.4f}':<18}")
    print(f"{'Precision@3':<40} | {f'{base_report.precision_at_k[3]:.4f}':<18} | {f'{dclc_report.precision_at_k[3]:.4f}':<18}")
    print(f"{'Precision@5':<40} | {f'{base_report.precision_at_k[5]:.4f}':<18} | {f'{dclc_report.precision_at_k[5]:.4f}':<18}")
    print(f"{'Recall@1':<40} | {f'{base_report.recall_at_k[1]:.4f}':<18} | {f'{dclc_report.recall_at_k[1]:.4f}':<18}")
    print(f"{'Recall@3':<40} | {f'{base_report.recall_at_k[3]:.4f}':<18} | {f'{dclc_report.recall_at_k[3]:.4f}':<18}")
    print(f"{'Recall@5':<40} | {f'{base_report.recall_at_k[5]:.4f}':<18} | {f'{dclc_report.recall_at_k[5]:.4f}':<18}")
    print(f"{'Version Isolation Violations':<40} | {base_report.version_isolation_violations:<18} | {dclc_report.version_isolation_violations:<18}")
    print("-" * 80)
    base_mean_str = f"{base_lat['mean_ms']:.3f} ms"
    dclc_mean_str = f"{dclc_lat['mean_ms']:.3f} ms"
    base_med_str = f"{base_lat['median_ms']:.3f} ms"
    dclc_med_str = f"{dclc_lat['median_ms']:.3f} ms"
    base_p90_str = f"{base_lat['p90_ms']:.3f} ms"
    dclc_p90_str = f"{dclc_lat['p90_ms']:.3f} ms"
    base_p95_str = f"{base_lat['p95_ms']:.3f} ms"
    dclc_p95_str = f"{dclc_lat['p95_ms']:.3f} ms"

    print(f"{'Mean Retrieval Latency':<40} | {base_mean_str:<18} | {dclc_mean_str:<18}")
    print(f"{'Median Retrieval Latency':<40} | {base_med_str:<18} | {dclc_med_str:<18}")
    print(f"{'p90 Retrieval Latency':<40} | {base_p90_str:<18} | {dclc_p90_str:<18}")
    print(f"{'p95 Retrieval Latency':<40} | {base_p95_str:<18} | {dclc_p95_str:<18}")
    print("=" * 80)

    # Save benchmark dictionary as json for report generation
    results = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "total_slots": baseline.total_slots,
        "unique_chunks_dclc": dclc_report.unique_vectors,
        "unique_chunks_baseline": base_report.unique_vectors,
        "dedup_ratio": chunk_index.dedup_ratio(),
        "embedding_dim": store_dclc.embedding_dim,
        "matrix_dtype": str(store_dclc._matrix.dtype),
        "base_raw_matrix_bytes": base_raw_matrix_bytes,
        "dclc_raw_matrix_bytes": dclc_raw_matrix_bytes,
        "base_store_deep_bytes": base_store_deep_bytes,
        "dclc_store_deep_bytes": dclc_store_deep_bytes,
        "base_bm25_deep_bytes": base_bm25_deep_bytes,
        "dclc_bm25_deep_bytes": dclc_bm25_deep_bytes,
        "base_slots_deep_bytes": base_slots_deep_bytes,
        "dclc_chunk_idx_deep_bytes": dclc_chunk_idx_deep_bytes,
        "base_retriever_total_bytes": base_retriever_total_bytes,
        "dclc_retriever_total_bytes": dclc_retriever_total_bytes,
        "registry_deep_bytes": registry_deep_bytes,
        "base_report": {
            "p1": base_report.precision_at_k[1],
            "p3": base_report.precision_at_k[3],
            "p5": base_report.precision_at_k[5],
            "r1": base_report.recall_at_k[1],
            "r3": base_report.recall_at_k[3],
            "r5": base_report.recall_at_k[5],
            "violations": base_report.version_isolation_violations,
        },
        "dclc_report": {
            "p1": dclc_report.precision_at_k[1],
            "p3": dclc_report.precision_at_k[3],
            "p5": dclc_report.precision_at_k[5],
            "r1": dclc_report.recall_at_k[1],
            "r3": dclc_report.recall_at_k[3],
            "r5": dclc_report.recall_at_k[5],
            "violations": dclc_report.version_isolation_violations,
        },
        "base_lat": base_lat,
        "dclc_lat": dclc_lat,
    }

    out_file = Path("results/phase7_baseline_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline metrics to {out_file.resolve()}")


if __name__ == "__main__":
    main()
