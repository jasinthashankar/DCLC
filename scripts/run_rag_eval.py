"""
scripts/run_rag_eval.py
-----------------------
Runs comparative evaluation between DCLC-Aware Hybrid Retrieval
and Standard Non-Deduplicated Baseline Retrieval on data/rag_eval.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag.version_mapper import VersionRegistry
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.baseline_retriever import BaselineRetriever
from src.rag.retrieval.evaluation_harness import EvaluationHarness, EvalQuery


def main() -> None:
    corpus_dir = Path("data/rag_eval")
    queries_file = corpus_dir / "queries.json"

    with open(queries_file, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    _SMALL_PARAMS = dict(min_size=64, avg_size=256, max_size=1024)

    print("=" * 70)
    print("DCLC-RAG Phase 6C: Comparative Retrieval Evaluation")
    print("=" * 70)

    # 1. Ingest documents into VersionRegistry
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
        print(f"Ingested {fname:20s}: {len(v.chunks)} chunks, {v.total_bytes} bytes")

    # 2. Build DCLC-Aware Index
    print("\nBuilding DCLC-Aware Index...")
    chunk_index = ChunkVersionIndex()
    chunk_index.build(registry)

    bm25_dclc = BM25Index()
    bm25_dclc.build_dedup_aware(chunk_index)

    store_dclc = EmbeddingStore()
    store_dclc.build(chunk_index, batch_size=16)

    dclc_retriever = HybridRetriever(bm25_dclc, store_dclc, chunk_index)

    # 3. Build Standard Baseline Index
    print("Building Standard Baseline Index...")
    baseline = BaselineRetriever()
    baseline.build(registry, batch_size=16)

    # 4. Prepare evaluation queries
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

    # 5. Run Evaluation Harness
    harness = EvaluationHarness(ks=[1, 3, 5])
    print(f"\nEvaluating across {len(eval_queries)} queries (alpha=0.5)...")

    dclc_report = harness.evaluate_dclc(dclc_retriever, eval_queries, alpha=0.5)
    base_report = harness.evaluate_baseline(baseline, eval_queries, alpha=0.5)

    # Also test alpha=0.0 (pure dense) and alpha=1.0 (pure BM25) for DCLC
    dclc_dense = harness.evaluate_dclc(dclc_retriever, eval_queries, alpha=0.0)
    dclc_bm25 = harness.evaluate_dclc(dclc_retriever, eval_queries, alpha=1.0)

    # 6. Print Report
    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Standard Baseline':<20} | {'DCLC-Aware Hybrid':<20}")
    print("-" * 80)
    print(f"{'Total Version Chunks (Slots)':<35} | {baseline.total_slots:<20} | {chunk_index.total_slots():<20}")
    print(f"{'Unique Chunks Embedded':<35} | {base_report.unique_vectors:<20} | {dclc_report.unique_vectors:<20}")
    print(f"{'Embedding Computations':<35} | {base_report.embedding_computations:<20} | {dclc_report.embedding_computations:<20}")
    print(f"{'Vector Payload Size':<35} | {f'{base_report.vector_payload_bytes:,} B':<20} | {f'{dclc_report.vector_payload_bytes:,} B':<20}")
    print(f"{'Embedding Generation Time':<35} | {f'{base_report.embedding_build_time_seconds:.4f} s':<20} | {f'{dclc_report.embedding_build_time_seconds:.4f} s':<20}")
    print(f"{'Mean Retrieval Latency':<35} | {f'{base_report.mean_retrieval_latency_ms:.2f} ms':<20} | {f'{dclc_report.mean_retrieval_latency_ms:.2f} ms':<20}")
    print(f"{'Precision@1':<35} | {f'{base_report.precision_at_k[1]:.4f}':<20} | {f'{dclc_report.precision_at_k[1]:.4f}':<20}")
    print(f"{'Precision@3':<35} | {f'{base_report.precision_at_k[3]:.4f}':<20} | {f'{dclc_report.precision_at_k[3]:.4f}':<20}")
    print(f"{'Precision@5':<35} | {f'{base_report.precision_at_k[5]:.4f}':<20} | {f'{dclc_report.precision_at_k[5]:.4f}':<20}")
    print(f"{'Recall@1':<35} | {f'{base_report.recall_at_k[1]:.4f}':<20} | {f'{dclc_report.recall_at_k[1]:.4f}':<20}")
    print(f"{'Recall@3':<35} | {f'{base_report.recall_at_k[3]:.4f}':<20} | {f'{dclc_report.recall_at_k[3]:.4f}':<20}")
    print(f"{'Recall@5':<35} | {f'{base_report.recall_at_k[5]:.4f}':<20} | {f'{dclc_report.recall_at_k[5]:.4f}':<20}")
    print(f"{'Version Isolation Violations':<35} | {base_report.version_isolation_violations:<20} | {dclc_report.version_isolation_violations:<20}")
    print("=" * 80)

    print("\nAblation: DCLC Retrieval across Alpha settings:")
    print(f"  alpha=0.0 (Pure Dense):  P@1={dclc_dense.precision_at_k[1]:.4f}, R@1={dclc_dense.recall_at_k[1]:.4f}")
    print(f"  alpha=0.5 (Hybrid):      P@1={dclc_report.precision_at_k[1]:.4f}, R@1={dclc_report.recall_at_k[1]:.4f}")
    print(f"  alpha=1.0 (Pure BM25):   P@1={dclc_bm25.precision_at_k[1]:.4f}, R@1={dclc_bm25.recall_at_k[1]:.4f}")


if __name__ == "__main__":
    main()
