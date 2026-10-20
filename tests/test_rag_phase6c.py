"""
tests/test_rag_phase6c.py
--------------------------
DCLC-RAG Phase 6C Test Suite — Deduplication-Aware Hybrid Retrieval.

Covers:
  1. ChunkVersionIndex reverse mapping and reference counting.
  2. Proof of exact embedding reuse: identical chunk bytes reuse single vector.
  3. Modified chunk isolation: different bytes produce distinct vectors.
  4. BM25Index tokenization, scoring, and ranking correctness.
  5. Dense EmbeddingStore vector generation and cosine scoring.
  6. HybridRetriever score fusion (BM25 + Cosine) and alpha weights.
  7. Version-scoped filtering: queries for Version 1 cannot return Version 2-only chunks.
  8. BaselineRetriever: standard non-deduplicated baseline behavior.
  9. EvaluationHarness: metrics, Precision@K, Recall@K, and version isolation verification.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from src.rag.version_mapper import VersionRegistry
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex, VersionRef
from src.rag.retrieval.bm25_index import BM25Index, tokenize
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.baseline_retriever import BaselineRetriever
from src.rag.retrieval.evaluation_harness import EvaluationHarness, EvalQuery


_SMALL_PARAMS = dict(min_size=64, avg_size=256, max_size=1024)


def _lorem(n_bytes: int) -> bytes:
    unit = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    repeats = (n_bytes // len(unit)) + 1
    return (unit * repeats)[:n_bytes]


# ---------------------------------------------------------------------------
# Test 1 — ChunkVersionIndex reverse mapping & duplicate ratio
# ---------------------------------------------------------------------------


def test_chunk_version_index_building(tmp_path: Path) -> None:
    shared_block = _lorem(2048)  # ~8 chunks at avg=256
    v1_unique = b"VERSION_1_UNIQUE_BODY_CONTENT_" * 80
    v2_unique = b"VERSION_2_UNIQUE_BODY_CONTENT_" * 80

    p = tmp_path / "spec.txt"
    p.write_bytes(shared_block + v1_unique)

    registry = VersionRegistry()
    v1 = registry.ingest_document(p, **_SMALL_PARAMS)

    p.write_bytes(shared_block + v2_unique)
    v2 = registry.ingest_document(p, **_SMALL_PARAMS)

    idx = ChunkVersionIndex()
    idx.build(registry)

    assert idx.total_slots() == len(v1.chunks) + len(v2.chunks)
    assert idx.unique_chunk_count() < idx.total_slots()
    assert idx.dedup_ratio() > 0.0

    # Test reverse mapping
    shared_ids = {c.chunk_id for c in v1.chunks} & {c.chunk_id for c in v2.chunks}
    assert len(shared_ids) > 0
    for sid in shared_ids:
        refs = idx.get_refs(sid)
        assert len(refs) == 2
        v_ids = {r.version_id for r in refs}
        assert v1.version_id in v_ids
        assert v2.version_id in v_ids


# ---------------------------------------------------------------------------
# Test 2 — Exact embedding reuse: identical chunk bytes reuse single vector
# ---------------------------------------------------------------------------


def test_exact_embedding_reuse_proof(tmp_path: Path) -> None:
    shared_content = b"IDENTICAL_CONTENT_ACROSS_FILES_" * 30

    p1 = tmp_path / "file1.txt"
    p1.write_bytes(shared_content)
    p2 = tmp_path / "file2.txt"
    p2.write_bytes(shared_content)

    registry = VersionRegistry()
    v1 = registry.ingest_document(p1, **_SMALL_PARAMS)
    v2 = registry.ingest_document(p2, **_SMALL_PARAMS)

    idx = ChunkVersionIndex()
    idx.build(registry)

    store = EmbeddingStore()
    store.build(idx, batch_size=16)

    # Both documents have identical content -> unique_chunk_count must equal chunks in single file
    assert idx.unique_chunk_count() == len(v1.chunks)
    assert store.embedding_count == len(v1.chunks)

    # Prove each chunk in v2 references the exact same embedding in store
    chunk_ids_in_store = set(store.get_chunk_ids())
    for chunk in v2.chunks:
        assert chunk.chunk_id in chunk_ids_in_store
        refs = idx.get_refs(chunk.chunk_id)
        assert len(refs) >= 2  # Provenance tracks both file1 and file2


# ---------------------------------------------------------------------------
# Test 3 — Modified chunk produces separate embedding (no accidental merging)
# ---------------------------------------------------------------------------


def test_modified_chunk_produces_separate_embedding(tmp_path: Path) -> None:
    p = tmp_path / "doc.txt"
    p.write_bytes(b"Exact line Alpha.\n" * 50)
    registry = VersionRegistry()
    v1 = registry.ingest_document(p, **_SMALL_PARAMS)

    p.write_bytes(b"Exact line Beta.\n" * 50)
    v2 = registry.ingest_document(p, **_SMALL_PARAMS)

    idx = ChunkVersionIndex()
    idx.build(registry)

    store = EmbeddingStore()
    store.build(idx, batch_size=16)

    # Different text must never share chunk_id
    v1_ids = {c.chunk_id for c in v1.chunks}
    v2_ids = {c.chunk_id for c in v2.chunks}
    assert v1_ids.isdisjoint(v2_ids)
    assert store.embedding_count == len(v1.chunks) + len(v2.chunks)


# ---------------------------------------------------------------------------
# Test 4 — BM25Index tokenization, scoring, and ranking
# ---------------------------------------------------------------------------


def test_bm25_tokenization_and_ranking(tmp_path: Path) -> None:
    tokens = tokenize("Cloud-Native, storage_cluster 10Gbps!")
    assert "cloud" in tokens
    assert "native" in tokens
    assert "storage" in tokens
    assert "10gbps" in tokens

    p1 = tmp_path / "doc1.txt"
    p1.write_bytes(b"Compression engine configuration with LZ4 algorithm.\n" * 20)
    p2 = tmp_path / "doc2.txt"
    p2.write_bytes(b"Network security policy enforcing TLS encryption certificates.\n" * 20)

    registry = VersionRegistry()
    registry.ingest_document(p1, **_SMALL_PARAMS)
    registry.ingest_document(p2, **_SMALL_PARAMS)

    idx = ChunkVersionIndex()
    idx.build(registry)

    bm25 = BM25Index()
    bm25.build_dedup_aware(idx)

    scores_lz4 = bm25.score("LZ4 compression algorithm")
    assert len(scores_lz4) > 0
    top_chunk_id = max(scores_lz4, key=scores_lz4.get)
    assert "lz4" in idx.get_text(top_chunk_id).lower()

    scores_tls = bm25.score("TLS certificates")
    assert len(scores_tls) > 0
    top_chunk_tls = max(scores_tls, key=scores_tls.get)
    assert "tls" in idx.get_text(top_chunk_tls).lower()


# ---------------------------------------------------------------------------
# Test 5 — HybridRetriever score fusion and weights (alpha)
# ---------------------------------------------------------------------------


def test_hybrid_retriever_score_fusion(tmp_path: Path) -> None:
    p = tmp_path / "sys.txt"
    p.write_bytes(b"Primary cluster configuration with redundant power supplies.\n" * 30)

    registry = VersionRegistry()
    registry.ingest_document(p, **_SMALL_PARAMS)

    idx = ChunkVersionIndex()
    idx.build(registry)

    bm25 = BM25Index()
    bm25.build_dedup_aware(idx)

    store = EmbeddingStore()
    store.build(idx, batch_size=16)

    retriever = HybridRetriever(bm25, store, idx)

    # Retrieve with alpha=1.0 (pure BM25)
    res_bm25 = retriever.retrieve("redundant power", top_k=3, alpha=1.0)
    assert len(res_bm25) > 0
    assert res_bm25[0].bm25_score > 0

    # Retrieve with alpha=0.0 (pure Dense)
    res_dense = retriever.retrieve("redundant power", top_k=3, alpha=0.0)
    assert len(res_dense) > 0
    assert res_dense[0].cosine_score > 0

    # Retrieve with hybrid alpha=0.5
    res_hybrid = retriever.retrieve("redundant power", top_k=3, alpha=0.5)
    assert len(res_hybrid) > 0
    assert 0.0 <= res_hybrid[0].score <= 1.0


# ---------------------------------------------------------------------------
# Test 6 — Version filtering strictly isolates Version 1 from Version 2
# ---------------------------------------------------------------------------


def test_version_isolation_enforcement(tmp_path: Path) -> None:
    p = tmp_path / "arch.txt"
    p.write_bytes(b"Allocated memory is 16GB ECC RAM in release one.\n" * 30)

    registry = VersionRegistry()
    v1 = registry.ingest_document(p, **_SMALL_PARAMS)

    p.write_bytes(b"Allocated memory is 64GB ECC RAM in release two.\n" * 30)
    v2 = registry.ingest_document(p, **_SMALL_PARAMS)

    idx = ChunkVersionIndex()
    idx.build(registry)

    bm25 = BM25Index()
    bm25.build_dedup_aware(idx)

    store = EmbeddingStore()
    store.build(idx, batch_size=16)

    retriever = HybridRetriever(bm25, store, idx)

    # Query scoped strictly to Version 1
    v1_results = retriever.retrieve(
        query="Allocated memory ECC RAM",
        top_k=5,
        doc_id=v1.doc_id,
        version_id=v1.version_id,
    )
    assert len(v1_results) > 0
    for r in v1_results:
        assert "16gb" in r.text_content.lower()
        assert "64gb" not in r.text_content.lower()
        assert any(ref.version_id == v1.version_id for ref in r.filtered_refs)
        assert not any(ref.version_id == v2.version_id for ref in r.filtered_refs)

    # Query scoped strictly to Version 2
    v2_results = retriever.retrieve(
        query="Allocated memory ECC RAM",
        top_k=5,
        doc_id=v2.doc_id,
        version_id=v2.version_id,
    )
    assert len(v2_results) > 0
    for r in v2_results:
        assert "64gb" in r.text_content.lower()
        assert "16gb" not in r.text_content.lower()
        assert any(ref.version_id == v2.version_id for ref in r.filtered_refs)


# ---------------------------------------------------------------------------
# Test 7 — BaselineRetriever operation
# ---------------------------------------------------------------------------


def test_baseline_retriever_operation(tmp_path: Path) -> None:
    p = tmp_path / "base_test.txt"
    p.write_bytes(b"Baseline standard retrieval with independent slot keys.\n" * 30)

    registry = VersionRegistry()
    registry.ingest_document(p, **_SMALL_PARAMS)

    baseline = BaselineRetriever()
    baseline.build(registry, batch_size=16)

    assert baseline.total_slots > 0
    results = baseline.retrieve("independent slot keys", top_k=2)
    assert len(results) > 0
    assert "independent" in results[0].text_content.lower()


# ---------------------------------------------------------------------------
# Test 8 — EvaluationHarness on data/rag_eval corpus
# ---------------------------------------------------------------------------


def test_evaluation_harness_on_rag_eval_corpus() -> None:
    corpus_dir = Path("data/rag_eval")
    queries_file = corpus_dir / "queries.json"

    if not queries_file.exists():
        pytest.skip("data/rag_eval corpus not found.")

    with open(queries_file, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    registry = VersionRegistry()
    doc_file_to_version = {}

    for fname in ["system_spec_v1.txt", "system_spec_v2.txt", "system_spec_v3.txt", "network_policy.md", "backup_service.py"]:
        fpath = corpus_dir / fname
        if fpath.exists():
            v = registry.ingest_document(fpath, **_SMALL_PARAMS)
            doc_file_to_version[fname] = v

    idx = ChunkVersionIndex()
    idx.build(registry)

    bm25 = BM25Index()
    bm25.build_dedup_aware(idx)

    store = EmbeddingStore()
    store.build(idx, batch_size=16)

    dclc_retriever = HybridRetriever(bm25, store, idx)

    baseline = BaselineRetriever()
    baseline.build(registry, batch_size=16)

    # Build EvalQuery objects
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

    harness = EvaluationHarness(ks=[1, 3, 5])
    dclc_report = harness.evaluate_dclc(dclc_retriever, eval_queries, alpha=0.5)
    base_report = harness.evaluate_baseline(baseline, eval_queries, alpha=0.5)

    assert dclc_report.version_isolation_violations == 0
    assert base_report.version_isolation_violations == 0
    assert dclc_report.unique_vectors < base_report.unique_vectors
    assert dclc_report.vector_payload_bytes < base_report.vector_payload_bytes
    assert dclc_report.precision_at_k[1] > 0.0
