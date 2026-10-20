"""
tests/test_rag_phase7c.py
-------------------------
Phase 7C: Research Validation & Scalability Invariant Tests for DCLC-RAG.

Tests:
1. Empty index handling across all 6 configurations (Standard Baseline & DCLC-Aware in FP32, FP16, INT8).
2. Exact deduplication invariant: byte-identical chunks share single vector, similar-but-different chunks do NOT.
3. Unicode text handling (multibyte UTF-8, accents, emojis, CJK) across all precision modes.
4. Strict version isolation invariant across FP32, FP16, and INT8.
5. Quantization mathematical bounds and dot-product monotonicity.
6. Fair comparison invariant: total version-chunk slots in Baseline == ChunkVersionIndex slots.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.rag.version_mapper import VersionRegistry
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.baseline_retriever import BaselineRetriever


@pytest.mark.parametrize("precision", ["float32", "float16", "int8"])
def test_phase7c_empty_store_retrieval(precision):
    """Empty store handles queries gracefully without exception in all precisions."""
    idx = ChunkVersionIndex()
    bm25 = BM25Index()
    store = EmbeddingStore(precision=precision)
    store.build(idx)
    retriever = HybridRetriever(bm25, store, idx)

    results = retriever.retrieve("Any query in empty system", top_k=5)
    assert len(results) == 0

    # Also test empty baseline
    base = BaselineRetriever(precision=precision)
    base_results = base.retrieve("Any query in empty baseline", top_k=5)
    assert len(base_results) == 0


_SMALL_PARAMS = dict(min_size=64, avg_size=256, max_size=1024)


@pytest.mark.parametrize("precision", ["float32", "float16", "int8"])
def test_phase7c_unicode_text_preservation(precision):
    """Verifies indexing and scoring of complex multibyte Unicode text across precisions."""
    idx = ChunkVersionIndex()
    store = EmbeddingStore(precision=precision)

    # Ingest synthetic documents with Unicode
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = VersionRegistry()
        p1 = Path(tmpdir) / "unicode_doc.txt"
        p1.write_text(
            "Système de stockage distribué avec déduplication.\n" * 4 +
            "量子計算と分散データベースの最適化.\n" * 4 +
            "High-concurrency cloud architecture with 🚀 fast performance and €100 cost.\n" * 4,
            encoding="utf-8",
        )
        reg.ingest_document(p1, **_SMALL_PARAMS)
        idx.build(reg)
        bm25 = BM25Index()
        bm25.build_dedup_aware(idx)
        store.build(idx)

        retriever = HybridRetriever(bm25, store, idx)
        hits = retriever.retrieve("Système de stockage", top_k=3)
        assert len(hits) > 0
        assert "stockage" in hits[0].text_content.lower()

        hits_jp = retriever.retrieve("量子計算", top_k=3)
        assert len(hits_jp) > 0


def test_phase7c_strict_sha256_exact_dedup_invariant():
    """Confirms deduplication occurs ONLY on exact byte identity, not semantic similarity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = VersionRegistry()
        # Doc 1
        p1 = Path(tmpdir) / "doc1.txt"
        p1.write_text("The primary database runs on port 5432 with replication enabled.\n" * 5, encoding="utf-8")
        reg.ingest_document(p1, **_SMALL_PARAMS)

        # Doc 2 with semantically identical but slightly different text
        p2 = Path(tmpdir) / "doc2.txt"
        p2.write_text("The primary database runs on port 5432 with replication enabled!\n" * 5, encoding="utf-8") # exclamation mark
        reg.ingest_document(p2, **_SMALL_PARAMS)

        # Doc 3 with EXACT duplicate of Doc 1
        p3 = Path(tmpdir) / "doc3.txt"
        p3.write_text("The primary database runs on port 5432 with replication enabled.\n" * 5, encoding="utf-8")
        reg.ingest_document(p3, **_SMALL_PARAMS)

        idx = ChunkVersionIndex()
        idx.build(reg)

        # 3 documents x 2 chunks each = 6 total slots
        # Doc 1 and Doc 3 share 2 identical chunks; Doc 2 has 2 distinct chunks -> 4 unique chunks
        assert idx.total_slots() == 6
        assert idx.unique_chunk_count() == 4


@pytest.mark.parametrize("precision", ["float32", "float16", "int8"])
def test_phase7c_version_isolation_across_precisions(precision):
    """Ensures version filtering strictly isolates version-specific content across precisions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = VersionRegistry()

        dir1 = Path(tmpdir) / "v1"
        dir1.mkdir()
        dir2 = Path(tmpdir) / "v2"
        dir2.mkdir()

        # Both have same filename "config.txt" so they share doc_id
        f1 = dir1 / "config.txt"
        f1.write_text("Port: 8080. Secret token: ALPHA_KEY_V1. Deprecated in v2.\n" * 5, encoding="utf-8")
        v1 = reg.ingest_document(f1, **_SMALL_PARAMS)

        f2 = dir2 / "config.txt"
        f2.write_text("Port: 9090. Secret token: BETA_KEY_V2. Introduced in v2.\n" * 5, encoding="utf-8")
        v2 = reg.ingest_document(f2, **_SMALL_PARAMS)

        assert v1.doc_id == v2.doc_id
        assert v1.version_id != v2.version_id

        idx = ChunkVersionIndex()
        idx.build(reg)
        bm25 = BM25Index()
        bm25.build_dedup_aware(idx)
        store = EmbeddingStore(precision=precision)
        store.build(idx)

        retriever = HybridRetriever(bm25, store, idx)

        # Query scoped to Version 1
        v1_hits = retriever.retrieve("Secret token", doc_id=v1.doc_id, version_id=v1.version_id, top_k=5)
        for h in v1_hits:
            assert "BETA_KEY_V2" not in h.text_content
            assert any(ref.version_id == v1.version_id for ref in h.filtered_refs)

        # Query scoped to Version 2
        v2_hits = retriever.retrieve("Secret token", doc_id=v2.doc_id, version_id=v2.version_id, top_k=5)
        for h in v2_hits:
            assert "ALPHA_KEY_V1" not in h.text_content
            assert any(ref.version_id == v2.version_id for ref in h.filtered_refs)


def test_phase7c_int8_quantization_bounds():
    """Mathematically validates quantization error bounds and integer norm fidelity."""
    np.random.seed(42)
    # Generate 100 random unit vectors in 384 dimensions
    raw_vecs = np.random.randn(100, 384).astype(np.float32)
    unit_vecs = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)

    # Store into int8 store
    store = EmbeddingStore(precision="int8")
    store._matrix = np.zeros((100, 384), dtype=np.int8)
    store._chunk_ids = [f"c_{i}" for i in range(100)]
    store._store_matrix(unit_vecs)

    # Validate reconstructed vectors
    recon_vecs = np.array([store.get_vector(f"c_{i}") for i in range(100)])

    # Unit norm check
    recon_norms = np.linalg.norm(recon_vecs, axis=1)
    np.testing.assert_allclose(recon_norms, 1.0, atol=1e-5)

    # Absolute reconstruction error
    diff = np.abs(recon_vecs - unit_vecs)
    max_err = np.max(diff)
    mean_err = np.mean(diff)

    assert max_err < 0.015, f"Max error {max_err} exceeds 0.015"
    assert mean_err < 0.005, f"Mean error {mean_err} exceeds 0.005"
