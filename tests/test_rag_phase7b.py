"""
tests/test_rag_phase7b.py
-------------------------
DCLC Phase 7B Test Suite — Research-Guided Embedding Memory Optimization.

Verifies:
  1. Backward compatibility: default precision is float32 without API changes.
  2. Precision validation: invalid precision modes raise clear ValueError.
  3. INT8 / SQ8 mathematical correctness: symmetric scaling to [-127, 127], clipping, and norms.
  4. Dequantization fidelity: reconstructed vectors maintain unit norm with MAE < 0.005.
  5. Zero vector numerical stability: all-zero vectors never produce NaNs or ZeroDivisionError.
  6. FP16 compact storage: 2x compression with cosine score deviation < 1e-4.
  7. INT8 SQ8 ranking preservation: top-1 ranking preserved identical to FP32.
  8. Empty store edge cases: graceful handling of empty documents across float32, float16, int8.
  9. Cross-version deduplication preservation: shared chunks reuse one physical quantized vector.
  10. End-to-end grounded question answering: strict version isolation and valid citations under INT8.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
import numpy as np

from src.rag.version_mapper import VersionRegistry
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.generation.provider import MockLLMProvider
from src.rag.generation.answer_engine import RAGAnswerEngine


_SMALL_PARAMS = dict(min_size=64, avg_size=256, max_size=1024)


# ---------------------------------------------------------------------------
# Test 1: Backward compatibility: default is float32
# ---------------------------------------------------------------------------


def test_default_precision_is_float32():
    store = EmbeddingStore()
    assert store.precision == "float32"
    assert store.compression_ratio_vs_fp32 == 1.0


# ---------------------------------------------------------------------------
# Test 2: Invalid precision validation
# ---------------------------------------------------------------------------


def test_invalid_precision_raises_value_error():
    with pytest.raises(ValueError, match="Invalid precision 'int4'"):
        EmbeddingStore(precision="int4")

    store = EmbeddingStore()
    with pytest.raises(ValueError, match="Invalid precision 'bfloat16'"):
        c_idx = ChunkVersionIndex()
        store.build(c_idx, precision="bfloat16")


# ---------------------------------------------------------------------------
# Test 3: INT8 SQ8 mathematical quantization correctness
# ---------------------------------------------------------------------------


def test_int8_quantization_math_and_scaling(tmp_path: Path):
    p = tmp_path / "test.txt"
    p.write_bytes(b"Quantum memory and vector quantization specification." * 10)

    reg = VersionRegistry()
    reg.ingest_document(p, **_SMALL_PARAMS)

    c_idx = ChunkVersionIndex()
    c_idx.build(reg)

    store = EmbeddingStore(precision="int8")
    store.build(c_idx, batch_size=8)

    assert store.precision == "int8"
    assert store._matrix.dtype == np.int8
    assert store._matrix.ndim == 2
    assert store.embedding_dim == 384
    assert store._scales is not None
    assert np.all(store._scales == 127.0)
    assert np.all(store._matrix >= -127)
    assert np.all(store._matrix <= 127)
    assert store.raw_vector_bytes == store.embedding_count * store.embedding_dim * 1


# ---------------------------------------------------------------------------
# Test 4: Dequantization fidelity
# ---------------------------------------------------------------------------


def test_int8_dequantization_accuracy(tmp_path: Path):
    p = tmp_path / "test.txt"
    p.write_bytes(b"Cloud storage deduplication via contrastive learning." * 10)

    reg = VersionRegistry()
    reg.ingest_document(p, **_SMALL_PARAMS)

    c_idx = ChunkVersionIndex()
    c_idx.build(reg)

    store_fp32 = EmbeddingStore(precision="float32")
    store_fp32.build(c_idx, batch_size=8)

    store_int8 = EmbeddingStore(precision="int8")
    store_int8.build(c_idx, batch_size=8)

    cid = c_idx.all_chunk_ids()[0]
    vec_orig = store_fp32.get_vector(cid)
    vec_rec = store_int8.dequantize(cid)

    assert vec_orig.dtype == np.float32
    assert vec_rec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec_rec), 1.0, atol=1e-5)

    mae = float(np.mean(np.abs(vec_orig - vec_rec)))
    assert mae < 0.005, f"Mean absolute reconstruction error {mae} exceeded 0.005"


# ---------------------------------------------------------------------------
# Test 5: Zero vector and numerical edge cases
# ---------------------------------------------------------------------------


def test_zero_vector_handling_safely():
    store = EmbeddingStore(precision="int8")
    store._chunk_ids = ["chunk_zero"]
    store.embedding_dim = 384
    # Store explicit zero vector
    store._matrix = np.zeros((1, 384), dtype=np.int8)
    store._scales = np.array([127.0], dtype=np.float32)
    store._int_norms = np.array([0.0], dtype=np.float32)

    q_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
    scores = store.cosine_scores(q_vec)
    assert scores["chunk_zero"] == 0.0

    rec_vec = store.get_vector("chunk_zero")
    assert np.all(rec_vec == 0.0)
    assert not np.any(np.isnan(rec_vec))


# ---------------------------------------------------------------------------
# Test 6: FP16 compact storage and scoring
# ---------------------------------------------------------------------------


def test_fp16_compact_storage_and_scoring(tmp_path: Path):
    p = tmp_path / "test.txt"
    p.write_bytes(b"FastCDC variable chunking and SHA-256 fingerprinting." * 10)

    reg = VersionRegistry()
    reg.ingest_document(p, **_SMALL_PARAMS)

    c_idx = ChunkVersionIndex()
    c_idx.build(reg)

    store_fp32 = EmbeddingStore(precision="float32")
    store_fp32.build(c_idx, batch_size=8)

    store_fp16 = EmbeddingStore(precision="float16")
    store_fp16.build(c_idx, batch_size=8)

    assert store_fp16.raw_vector_bytes == store_fp32.raw_vector_bytes // 2
    assert store_fp16._matrix.dtype == np.float16

    q_vec = store_fp32.query_vector("FastCDC chunking")
    scores_fp32 = store_fp32.cosine_scores(q_vec)
    scores_fp16 = store_fp16.cosine_scores(q_vec)

    for cid in c_idx.all_chunk_ids():
        diff = abs(scores_fp32[cid] - scores_fp16[cid])
        assert diff < 1e-4, f"FP16 score deviation {diff} exceeded 1e-4 for {cid}"


# ---------------------------------------------------------------------------
# Test 7: INT8 SQ8 ranking preservation
# ---------------------------------------------------------------------------


def test_int8_ranking_preservation(tmp_path: Path):
    p = tmp_path / "corpus.txt"
    p.write_bytes(
        b"Cluster Specification: Node RAM is 64GB ECC.\n"
        b"Network Policy: Mandates TLS 1.3 encryption.\n"
        b"Backup Service: Daily snapshot at 02:00 UTC.\n" * 15
    )

    reg = VersionRegistry()
    reg.ingest_document(p, **_SMALL_PARAMS)

    c_idx = ChunkVersionIndex()
    c_idx.build(reg)

    store_fp32 = EmbeddingStore(precision="float32")
    store_fp32.build(c_idx, batch_size=8)

    store_int8 = EmbeddingStore(precision="int8")
    store_int8.build(c_idx, batch_size=8)

    q = "What is the memory size in RAM?"
    q_vec = store_fp32.query_vector(q)

    scores_fp32 = store_fp32.cosine_scores(q_vec)
    scores_int8 = store_int8.cosine_scores(q_vec)

    top1_fp32 = sorted(scores_fp32.items(), key=lambda x: x[1], reverse=True)[0][0]
    top1_int8 = sorted(scores_int8.items(), key=lambda x: x[1], reverse=True)[0][0]

    assert top1_fp32 == top1_int8, f"Top-1 mismatch: {top1_fp32} != {top1_int8}"


# ---------------------------------------------------------------------------
# Test 8: Empty store handling across all precisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prec", ["float32", "float16", "int8"])
def test_empty_embedding_store_across_precisions(prec: str):
    c_idx = ChunkVersionIndex()
    c_idx.build(VersionRegistry())

    store = EmbeddingStore(precision=prec)
    store.build(c_idx)

    assert store.embedding_count == 0
    assert store.raw_vector_bytes == 0
    assert store.storage_bytes == 0

    scores = store.cosine_scores(np.zeros(384, dtype=np.float32))
    assert scores == {}


# ---------------------------------------------------------------------------
# Test 9: Deduplication preserved with INT8 storage
# ---------------------------------------------------------------------------


def test_deduplication_preserved_with_int8(tmp_path: Path):
    shared = b"SHARED_CLUSTER_SPEC_ACROSS_REVISIONS_" * 20
    v1_text = shared + b"V1_MEMORY_16GB_" * 20
    v2_text = shared + b"V2_MEMORY_64GB_" * 20

    p1 = tmp_path / "spec_v1.txt"
    p2 = tmp_path / "spec_v2.txt"
    p1.write_bytes(v1_text)
    p2.write_bytes(v2_text)

    reg = VersionRegistry()
    reg.ingest_document(p1, **_SMALL_PARAMS)
    reg.ingest_document(p2, **_SMALL_PARAMS)

    c_idx = ChunkVersionIndex()
    c_idx.build(reg)

    store = EmbeddingStore(precision="int8")
    store.build(c_idx, batch_size=8)

    assert store.embedding_count == c_idx.unique_chunk_count()
    assert store.embedding_count < c_idx.total_slots()

    # Find a shared chunk
    for cid in c_idx.all_chunk_ids():
        refs = c_idx.get_refs(cid)
        if len(refs) > 1:
            assert store.get_vector(cid) is not None
            break


# ---------------------------------------------------------------------------
# Test 10: End-to-end grounded QA with INT8 storage
# ---------------------------------------------------------------------------


def test_grounded_qa_with_int8(tmp_path: Path):
    p = tmp_path / "system.txt"
    p.write_bytes(
        b"Cluster Specification:\n"
        b"Memory allocation is exactly 64GB ECC RAM per worker node.\n"
        b"Primary gateway port is 9090.\n" * 10
    )

    reg = VersionRegistry()
    v = reg.ingest_document(p, **_SMALL_PARAMS)

    c_idx = ChunkVersionIndex()
    c_idx.build(reg)

    bm25 = BM25Index()
    bm25.build_dedup_aware(c_idx)

    store = EmbeddingStore(precision="int8")
    store.build(c_idx, batch_size=8)

    retriever = HybridRetriever(bm25, store, c_idx)
    mock_provider = MockLLMProvider(
        default_answer="The worker node memory allocation is 64GB ECC RAM [chunk_1]."
    )

    engine = RAGAnswerEngine(
        retriever=retriever,
        version_index=c_idx,
        registry=reg,
        provider=mock_provider,
    )

    ans = engine.answer_question(
        query="What is the worker node memory allocation?",
        doc_id=v.doc_id,
        version_id=v.version_id,
    )

    assert ans.status == "SUCCESS"
    assert "64GB ECC RAM" in ans.answer
    assert len(ans.citations) == 1
    assert ans.citations[0].doc_id == v.doc_id
    assert ans.citations[0].version_id == v.version_id
    assert ans.citations[0].byte_length > 0
