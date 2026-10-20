"""
tests/test_rag_phase7d.py
-------------------------
DCLC Phase 7D Test Suite — Configurable Embedding Precision.

Verifies:
  1. Default configuration: precision defaults to 'float32' (IEEE 754 full precision).
  2. Environment configuration: DCLC_EMBEDDING_PRECISION controls precision across float16, int8, sq8.
  3. Case and whitespace normalization: ' FLOAT16 ', 'SQ8', 'Int8' normalized cleanly.
  4. Invalid configuration rejection: invalid precision strings raise descriptive ValueError listing valid options.
  5. Precedence hierarchy: explicit constructor arguments override environment variables.
  6. Persistence & Serialization: save() and load() round-trip preserves exact precision, matrix, scales, norms, and cosine scores.
  7. API Status Telemetry: GET /rag/status exposes active precision, storage bytes, and compression ratio.
  8. End-to-end question answering and strict version isolation under all precision modes (FP32, FP16, INT8).
  9. Baseline retriever precision configuration and backward compatibility.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.rag_routes import rag_service
from src.rag.config import (
    DEFAULT_PRECISION,
    SUPPORTED_PRECISIONS,
    EmbeddingConfig,
    get_embedding_precision,
    validate_precision,
)
from src.rag.generation.answer_engine import RAGAnswerEngine
from src.rag.generation.provider import MockLLMProvider
from src.rag.retrieval.baseline_retriever import BaselineRetriever
from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.version_mapper import VersionRegistry

_SMALL_PARAMS = dict(min_size=64, avg_size=256, max_size=1024)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure DCLC_EMBEDDING_PRECISION is reset before and after each test."""
    old_val = os.environ.get("DCLC_EMBEDDING_PRECISION")
    if "DCLC_EMBEDDING_PRECISION" in os.environ:
        del os.environ["DCLC_EMBEDDING_PRECISION"]
    yield
    if old_val is not None:
        os.environ["DCLC_EMBEDDING_PRECISION"] = old_val
    elif "DCLC_EMBEDDING_PRECISION" in os.environ:
        del os.environ["DCLC_EMBEDDING_PRECISION"]


@pytest.fixture
def sample_registry(tmp_path: Path) -> VersionRegistry:
    """Build a multi-version sample registry with shared and revised chunks."""
    reg = VersionRegistry()

    v1_content = (
        "Project Orion Architecture Document.\n"
        "Storage allocation is 16GB ECC RAM per worker node.\n"
        "DCLC deduplication handles version delta encoding.\n"
    ) * 4

    v2_content = (
        "Project Orion Architecture Document.\n"
        "Storage allocation is 64GB DDR5 RAM per worker node.\n"
        "DCLC deduplication handles version delta encoding.\n"
    ) * 4

    d1 = tmp_path / "v1"
    d2 = tmp_path / "v2"
    d1.mkdir()
    d2.mkdir()
    p1 = d1 / "orion.txt"
    p2 = d2 / "orion.txt"
    p1.write_text(v1_content, encoding="utf-8")
    p2.write_text(v2_content, encoding="utf-8")

    reg.ingest_document(p1, **_SMALL_PARAMS)
    reg.ingest_document(p2, **_SMALL_PARAMS)
    return reg


# ---------------------------------------------------------------------------
# 1. Default Configuration Tests
# ---------------------------------------------------------------------------


def test_default_config_resolution():
    """Verify default precision is float32 without environment variables."""
    assert get_embedding_precision() == "float32"
    assert DEFAULT_PRECISION == "float32"
    cfg = EmbeddingConfig.from_env()
    assert cfg.precision == "float32"
    assert cfg.embedding_dim == 384


def test_embedding_store_default_is_float32():
    """Verify EmbeddingStore defaults to float32 IEEE representation."""
    store = EmbeddingStore()
    assert store.precision == "float32"
    assert store.compression_ratio_vs_fp32 == 1.0


def test_baseline_retriever_default_is_float32():
    """Verify BaselineRetriever defaults to float32."""
    retriever = BaselineRetriever()
    assert retriever.store.precision == "float32"


# ---------------------------------------------------------------------------
# 2. Environment Variable Configuration Tests
# ---------------------------------------------------------------------------


def test_env_precision_float16(monkeypatch):
    """Verify DCLC_EMBEDDING_PRECISION=float16 configures store and retriever."""
    monkeypatch.setenv("DCLC_EMBEDDING_PRECISION", "float16")
    assert get_embedding_precision() == "float16"

    store = EmbeddingStore()
    assert store.precision == "float16"

    retriever = BaselineRetriever()
    assert retriever.store.precision == "float16"


def test_env_precision_int8(monkeypatch):
    """Verify DCLC_EMBEDDING_PRECISION=int8 configures symmetric SQ8 storage."""
    monkeypatch.setenv("DCLC_EMBEDDING_PRECISION", "int8")
    assert get_embedding_precision() == "int8"

    store = EmbeddingStore()
    assert store.precision == "int8"


def test_env_precision_sq8_alias(monkeypatch):
    """Verify DCLC_EMBEDDING_PRECISION=sq8 alias is normalized to int8."""
    monkeypatch.setenv("DCLC_EMBEDDING_PRECISION", "sq8")
    assert get_embedding_precision() == "int8"

    store = EmbeddingStore()
    assert store.precision == "int8"


def test_env_precision_case_and_whitespace(monkeypatch):
    """Verify normalization strips whitespace and handles mixed case."""
    monkeypatch.setenv("DCLC_EMBEDDING_PRECISION", "  FLOAT16  ")
    assert get_embedding_precision() == "float16"

    monkeypatch.setenv("DCLC_EMBEDDING_PRECISION", " Int8\t")
    assert get_embedding_precision() == "int8"


# ---------------------------------------------------------------------------
# 3. Invalid Configuration Rejection Tests
# ---------------------------------------------------------------------------


def test_invalid_env_precision_raises_error(monkeypatch):
    """Verify invalid environment variable raises descriptive ValueError."""
    monkeypatch.setenv("DCLC_EMBEDDING_PRECISION", "int4")
    with pytest.raises(ValueError, match="Invalid precision 'int4'"):
        get_embedding_precision()

    with pytest.raises(ValueError, match="Invalid precision 'int4'"):
        EmbeddingStore()


def test_invalid_explicit_precision_raises_error():
    """Verify invalid explicit argument raises descriptive ValueError."""
    with pytest.raises(ValueError, match="Invalid precision 'fp64'"):
        validate_precision("fp64")

    with pytest.raises(ValueError, match="Invalid precision 'fp64'"):
        EmbeddingStore(precision="fp64")

    store = EmbeddingStore(precision="float32")
    c_idx = ChunkVersionIndex()
    with pytest.raises(ValueError, match="Invalid precision 'bfloat16'"):
        store.build(c_idx, precision="bfloat16")


# ---------------------------------------------------------------------------
# 4. Precedence Hierarchy Tests
# ---------------------------------------------------------------------------


def test_explicit_argument_overrides_env_var(monkeypatch):
    """Verify explicit constructor argument takes precedence over environment variable."""
    monkeypatch.setenv("DCLC_EMBEDDING_PRECISION", "int8")

    # Explicit override to float32
    store_fp32 = EmbeddingStore(precision="float32")
    assert store_fp32.precision == "float32"

    # Explicit override to float16
    store_fp16 = EmbeddingStore(precision="float16")
    assert store_fp16.precision == "float16"

    # Default None falls back to env var (int8)
    store_env = EmbeddingStore()
    assert store_env.precision == "int8"


# ---------------------------------------------------------------------------
# 5. Persistence & Serialization Tests (save / load)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("precision", ["float32", "float16", "int8"])
def test_embedding_store_save_load_roundtrip(precision: str, sample_registry: VersionRegistry, tmp_path: Path):
    """Verify save() and load() roundtrip preserves precision, metadata, and query scores."""
    c_idx = ChunkVersionIndex()
    c_idx.build(sample_registry)

    original_store = EmbeddingStore(precision=precision)
    original_store.build(c_idx)

    assert original_store.embedding_count > 0
    save_path = tmp_path / f"store_{precision}.npz"
    original_store.save(save_path)
    assert save_path.is_file()

    # Load from disk
    loaded_store = EmbeddingStore.load(save_path)

    assert loaded_store.precision == original_store.precision
    assert loaded_store.embedding_count == original_store.embedding_count
    assert loaded_store.embedding_dim == original_store.embedding_dim
    assert loaded_store.model_name == original_store.model_name
    assert loaded_store.get_chunk_ids() == original_store.get_chunk_ids()
    assert loaded_store.storage_bytes == original_store.storage_bytes

    # Matrix dtype check
    if precision == "float32":
        assert loaded_store._matrix.dtype == np.float32
    elif precision == "float16":
        assert loaded_store._matrix.dtype == np.float16
    elif precision == "int8":
        assert loaded_store._matrix.dtype == np.int8
        assert loaded_store._scales is not None
        assert loaded_store._int_norms is not None
        np.testing.assert_allclose(loaded_store._int_norms, original_store._int_norms, rtol=1e-5)

    # Cosine score fidelity check
    query = "Storage allocation RAM per worker node"
    q_vec = original_store.query_vector(query)
    orig_scores = original_store.cosine_scores(q_vec)
    loaded_scores = loaded_store.cosine_scores(q_vec)

    assert orig_scores.keys() == loaded_scores.keys()
    for cid in orig_scores:
        assert abs(orig_scores[cid] - loaded_scores[cid]) < 1e-6


def test_save_unbuilt_store_raises_runtime_error(tmp_path: Path):
    """Verify attempting to save an unbuilt store raises RuntimeError."""
    store = EmbeddingStore()
    with pytest.raises(RuntimeError, match="Cannot save an unbuilt EmbeddingStore"):
        store.save(tmp_path / "unbuilt.npz")


def test_load_nonexistent_file_raises_filenotfound(tmp_path: Path):
    """Verify loading non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Embedding store file not found"):
        EmbeddingStore.load(tmp_path / "does_not_exist.npz")


# ---------------------------------------------------------------------------
# 6. API Status Telemetry Tests
# ---------------------------------------------------------------------------


def test_rag_status_telemetry_defaults():
    """Verify GET /rag/status returns precision and storage telemetry."""
    client = TestClient(app)

    # Reset service state
    rag_service.registry = VersionRegistry()
    rag_service.chunk_index = ChunkVersionIndex()
    rag_service.bm25_index = BM25Index()
    rag_service.embedding_store = EmbeddingStore(precision="float32")

    resp = client.get("/rag/status")
    assert resp.status_code == 200
    data = resp.json()

    assert data["embedding_precision"] == "float32"
    assert data["storage_bytes"] == 0
    assert data["compression_ratio_vs_fp32"] == 1.0


def test_rag_service_configured_with_precision(sample_registry: VersionRegistry, monkeypatch):
    """Verify RAGBackendService respects configured precision in status."""
    monkeypatch.setenv("DCLC_EMBEDDING_PRECISION", "int8")

    rag_service.registry = sample_registry
    rag_service.chunk_index = ChunkVersionIndex()
    rag_service.chunk_index.build(sample_registry)
    rag_service.bm25_index = BM25Index()
    rag_service.bm25_index.build_dedup_aware(rag_service.chunk_index)
    rag_service.embedding_store = EmbeddingStore()  # picks up int8 from env
    rag_service.embedding_store.build(rag_service.chunk_index)

    status = rag_service.get_status()
    assert status.embedding_precision == "int8"
    assert status.storage_bytes > 0
    # INT8 compression ratio should be approximately 3.96x
    assert status.compression_ratio_vs_fp32 >= 3.5


# ---------------------------------------------------------------------------
# 7. End-to-End Grounded QA & Strict Version Isolation across Precisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("precision", ["float32", "float16", "int8"])
def test_end_to_end_qa_version_isolation(precision: str, sample_registry: VersionRegistry):
    """
    Verify grounded QA and strict version isolation function identically
    across all three precision modes.
    """
    c_idx = ChunkVersionIndex()
    c_idx.build(sample_registry)

    bm25 = BM25Index()
    bm25.build_dedup_aware(c_idx)

    store = EmbeddingStore(precision=precision)
    store.build(c_idx)

    retriever = HybridRetriever(bm25, store, c_idx)
    provider = MockLLMProvider()
    engine = RAGAnswerEngine(
        retriever=retriever,
        version_index=c_idx,
        registry=sample_registry,
        provider=provider,
    )

    doc_id = sample_registry.list_doc_ids()[0]
    versions = sample_registry.get_all_versions(doc_id)
    v1_id = versions[0].version_id
    v2_id = versions[1].version_id

    # 1. Query scoped to Version 1: MUST retrieve only Version 1 chunks
    ans_v1 = engine.answer_question(
        query="What is the storage allocation RAM?",
        doc_id=doc_id,
        version_id=v1_id,
    )
    assert ans_v1.status == "SUCCESS"
    assert len(ans_v1.citations) > 0
    for cit in ans_v1.citations:
        assert cit.version_id == v1_id
        assert cit.is_valid is True

    # 2. Query scoped to Version 2: MUST retrieve only Version 2 chunks
    ans_v2 = engine.answer_question(
        query="What is the storage allocation RAM?",
        doc_id=doc_id,
        version_id=v2_id,
    )
    assert ans_v2.status == "SUCCESS"
    assert len(ans_v2.citations) > 0
    for cit in ans_v2.citations:
        assert cit.version_id == v2_id
        assert cit.is_valid is True
