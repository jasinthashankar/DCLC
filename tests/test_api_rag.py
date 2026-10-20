"""
tests/test_api_rag.py
---------------------
API integration test suite for DCLC-RAG endpoints (Phase 6E).

Verifies:
  1. GET /rag/status returns initial empty/ready state with model and provider information.
  2. POST /rag/ingest accepts valid .txt, .md, .py files and returns chunk metadata.
  3. POST /rag/ingest rejects unsupported file extensions (.pdf, .bin) with HTTP 400.
  4. POST /rag/ingest with valid doc_id registers a new version for that document.
  5. POST /rag/ingest with invalid doc_id returns HTTP 404.
  6. GET /rag/documents returns registered documents, version details, and chunk counts.
  7. POST /rag/query executes grounded QA and returns citations with verified metadata.
  8. POST /rag/query enforces strict version isolation between Version 1 and Version 2.
  9. POST /rag/query abstains gracefully when requested content is not in evidence.
  10. POST /rag/query rejects empty query strings with HTTP 400.
"""

from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.rag_routes import rag_service
from src.rag.generation.provider import MockLLMProvider
from src.rag.version_mapper import VersionRegistry
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.embedding_store import EmbeddingStore


@pytest.fixture(autouse=True)
def reset_rag_service():
    """Reset the singleton rag_service before each test to guarantee test isolation."""
    rag_service.registry = VersionRegistry()
    rag_service.chunk_index = ChunkVersionIndex()
    rag_service.bm25_index = BM25Index()
    rag_service.embedding_store = EmbeddingStore()
    rag_service.retriever = None
    rag_service.engine = None
    rag_service.provider = MockLLMProvider(
        default_answer="Grounded answer based on evidence [chunk_1]."
    )
    yield


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: RAG Status on Empty Service
# ---------------------------------------------------------------------------


def test_rag_status_empty(client: TestClient):
    resp = client.get("/rag/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "empty"
    assert data["total_documents"] == 0
    assert data["total_versions"] == 0
    assert data["total_unique_chunks"] == 0
    assert data["provider_name"] == "mock"


# ---------------------------------------------------------------------------
# Test 2: Ingest Valid .txt File
# ---------------------------------------------------------------------------


def test_rag_ingest_txt(client: TestClient):
    content = (
        "Cloud Storage Cluster Specification Release 1.0.\n"
        "Memory allocation is 16GB ECC RAM per worker node.\n"
        "Storage compression is configured with standard LZ4 algorithm.\n"
    ) * 8
    file_bytes = content.encode("utf-8")

    resp = client.post(
        "/rag/ingest",
        files={"file": ("system_spec.txt", io.BytesIO(file_bytes), "text/plain")},
        data={"min_chunk_size": 64, "avg_chunk_size": 256, "max_chunk_size": 1024},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "system_spec.txt"
    assert data["extension"] == ".txt"
    assert data["chunk_count"] > 0
    assert data["total_bytes"] == len(file_bytes)
    assert data["is_new_document"] is True
    assert "doc_id" in data
    assert "version_id" in data


# ---------------------------------------------------------------------------
# Test 3: Reject Unsupported Extension
# ---------------------------------------------------------------------------


def test_rag_ingest_unsupported_extension(client: TestClient):
    resp = client.post(
        "/rag/ingest",
        files={"file": ("unsupported.pdf", io.BytesIO(b"%PDF-1.4 dummy"), "application/pdf")},
    )
    assert resp.status_code == 400
    assert "Unsupported file extension" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Test 4: Ingest Multi-Version Document
# ---------------------------------------------------------------------------


def test_rag_ingest_multi_version(client: TestClient):
    # Ingest Version 1
    v1_bytes = (
        "Documentation v1: Port is 8080.\n"
        "Worker pool size is 10 nodes.\n"
    ).encode("utf-8") * 10

    resp1 = client.post(
        "/rag/ingest",
        files={"file": ("app_config.md", io.BytesIO(v1_bytes), "text/markdown")},
        data={"min_chunk_size": 64, "avg_chunk_size": 256, "max_chunk_size": 1024},
    )
    assert resp1.status_code == 200
    doc_id = resp1.json()["doc_id"]
    v1_id = resp1.json()["version_id"]

    # Ingest Version 2 for the same document
    v2_bytes = (
        "Documentation v2: Port is upgraded to 9090.\n"
        "Worker pool size is 25 nodes.\n"
    ).encode("utf-8") * 10

    resp2 = client.post(
        "/rag/ingest",
        files={"file": ("app_config_updated.md", io.BytesIO(v2_bytes), "text/markdown")},
        data={
            "doc_id": doc_id,
            "min_chunk_size": 64,
            "avg_chunk_size": 256,
            "max_chunk_size": 1024,
        },
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["doc_id"] == doc_id
    assert data2["version_id"] != v1_id
    assert data2["is_new_document"] is False

    # Check documents listing
    list_resp = client.get("/rag/documents")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total_documents"] == 1
    assert list_data["total_versions"] == 2
    doc = list_data["documents"][0]
    assert doc["doc_id"] == doc_id
    assert len(doc["versions"]) == 2


# ---------------------------------------------------------------------------
# Test 5: Ingest with Non-existent doc_id returns 404
# ---------------------------------------------------------------------------


def test_rag_ingest_invalid_doc_id(client: TestClient):
    resp = client.post(
        "/rag/ingest",
        files={"file": ("test.txt", io.BytesIO(b"Hello world!"), "text/plain")},
        data={"doc_id": "nonexistent_doc_123"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 6: Query Grounded QA with Verified Citations
# ---------------------------------------------------------------------------


def test_rag_query_grounded(client: TestClient):
    # Ingest document
    text = (
        "Architecture Notes:\n"
        "The authentication secret key is stored in AWS Secrets Manager.\n"
        "Session timeout is configured for exactly 3600 seconds.\n"
    ) * 10
    resp_ingest = client.post(
        "/rag/ingest",
        files={"file": ("arch.txt", io.BytesIO(text.encode("utf-8")), "text/plain")},
        data={"min_chunk_size": 64, "avg_chunk_size": 256, "max_chunk_size": 1024},
    )
    assert resp_ingest.status_code == 200
    doc_id = resp_ingest.json()["doc_id"]

    # Provide mock answer citing chunk_1
    rag_service.provider = MockLLMProvider(
        default_answer="The session timeout is 3600 seconds [chunk_1]."
    )
    rag_service.engine.provider = rag_service.provider

    query_resp = client.post(
        "/rag/query",
        json={
            "query": "What is the session timeout?",
            "doc_id": doc_id,
            "top_k": 3,
            "alpha": 0.5,
        },
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert q_data["status"] == "SUCCESS"
    assert "3600 seconds" in q_data["answer"]
    assert len(q_data["citations"]) > 0
    cit = q_data["citations"][0]
    assert cit["tag"] == "[chunk_1]"
    assert cit["doc_id"] == doc_id
    assert cit["byte_length"] > 0


# ---------------------------------------------------------------------------
# Test 7: Strict Version Isolation in Queries
# ---------------------------------------------------------------------------


def test_rag_query_version_isolation(client: TestClient):
    v1_text = ("Release 1: Port is 8080. DB is PostgreSQL 14.\n") * 15
    resp1 = client.post(
        "/rag/ingest",
        files={"file": ("service.txt", io.BytesIO(v1_text.encode("utf-8")), "text/plain")},
        data={"min_chunk_size": 64, "avg_chunk_size": 256, "max_chunk_size": 1024},
    )
    doc_id = resp1.json()["doc_id"]
    v1_id = resp1.json()["version_id"]

    v2_text = ("Release 2: Port is 9090. DB is PostgreSQL 16.\n") * 15
    resp2 = client.post(
        "/rag/ingest",
        files={"file": ("service.txt", io.BytesIO(v2_text.encode("utf-8")), "text/plain")},
        data={"doc_id": doc_id, "min_chunk_size": 64, "avg_chunk_size": 256, "max_chunk_size": 1024},
    )
    v2_id = resp2.json()["version_id"]

    # When querying v1, mock provider answers from v1 evidence
    rag_service.provider = MockLLMProvider(
        default_answer="In version 1, Port is 8080 [chunk_1]."
    )
    rag_service.engine.provider = rag_service.provider

    q_v1 = client.post(
        "/rag/query",
        json={
            "query": "What port is used?",
            "doc_id": doc_id,
            "version_id": v1_id,
        },
    )
    assert q_v1.status_code == 200
    data_v1 = q_v1.json()
    assert data_v1["status"] == "SUCCESS"
    assert data_v1["target_version_id"] == v1_id
    for c in data_v1["citations"]:
        assert c["version_id"] == v1_id
        assert "9090" not in c["snippet"]


# ---------------------------------------------------------------------------
# Test 8: Abstention on Missing Evidence
# ---------------------------------------------------------------------------


def test_rag_query_abstain(client: TestClient):
    text = ("Python script defining a simple web server.\n") * 10
    client.post(
        "/rag/ingest",
        files={"file": ("server.py", io.BytesIO(text.encode("utf-8")), "text/x-python")},
        data={"min_chunk_size": 64, "avg_chunk_size": 256, "max_chunk_size": 1024},
    )

    # Set mock answer to signal insufficient evidence
    rag_service.provider = MockLLMProvider(
        default_answer="INSUFFICIENT_EVIDENCE: The document contains no information about aerospace propulsion."
    )
    rag_service.engine.provider = rag_service.provider

    query_resp = client.post(
        "/rag/query",
        json={"query": "What is the specific impulse of the rocket engine?"},
    )
    assert query_resp.status_code == 200
    data = query_resp.json()
    assert data["status"] == "ABSTAIN"
    assert "INSUFFICIENT_EVIDENCE" in data["answer"]


# ---------------------------------------------------------------------------
# Test 9: Reject Empty Query
# ---------------------------------------------------------------------------


def test_rag_query_empty_string(client: TestClient):
    resp = client.post("/rag/query", json={"query": "   "})
    assert resp.status_code == 400
    assert "must not be empty" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 10: Query Non-existent Version returns 404
# ---------------------------------------------------------------------------


def test_rag_query_nonexistent_version(client: TestClient):
    text = ("Some random content.\n") * 10
    client.post(
        "/rag/ingest",
        files={"file": ("doc.txt", io.BytesIO(text.encode("utf-8")), "text/plain")},
    )
    resp = client.post(
        "/rag/query",
        json={"query": "hello", "version_id": "nonexistent_version_sha"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
