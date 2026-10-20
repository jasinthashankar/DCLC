"""Tests for DCLC FastAPI Backend Endpoints (Phase 5A).

Tests all endpoints:
- GET /health
- POST /chunk-preview
- POST /compare-revisions (Identical files, modified revisions, different candidate strategies)
- GET /benchmark-storage and POST /benchmark-storage
- Error handling: Oversized payload (413), Invalid parameters (400), Degraded health check.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app, service
from api.service import DCLCBackendService


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Verify GET / returns welcome message and documentation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["documentation"] == "/docs"
    assert data["health"] == "/health"
    assert "endpoints" in data


def test_health_endpoint(client: TestClient):
    """Verify /health returns healthy status, active features, and model availability."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("healthy", "degraded_missing_checkpoint")
    assert "checkpoint_available" in data
    assert "active_features" in data
    assert len(data["active_features"]) >= 5
    assert data["device"] == "cpu"
    if data["checkpoint_available"]:
        assert data["status"] == "healthy"
        assert "Trained Context-Aware Transformer Inference" in data["active_features"]


def test_chunk_preview_valid_file(client: TestClient):
    """Verify /chunk-preview chunks payload and calculates valid metadata."""
    sample_content = b"DCLC research project test content. " * 300  # ~10.8 KB
    files = {"file": ("sample.txt", sample_content, "text/plain")}

    response = client.post(
        "/chunk-preview",
        files=files,
        params={"min_chunk_size": 2048, "avg_chunk_size": 4096, "max_chunk_size": 8192},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "sample.txt"
    assert data["total_bytes"] == len(sample_content)
    assert data["chunk_count"] >= 1
    assert len(data["chunks"]) == data["chunk_count"]

    first_chunk = data["chunks"][0]
    assert first_chunk["chunk_index"] == 0
    assert first_chunk["byte_offset"] == 0
    assert len(first_chunk["sha256"]) == 64  # SHA-256 hex string


def test_chunk_preview_empty_file(client: TestClient):
    """Verify /chunk-preview handles 0-byte file gracefully."""
    files = {"file": ("empty.bin", b"", "application/octet-stream")}
    response = client.post("/chunk-preview", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_bytes"] == 0
    assert data["chunk_count"] == 0
    assert data["chunks"] == []


def test_chunk_preview_invalid_constraints(client: TestClient):
    """Verify /chunk-preview rejects min_chunk_size > avg_chunk_size with 400 Bad Request."""
    files = {"file": ("test.bin", b"some bytes", "application/octet-stream")}
    response = client.post(
        "/chunk-preview",
        files=files,
        params={"min_chunk_size": 8192, "avg_chunk_size": 4096, "max_chunk_size": 16384},
    )
    assert response.status_code == 400
    assert "Chunk size constraints violated" in response.json()["detail"]


def test_chunk_preview_file_too_large(client: TestClient, monkeypatch):
    """Verify /chunk-preview enforces max upload limit with 413 Request Entity Too Large."""
    import api.main
    monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", 100)

    oversized_content = b"A" * 150
    files = {"file": ("large.txt", oversized_content, "text/plain")}
    response = client.post("/chunk-preview", files=files)
    assert response.status_code == 413
    assert "exceeds the maximum allowed limit" in response.json()["detail"]


def test_compare_revisions_identical_files(client: TestClient):
    """Verify /compare-revisions marks all chunks as exact duplicates when files match."""
    # Generate ~20KB synthetic content
    content = (b"DCLC Cloud Deduplication Framework revision content block.\n" * 400)
    files = {
        "base_file": ("v1.txt", content, "text/plain"),
        "target_file": ("v2.txt", content, "text/plain"),
    }
    data_form = {
        "candidate_strategy": "stream_locality",
        "resemblance_threshold": 0.70,
        "min_chunk_size": 2048,
        "avg_chunk_size": 4096,
        "max_chunk_size": 8192,
    }

    response = client.post("/compare-revisions", files=files, data=data_form)
    assert response.status_code == 200
    data = response.json()

    assert data["base_filename"] == "v1.txt"
    assert data["target_filename"] == "v2.txt"
    assert data["target_chunk_count"] > 0
    assert data["exact_duplicates_count"] == data["target_chunk_count"]
    assert data["resemblance_pairs_count"] == 0
    assert data["target_dclc_stored_bytes"] == 32 * data["target_chunk_count"]
    assert data["target_storage_savings_pct"] > 99.0


def test_compare_revisions_modified_files_stream_locality(client: TestClient):
    """Verify /compare-revisions on modified revisions with stream_locality indexer."""
    base_content = (b"Version 1 line entry: Alpha Bravo Charlie Delta Echo Foxtrot.\n" * 500)
    # Modify 20% of lines in target
    target_content = (b"Version 2 MODIFIED line entry: Alpha Bravo Charlie Delta Echo Foxtrot.\n" * 100) + \
                     (b"Version 1 line entry: Alpha Bravo Charlie Delta Echo Foxtrot.\n" * 400)

    files = {
        "base_file": ("base.txt", base_content, "text/plain"),
        "target_file": ("target.txt", target_content, "text/plain"),
    }
    data_form = {
        "candidate_strategy": "stream_locality",
        "resemblance_threshold": 0.70,
        "min_chunk_size": 2048,
        "avg_chunk_size": 4096,
        "max_chunk_size": 8192,
    }

    response = client.post("/compare-revisions", files=files, data=data_form)
    assert response.status_code == 200
    data = response.json()

    assert data["target_chunk_count"] >= 1
    assert data["reconstruction_success_rate"] == 100.0
    assert data["target_dclc_stored_bytes"] < data["target_raw_bytes"]
    assert len(data["detailed_chunk_comparisons"]) == data["target_chunk_count"]


def test_compare_revisions_strategies(client: TestClient):
    """Verify candidate strategies rp_lsh and embedding_knn execute properly."""
    base_content = (b"Block sequence record base: data content segment for testing.\n" * 300)
    target_content = (b"Block sequence record target: data content segment for testing.\n" * 300)

    for strat in ["rp_lsh", "embedding_knn"]:
        files = {
            "base_file": ("base.txt", base_content, "text/plain"),
            "target_file": ("target.txt", target_content, "text/plain"),
        }
        data_form = {
            "candidate_strategy": strat,
            "resemblance_threshold": 0.60,
            "min_chunk_size": 2048,
            "avg_chunk_size": 4096,
            "max_chunk_size": 8192,
        }
        response = client.post("/compare-revisions", files=files, data=data_form)
        assert response.status_code == 200
        data = response.json()
        assert data["candidate_strategy"] in (strat, "lsh", "knn")
        assert data["reconstruction_success_rate"] == 100.0


def test_compare_revisions_invalid_strategy(client: TestClient):
    """Verify /compare-revisions rejects unrecognized candidate strategy."""
    files = {
        "base_file": ("base.txt", b"abc", "text/plain"),
        "target_file": ("target.txt", b"xyz", "text/plain"),
    }
    data_form = {"candidate_strategy": "non_existent_strategy"}
    response = client.post("/compare-revisions", files=files, data=data_form)
    assert response.status_code == 400
    assert "Invalid candidate_strategy" in response.json()["detail"]


def test_benchmark_storage_get_and_post(client: TestClient):
    """Verify /benchmark-storage executes over the held-out test split."""
    # Test GET
    response_get = client.get("/benchmark-storage", params={"resemblance_threshold": 0.70})
    assert response_get.status_code == 200
    data_get = response_get.json()

    assert data_get["split_evaluated"] == "test"
    assert data_get["total_chunks_evaluated"] == 464
    assert "system_0_raw" in data_get
    assert "system_1_standalone_compression" in data_get
    assert "system_2_exact_dedup" in data_get
    assert "system_3_full_dclc" in data_get

    sys3 = data_get["system_3_full_dclc"]
    assert len(sys3) >= 1
    for indexer_name, res in sys3.items():
        assert res["stored_bytes"] < data_get["system_2_exact_dedup"]["stored_bytes"]
        assert res["reconstruction_success_rate"] == 100.0

    # Test POST
    response_post = client.post("/benchmark-storage", params={"resemblance_threshold": 0.70})
    assert response_post.status_code == 200
    assert response_post.json()["split_evaluated"] == "test"


def test_missing_checkpoint_service_behavior():
    """Verify service behavior when model checkpoint is not present."""
    fake_service = DCLCBackendService(checkpoint_path="nonexistent_checkpoint.pt")
    health = fake_service.check_health()
    assert health.status == "degraded_missing_checkpoint"
    assert health.checkpoint_available is False
    assert "Trained Context-Aware Transformer Inference" not in health.active_features

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        fake_service.get_model()


def test_compare_revisions_hierarchical_cache(client: TestClient):
    """Verify /compare-revisions with hierarchical_cache strategy succeeds and returns stats."""
    base_content = (b"Hierarchical Cache evaluation block: Line Alpha Beta Gamma Delta.\n" * 400)
    target_content = (b"Hierarchical Cache MODIFIED block: Line Alpha Beta Gamma Delta.\n" * 100) + \
                     (b"Hierarchical Cache evaluation block: Line Alpha Beta Gamma Delta.\n" * 300)

    files = {
        "base_file": ("base.txt", base_content, "text/plain"),
        "target_file": ("target.txt", target_content, "text/plain"),
    }
    data_form = {
        "candidate_strategy": "hierarchical_cache",
        "resemblance_threshold": 0.65,
        "min_chunk_size": 2048,
        "avg_chunk_size": 4096,
        "max_chunk_size": 8192,
        "delta_mode": "single",
    }
    response = client.post("/compare-revisions", files=files, data=data_form)
    assert response.status_code == 200
    data = response.json()

    assert data["candidate_strategy"] == "hierarchical_cache"
    assert data["reconstruction_success_rate"] == 100.0
    assert data["target_chunk_count"] >= 1
    assert data["indexer_stats"] is not None
    assert "cache_hits" in data["indexer_stats"]
    assert "cache_misses" in data["indexer_stats"]
    assert "ann_queries" in data["indexer_stats"]


def test_compare_revisions_mrd_mode(client: TestClient):
    """Verify /compare-revisions executes Multi-Reference Delta (MRD) mode with bit-exact restoration."""
    base_content = (b"MRD Pipeline Test Base Block: Segment One Two Three Four Five.\n" * 400)
    target_content = (b"MRD Pipeline Test Target Block: Segment One Two Three Four Five MODIFIED.\n" * 100) + \
                     (b"MRD Pipeline Test Base Block: Segment One Two Three Four Five.\n" * 300)

    files = {
        "base_file": ("base.txt", base_content, "text/plain"),
        "target_file": ("target.txt", target_content, "text/plain"),
    }
    data_form = {
        "candidate_strategy": "hierarchical_cache",
        "resemblance_threshold": 0.65,
        "min_chunk_size": 2048,
        "avg_chunk_size": 4096,
        "max_chunk_size": 8192,
        "delta_mode": "multi",
        "mrd_window_size": 3,
        "mrd_max_refs": 3,
    }
    response = client.post("/compare-revisions", files=files, data=data_form)
    assert response.status_code == 200
    data = response.json()

    assert data["delta_mode"] == "multi"
    assert data["mrd_window_size"] == 3
    assert data["mrd_max_refs"] == 3
    assert data["mrd_groups_count"] >= 1
    assert data["reconstruction_success_rate"] == 100.0
    assert data["target_dclc_stored_bytes"] < data["target_raw_bytes"]


def test_compare_revisions_invalid_delta_mode(client: TestClient):
    """Verify /compare-revisions rejects unrecognized delta_mode with 400 Bad Request."""
    files = {
        "base_file": ("base.txt", b"abc", "text/plain"),
        "target_file": ("target.txt", b"xyz", "text/plain"),
    }
    data_form = {
        "delta_mode": "invalid_mode",
    }
    response = client.post("/compare-revisions", files=files, data=data_form)
    assert response.status_code == 400
    assert "Invalid delta_mode" in response.json()["detail"]


def test_benchmark_storage_with_mrd(client: TestClient):
    """Verify /benchmark-storage with enable_mrd=True includes system_4_dclc_mrd."""
    response = client.get(
        "/benchmark-storage",
        params={"resemblance_threshold": 0.70, "enable_mrd": True},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["split_evaluated"] == "test"
    assert "system_4_dclc_mrd" in data
    assert data["system_4_dclc_mrd"] is not None
    mrd_data = data["system_4_dclc_mrd"]
    assert "single_reference_sr" in mrd_data
    assert "multi_reference_mrd" in mrd_data
    assert "restoration_throughput_comparison" in mrd_data

