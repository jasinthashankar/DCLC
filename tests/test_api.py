from fastapi.testclient import TestClient

from src.dclc.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_endpoint_reports_baseline_method():
    response = client.post(
        "/api/v1/analyze?chunk_size=4",
        files={
            "source_file": ("source.bin", b"AAAABBBBCCCC", "application/octet-stream"),
            "target_file": ("target.bin", b"AAAAXXXXCCCC", "application/octet-stream"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "fixed_size_sha256_baseline"
    assert body["research_status"] == "baseline_only"
    assert body["result"]["matched_target_bytes"] == 8


def test_chunk_size_out_of_range_is_rejected():
    response = client.post(
        "/api/v1/analyze?chunk_size=0",
        files={
            "source_file": ("source.bin", b"abc", "application/octet-stream"),
            "target_file": ("target.bin", b"abc", "application/octet-stream"),
        },
    )
    assert response.status_code == 422
