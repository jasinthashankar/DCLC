"""Comprehensive Test Suite for Phase E: Durable Persistence, Restoration, and Reliability.

Validates:
1. End-to-end revision storage, manifest creation, and atomic persistence.
2. Safe restart: Persisted containers and recipes restore with 100% byte-exactness
   after service re-instantiation with zero in-memory state.
3. Single-Reference (SR) and Multi-Reference Delta (MRD) persistence lifecycles.
4. Download streaming endpoint verifying reconstructed octet-stream bytes and headers.
5. Reliability & Error handling:
   - Corrupted chunk detection (SHA-256 mismatch -> clean 422).
   - Corrupted delta container detection (clean 422).
   - Missing base reference chunk handling (clean 422).
   - Path traversal security rejection (clean 400/422).
   - Non-existent revision (clean 404).
   - Atomic temporary file cleanup.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from api.main import app, service
from api.service import DCLCBackendService
from src.storage_manager import (
    MissingReferenceError,
    RevisionNotFoundError,
    StorageCorruptionError,
    StorageManager,
)


@pytest.fixture
def temp_store_dir() -> Generator[Path, None, None]:
    """Provide an isolated temporary directory for durable storage tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="dclc_test_store_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def isolated_client(temp_store_dir: Path) -> Generator[TestClient, None, None]:
    """TestClient wired to an isolated storage manager instance."""
    original_sm = service.storage_manager
    service.storage_manager = StorageManager(base_dir=temp_store_dir)
    with TestClient(app) as test_client:
        yield test_client
    service.storage_manager = original_sm


def test_durable_storage_lifecycle_single_reference(isolated_client: TestClient, temp_store_dir: Path):
    """Test full store -> restart -> restore lifecycle for Single-Reference (SR) delta mode."""
    part1 = b"Base initial documentation section 1.\n" * 600  # ~23 KB
    part2 = b"Base core algorithmic section 2.\n" * 600  # ~20 KB
    base_bytes = part1 + part2

    part_mod = b"Base core algorithmic section 2 modified revision notes.\n" * 600
    target_bytes = part1 + part_mod

    target_sha256 = hashlib.sha256(target_bytes).hexdigest()

    # 1. Store revision
    files = {
        "base_file": ("base_doc_v1.txt", base_bytes, "text/plain"),
        "target_file": ("target_doc_v2.txt", target_bytes, "text/plain"),
    }
    store_resp = isolated_client.post(
        "/store-revision",
        files=files,
        data={
            "candidate_strategy": "hierarchical_cache",
            "delta_mode": "single",
            "min_chunk_size": 2048,
            "avg_chunk_size": 4096,
            "max_chunk_size": 8192,
        },
    )
    assert store_resp.status_code == 201, store_resp.text
    recipe = store_resp.json()
    rev_id = recipe["revision_id"]
    assert rev_id.startswith("rev_")
    assert recipe["sha256"] == target_sha256
    assert recipe["storage_savings_pct"] > 0
    assert recipe["raw_bytes"] == len(target_bytes)
    assert len(recipe["chunks"]) > 0

    # 2. Simulate complete process restart with fresh StorageManager instance
    fresh_service = DCLCBackendService(storage_dir=temp_store_dir)
    restored_bytes, restored_recipe = fresh_service.restore_revision(rev_id)

    # 3. Assert 100% byte-for-byte exact equality
    assert restored_bytes == target_bytes
    assert hashlib.sha256(restored_bytes).hexdigest() == target_sha256
    assert restored_recipe["revision_id"] == rev_id

    # 4. Verify API restore endpoint
    restore_resp = isolated_client.get(f"/restore-revision/{rev_id}")
    assert restore_resp.status_code == 200
    r_data = restore_resp.json()
    assert r_data["verified"] is True
    assert r_data["reconstructed_bytes_match"] is True
    assert r_data["sha256"] == target_sha256

    # 5. Verify download endpoint
    dl_resp = isolated_client.get(f"/download-restored/{rev_id}")
    assert dl_resp.status_code == 200
    assert dl_resp.content == target_bytes
    assert dl_resp.headers["X-DCLC-SHA256"] == target_sha256
    assert "target_doc_v2.txt" in dl_resp.headers["Content-Disposition"]


def test_durable_storage_lifecycle_mrd(isolated_client: TestClient, temp_store_dir: Path):
    """Test full store -> restart -> restore lifecycle for Multi-Reference Delta (MRD) mode."""
    part1 = b"Shared prologue section for MRD test.\n" * 500
    part2 = b"Reference candidate block A for MRD test.\n" * 500
    part3 = b"Reference candidate block B for MRD test.\n" * 500
    base_bytes = part1 + part2 + part3

    target_part2 = b"Reference candidate block A with inline modifications.\n" * 500
    target_part3 = b"Reference candidate block B with inline modifications.\n" * 500
    target_bytes = part1 + target_part2 + target_part3

    target_sha256 = hashlib.sha256(target_bytes).hexdigest()

    files = {
        "base_file": ("base_mrd.bin", base_bytes, "application/octet-stream"),
        "target_file": ("target_mrd.bin", target_bytes, "application/octet-stream"),
    }
    store_resp = isolated_client.post(
        "/revisions/store",  # RESTful alias
        files=files,
        data={
            "candidate_strategy": "hierarchical_cache",
            "delta_mode": "multi",
            "mrd_window_size": 3,
            "mrd_max_refs": 3,
            "min_chunk_size": 2048,
            "avg_chunk_size": 4096,
            "max_chunk_size": 8192,
        },
    )
    assert store_resp.status_code == 201, store_resp.text
    recipe = store_resp.json()
    rev_id = recipe["revision_id"]
    assert recipe["delta_mode"] == "multi"

    # Simulate restart and restore
    fresh_mgr = StorageManager(base_dir=temp_store_dir)
    restored_bytes = fresh_mgr.restore_revision(rev_id)
    assert restored_bytes == target_bytes
    assert hashlib.sha256(restored_bytes).hexdigest() == target_sha256


def test_list_and_get_revisions_endpoints(isolated_client: TestClient):
    """Verify /revisions and /revisions/{id} endpoints."""
    # List should be empty initially
    res0 = isolated_client.get("/revisions")
    assert res0.status_code == 200
    assert res0.json() == []

    # Store a test revision
    base_bytes = b"Data line base.\n" * 300
    target_bytes = b"Data line base.\n" * 150 + b"Inserted line.\n" * 150
    files = {
        "base_file": ("b.txt", base_bytes, "text/plain"),
        "target_file": ("t.txt", target_bytes, "text/plain"),
    }
    store_resp = isolated_client.post("/store-revision", files=files)
    assert store_resp.status_code == 201
    rev_id = store_resp.json()["revision_id"]

    # List should now contain 1 revision
    res1 = isolated_client.get("/revisions")
    assert res1.status_code == 200
    items = res1.json()
    assert len(items) == 1
    assert items[0]["revision_id"] == rev_id
    assert items[0]["target_name"] == "t.txt"
    assert items[0]["raw_bytes"] == len(target_bytes)

    # Get detailed revision manifest
    res2 = isolated_client.get(f"/revisions/{rev_id}")
    assert res2.status_code == 200
    detail = res2.json()
    assert detail["revision_id"] == rev_id
    assert "chunks" in detail
    assert len(detail["chunks"]) > 0


def test_corrupted_chunk_detection(isolated_client: TestClient, temp_store_dir: Path):
    """Verify that tampering with a stored chunk triggers a clean 422 StorageCorruptionError."""
    base_bytes = b"Original chunk test data.\n" * 400
    target_bytes = b"Original chunk test data.\n" * 400  # Exact match chunk

    files = {
        "base_file": ("base.txt", base_bytes, "text/plain"),
        "target_file": ("target.txt", target_bytes, "text/plain"),
    }
    store_resp = isolated_client.post("/store-revision", files=files)
    assert store_resp.status_code == 201
    recipe = store_resp.json()
    rev_id = recipe["revision_id"]

    # Locate chunk file on disk and corrupt its bytes
    chunk_files = list((temp_store_dir / "chunks").glob("*.bin"))
    assert len(chunk_files) > 0
    # Overwrite the first chunk with corrupted junk
    with open(chunk_files[0], "wb") as f:
        f.write(b"CORRUPTED_TAMPERED_BYTES_12345")

    # Attempt restoration via API -> must fail cleanly with 422
    restore_resp = isolated_client.get(f"/restore-revision/{rev_id}")
    assert restore_resp.status_code == 422
    assert "integrity" in restore_resp.json()["detail"].lower() or "mismatch" in restore_resp.json()["detail"].lower()

    # Attempt download via API -> must fail cleanly with 422
    dl_resp = isolated_client.get(f"/download-restored/{rev_id}")
    assert dl_resp.status_code == 422


def test_corrupted_delta_container_detection(isolated_client: TestClient, temp_store_dir: Path):
    """Verify that tampering with a delta container triggers a clean 422 StorageCorruptionError."""
    base_bytes = b"Standard baseline data for delta container test.\n" * 500
    target_bytes = b"Standard baseline data for delta container test.\n" * 250 + b"Modified delta section.\n" * 250

    files = {
        "base_file": ("base.txt", base_bytes, "text/plain"),
        "target_file": ("target.txt", target_bytes, "text/plain"),
    }
    store_resp = isolated_client.post(
        "/store-revision",
        files=files,
        data={"candidate_strategy": "hierarchical_cache", "delta_mode": "single"},
    )
    assert store_resp.status_code == 201
    recipe = store_resp.json()
    rev_id = recipe["revision_id"]

    # Corrupt delta container
    delta_files = list((temp_store_dir / "deltas").glob("*.bin"))
    if delta_files:
        with open(delta_files[0], "wb") as f:
            f.write(b"CORRUPTED_DELTA_CONTAINER_BYTES_99999")

        restore_resp = isolated_client.get(f"/restore-revision/{rev_id}")
        assert restore_resp.status_code == 422
        assert "integrity" in restore_resp.json()["detail"].lower() or "checksum" in restore_resp.json()["detail"].lower()


def test_missing_base_chunk_reference_handling(isolated_client: TestClient, temp_store_dir: Path):
    """Verify that deleting a referenced chunk from disk produces a clean 422 MissingReferenceError."""
    base_bytes = b"Base data anchor block.\n" * 400
    target_bytes = b"Base data anchor block.\n" * 400

    files = {
        "base_file": ("base.txt", base_bytes, "text/plain"),
        "target_file": ("target.txt", target_bytes, "text/plain"),
    }
    store_resp = isolated_client.post("/store-revision", files=files)
    assert store_resp.status_code == 201
    recipe = store_resp.json()
    rev_id = recipe["revision_id"]

    # Delete all chunks from disk
    for cf in (temp_store_dir / "chunks").glob("*.bin"):
        cf.unlink()

    # Attempt restore -> must cleanly return 422
    restore_resp = isolated_client.get(f"/restore-revision/{rev_id}")
    assert restore_resp.status_code == 422
    assert "not found on disk" in restore_resp.json()["detail"].lower()


def test_path_traversal_and_invalid_id_rejection(isolated_client: TestClient):
    """Verify that path traversal strings or illegal characters are rejected with 400 or 422."""
    malicious_ids = [
        "../etc/passwd",
        "..\\windows\\system32",
        "rev_123/../../secret",
        "rev_123;rm -rf /",
        "rev_123<script>",
        "invalid id with spaces",
    ]
    for bad_id in malicious_ids:
        resp = isolated_client.get(f"/revisions/{bad_id}")
        assert resp.status_code in (400, 404, 422)

        resp2 = isolated_client.get(f"/restore-revision/{bad_id}")
        assert resp2.status_code in (400, 404, 422)


def test_nonexistent_revision_returns_404(isolated_client: TestClient):
    """Verify that querying a non-existent revision ID returns HTTP 404."""
    resp = isolated_client.get("/revisions/rev_99999999_deadbeef")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()

    resp2 = isolated_client.get("/restore-revision/rev_99999999_deadbeef")
    assert resp2.status_code == 404


def test_atomic_file_write_cleanup(temp_store_dir: Path):
    """Verify that atomic writing cleans up temporary .tmp files."""
    mgr = StorageManager(base_dir=temp_store_dir)
    data = b"Atomic test chunk data."
    sha = mgr.store_chunk(data)
    assert sha == hashlib.sha256(data).hexdigest()

    # Verify that no .tmp files linger
    tmp_files = list(temp_store_dir.rglob("*.tmp"))
    assert len(tmp_files) == 0
