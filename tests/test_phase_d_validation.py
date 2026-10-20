"""Phase D Comprehensive End-to-End Pipeline Validation Suite.

Validates:
- D1 & D2: Real file handling across identical files, minor edits, insertions (start/middle),
  deletions, shifted chunk boundaries, empty files, small files (< 4KB), and arbitrary binary files.
- D3: Deterministic Tier 1 cache-hit and Tier 2 cache-miss/BallTree fallback paths.
- D4: Multi-Reference Delta (MRD) group and single-target restoration byte-exactness,
  window size variations, and corrupted container handling.
- D5: Metric accounting audit for storage savings, compression ratio, and overhead.
"""

from __future__ import annotations

import hashlib
import os
import random
import struct
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.service import DCLCBackendService
from src.candidate_indexer import HierarchicalCacheANNIndexer
from src.delta_encoder import (
    DeltaCorruptionError,
    decode_delta,
    decode_mrd_group,
    decode_mrd_single,
    encode_mrd,
    evaluate_and_encode_delta,
    evaluate_and_encode_mrd,
)
from src.fastcdc_chunker import fastcdc_chunks


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def backend_service():
    return DCLCBackendService()


# =============================================================================
# D1 & D2: Real File Handling & Boundary Shift Tests
# =============================================================================

def test_identical_files_full_dedup(client: TestClient):
    """Identical files should achieve 100% exact duplicate recognition."""
    content = b"Exact identical repeat line for DCLC validation.\n" * 500
    files = {
        "base_file": ("v1.txt", content, "text/plain"),
        "target_file": ("v2.txt", content, "text/plain"),
    }
    resp = client.post("/compare-revisions", files=files, data={"candidate_strategy": "hierarchical_cache"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["target_chunk_count"] > 0
    assert data["exact_duplicates_count"] == data["target_chunk_count"]
    assert data["deltas_accepted_count"] == 0
    assert data["target_dclc_stored_bytes"] == 32 * data["target_chunk_count"]
    assert data["reconstruction_success_rate"] == 100.0


def test_empty_files_handling(client: TestClient):
    """Empty target file and empty base file edge cases."""
    # Empty target
    files_empty_target = {
        "base_file": ("base.txt", b"Some base content.\n" * 100, "text/plain"),
        "target_file": ("target_empty.txt", b"", "application/octet-stream"),
    }
    resp = client.post("/compare-revisions", files=files_empty_target)
    assert resp.status_code == 200
    assert resp.json()["target_chunk_count"] == 0
    assert resp.json()["target_raw_bytes"] == 0

    # Empty base
    files_empty_base = {
        "base_file": ("base_empty.txt", b"", "application/octet-stream"),
        "target_file": ("target.txt", b"Target content for empty base test.\n" * 100, "text/plain"),
    }
    resp = client.post("/compare-revisions", files=files_empty_base)
    assert resp.status_code == 200
    data = resp.json()
    assert data["base_chunk_count"] == 0
    assert data["target_chunk_count"] >= 1
    # Without base chunks, all target chunks are stored standalone
    assert data["exact_duplicates_count"] == 0
    assert data["deltas_accepted_count"] == 0
    assert data["reconstruction_success_rate"] == 100.0


def test_small_subchunk_files(client: TestClient):
    """Files smaller than FastCDC min chunk size (e.g. 500 bytes) should form 1 chunk."""
    base = b"Mini file content version A with small text data.\n" * 5
    target = b"Mini file content version B with small modified text data.\n" * 5
    files = {
        "base_file": ("small_v1.txt", base, "text/plain"),
        "target_file": ("small_v2.txt", target, "text/plain"),
    }
    resp = client.post(
        "/compare-revisions",
        files=files,
        data={"min_chunk_size": 2048, "avg_chunk_size": 4096, "max_chunk_size": 8192},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_chunk_count"] == 1
    assert data["reconstruction_success_rate"] == 100.0


def test_content_insertion_at_start(client: TestClient):
    """FastCDC content-defined chunking should absorb insertion at start without shifting all boundaries."""
    base = b"Shared paragraph record line: Alpha Bravo Charlie Delta Echo.\n" * 1500  # ~94 KB
    # Prepend 10 KB of new header
    insertion = b"BRAND NEW INSERTED HEADER SECTION\n" * 300
    target = insertion + base

    files = {
        "base_file": ("base.txt", base, "text/plain"),
        "target_file": ("target.txt", target, "text/plain"),
    }
    resp = client.post(
        "/compare-revisions",
        files=files,
        data={
            "candidate_strategy": "hierarchical_cache",
            "delta_mode": "single",
            "min_chunk_size": 2048,
            "avg_chunk_size": 4096,
            "max_chunk_size": 8192,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_chunk_count"] >= 10
    assert data["exact_duplicates_count"] >= 5
    assert data["reconstruction_success_rate"] == 100.0


def test_content_insertion_in_middle(client: TestClient):
    """FastCDC should isolate insertions in the middle of a document."""
    part1 = b"Section 1: Initial document configuration and setup.\n" * 1000  # ~53 KB
    part2 = b"Section 2: Detailed documentation and performance reports.\n" * 1000  # ~59 KB
    base = part1 + part2

    middle_insert = b"MIDDLE INSERTION: New patch notes and change logs.\n" * 200  # ~10 KB
    target = part1 + middle_insert + part2

    files = {
        "base_file": ("base.txt", base, "text/plain"),
        "target_file": ("target.txt", target, "text/plain"),
    }
    resp = client.post(
        "/compare-revisions",
        files=files,
        data={
            "candidate_strategy": "hierarchical_cache",
            "delta_mode": "single",
            "min_chunk_size": 2048,
            "avg_chunk_size": 4096,
            "max_chunk_size": 8192,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_chunk_count"] >= 10
    assert data["exact_duplicates_count"] >= 5
    assert data["reconstruction_success_rate"] == 100.0


def test_content_deletion(client: TestClient):
    """FastCDC should handle deletion of whole sections while retaining shared chunks."""
    part1 = b"Retained Section 1 content line data block.\n" * 1000  # ~44 KB
    part2_deleted = b"To be deleted Section 2 content data block.\n" * 800  # ~35 KB
    part3 = b"Retained Section 3 content line data block.\n" * 1000  # ~44 KB

    base = part1 + part2_deleted + part3
    target = part1 + part3

    files = {
        "base_file": ("base.txt", base, "text/plain"),
        "target_file": ("target.txt", target, "text/plain"),
    }
    resp = client.post(
        "/compare-revisions",
        files=files,
        data={
            "candidate_strategy": "hierarchical_cache",
            "delta_mode": "single",
            "min_chunk_size": 2048,
            "avg_chunk_size": 4096,
            "max_chunk_size": 8192,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_chunk_count"] >= 8
    assert data["exact_duplicates_count"] >= 4
    assert data["reconstruction_success_rate"] == 100.0


def test_arbitrary_binary_file_handling(backend_service: DCLCBackendService):
    """Arbitrary raw binary data with high entropy should not cause decode crashes."""
    rng = random.Random(1337)
    base_bytes = bytes(rng.getrandbits(8) for _ in range(16384))
    target_bytes = bytearray(base_bytes)
    # Mutate 100 bytes
    for _ in range(100):
        pos = rng.randint(0, len(target_bytes) - 1)
        target_bytes[pos] = rng.randint(0, 255)
    target_bytes = bytes(target_bytes)

    res = backend_service.compare_revisions(
        base_bytes=base_bytes,
        target_bytes=target_bytes,
        candidate_strategy="hierarchical_cache",
        delta_mode="single",
        resemblance_threshold=0.50,
    )
    assert res.target_chunk_count >= 1
    assert res.reconstruction_success_rate == 100.0


# =============================================================================
# D3: Cache-Hit and Cache-Miss / BallTree Verification
# =============================================================================

def test_hierarchical_cache_hit_and_miss_deterministic():
    """Verify Tier 1 FIFO cache hits and Tier 2 cache misses + BallTree queries."""
    indexer = HierarchicalCacheANNIndexer(
        cache_capacity=4,           # Small cache to test eviction
        similarity_threshold=0.70,
        top_k=3,
        min_ann_corpus_size=5,     # Triggers BallTree at 5 chunks
    )

    rng = np.random.RandomState(42)
    # Generate 8 random orthogonal unit vectors
    raw_vecs = rng.randn(8, 128).astype(np.float32)
    vecs = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)

    # Ingest 8 vectors (capacity=4, so items 0, 1, 2, 3 will be evicted from FIFO cache)
    for i in range(8):
        indexer.add_chunk(chunk_id=i, initial_feature=vecs[i], embedding=vecs[i], sha256=f"sha_{i}")

    assert indexer.stats.cache_hits == 0
    assert indexer.stats.cache_misses == 0

    # Query 1: Query with item 7 (currently in Tier 1 cache) -> Expect Cache Hit
    cands_hit = indexer.query_candidates(query_id=999, initial_feature=vecs[7], embedding=vecs[7], sha256="q7")
    assert 7 in cands_hit
    assert indexer.stats.cache_hits == 1
    assert indexer.stats.cache_misses == 0
    assert indexer.stats.ann_queries == 0

    # Query 2: Query with item 0 (evicted from cache, present only in Tier 2 global corpus)
    # -> Must miss Tier 1 cache, trigger Tier 2 BallTree query, and find candidate 0
    cands_miss = indexer.query_candidates(query_id=999, initial_feature=vecs[0], embedding=vecs[0], sha256="q0")
    assert 0 in cands_miss
    assert indexer.stats.cache_hits == 1
    assert indexer.stats.cache_misses == 1
    assert indexer.stats.ann_queries == 1

    # Query 3: Query with orthogonal vector that matches nothing >= 0.70
    ortho_vec = -vecs[0]  # Opposite direction
    cands_none = indexer.query_candidates(query_id=999, initial_feature=ortho_vec, embedding=ortho_vec, sha256="none")
    assert cands_none == []
    assert indexer.stats.cache_hits == 1
    assert indexer.stats.cache_misses == 2
    assert indexer.stats.ann_queries == 2


def test_hierarchical_indexer_edge_cases():
    """Verify empty indexer, invalid vector validation, and reset behavior."""
    indexer = HierarchicalCacheANNIndexer(cache_capacity=10, similarity_threshold=0.70)

    # Query empty indexer
    cands = indexer.query_candidates(query_id=0, initial_feature=np.zeros(128), embedding=np.ones(128, dtype=np.float32), sha256="empty")
    assert cands == []
    assert indexer.stats.cache_misses == 1

    # Query with NaN embedding
    nan_emb = np.full(128, np.nan, dtype=np.float32)
    cands_nan = indexer.query_candidates(query_id=1, initial_feature=np.zeros(128), embedding=nan_emb, sha256="nan")
    assert cands_nan == []

    # Reset
    indexer.reset()
    assert indexer.stats.cache_hits == 0
    assert indexer.stats.cache_misses == 0
    assert indexer.stats.ann_queries == 0


# =============================================================================
# D4: Multi-Reference Delta (MRD) vs Single-Reference In-Depth Verification
# =============================================================================

def test_mrd_window_and_max_refs_parameter_validation(client: TestClient):
    """Verify API parameter boundary validation on MRD parameters."""
    files = {"base_file": ("b.txt", b"abc" * 100, "text/plain"), "target_file": ("t.txt", b"xyz" * 100, "text/plain")}

    # Window size < 1 -> 422 Unprocessable Entity (FastAPI validation)
    resp = client.post("/compare-revisions", files=files, data={"delta_mode": "multi", "mrd_window_size": 0})
    assert resp.status_code == 422

    # Max refs > 8 -> 422 Unprocessable Entity
    resp = client.post("/compare-revisions", files=files, data={"delta_mode": "multi", "mrd_max_refs": 10})
    assert resp.status_code == 422


def test_mrd_group_and_single_restoration_bit_exactness():
    """Verify decode_mrd_group and decode_mrd_single reproduce byte-for-byte identical content."""
    rng = random.Random(2026)
    # Generate 3 reference chunks
    ref_chunks = [bytes(rng.getrandbits(8) for _ in range(4096)) for _ in range(3)]
    # Generate 4 target chunks derived with 3% mutation from reference chunks
    target_chunks = []
    for i in range(4):
        base_c = bytearray(ref_chunks[i % 3])
        for _ in range(120):
            base_c[rng.randint(0, len(base_c) - 1)] = rng.randint(0, 255)
        target_chunks.append(bytes(base_c))

    # Encode MRD
    mrd_container = encode_mrd(ref_chunks, target_chunks, level=6)
    assert mrd_container.startswith(b"DCLC_M01")

    # 1. MG Group Restoration
    restored_group = decode_mrd_group(ref_chunks, mrd_container)
    assert len(restored_group) == len(target_chunks)
    for idx, (orig, rest) in enumerate(zip(target_chunks, restored_group)):
        assert orig == rest, f"Target chunk {idx} failed MG group restoration bit-exactness!"

    # 2. MS Single-Target Restoration
    for idx, orig in enumerate(target_chunks):
        rest_single = decode_mrd_single(ref_chunks, mrd_container, idx)
        assert orig == rest_single, f"Target chunk {idx} failed MS single restoration bit-exactness!"


def test_mrd_corrupted_container_safety():
    """Corrupted MRD container bytes must trigger DeltaCorruptionError without returning bad data."""
    ref_chunks = [b"Reference chunk data block " * 100]
    target_chunks = [b"Target chunk data block mutated " * 100]

    container = bytearray(encode_mrd(ref_chunks, target_chunks, level=6))

    # Corrupt payload near the end
    container[-10] ^= 0xFF

    with pytest.raises(DeltaCorruptionError):
        decode_mrd_group(ref_chunks, bytes(container))

    with pytest.raises(DeltaCorruptionError):
        decode_mrd_single(ref_chunks, bytes(container), 0)


# =============================================================================
# D5: Storage Metric Accounting & Overhead Auditing
# =============================================================================

def test_storage_metric_formulas_audit(backend_service: DCLCBackendService):
    """Audit storage metric calculations: raw size, stored size, savings %, and compression ratio."""
    content_base = b"Base shared content stream block.\n" * 400
    content_target = (b"Target MODIFIED content stream block.\n" * 100) + (b"Base shared content stream block.\n" * 300)

    res = backend_service.compare_revisions(
        base_bytes=content_base,
        target_bytes=content_target,
        candidate_strategy="hierarchical_cache",
        delta_mode="single",
    )

    # 1. Target raw size must equal input target bytes exactly
    assert res.target_raw_bytes == len(content_target)

    # 2. Sum of stored components must equal target_dclc_stored_bytes
    expected_stored = 0
    for c in res.detailed_chunk_comparisons:
        if c.is_exact_duplicate:
            expected_stored += 32  # 32-byte pointer overhead
        elif c.delta_accepted:
            assert c.delta_size_bytes is not None
            expected_stored += c.delta_size_bytes
        else:
            expected_stored += c.standalone_compressed_size_bytes

    assert res.target_dclc_stored_bytes == expected_stored

    # 3. Savings percentage formula: (raw - stored) / raw * 100
    calculated_savings = (res.target_raw_bytes - res.target_dclc_stored_bytes) / res.target_raw_bytes * 100.0
    assert abs(res.target_storage_savings_pct - calculated_savings) < 1e-4

    # 4. Compression ratio formula: raw / stored
    calculated_ratio = res.target_raw_bytes / res.target_dclc_stored_bytes
    assert abs(res.target_compression_ratio - calculated_ratio) < 1e-4
