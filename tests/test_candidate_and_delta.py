"""Unit and Integration Tests for Phase 4C Candidate Indexing and Delta Encoding."""

import hashlib
import os
import zlib
import numpy as np
import pytest

from src.candidate_indexer import (
    EmbeddingCosineKNNIndexer,
    RandomProjectionLSHIndexer,
    StreamLocalityIndexer,
)
from src.delta_encoder import (
    BaseMismatchError,
    DeltaCorruptionError,
    compress_standalone,
    decode_delta,
    encode_delta,
    evaluate_and_encode_delta,
)


def test_delta_exact_reconstruction_random_binary():
    """Verify bit-exact reconstruction on pseudo-random binary chunks with high mutual overlap."""
    rng = np.random.default_rng(1234)
    # Generate base binary chunk (8 KB)
    base_bytes = bytes(rng.integers(0, 256, size=8192, dtype=np.uint8))
    # Modify 5% of bytes to simulate 95% resemblance
    target_list = list(base_bytes)
    for idx in rng.choice(8192, size=400, replace=False):
        target_list[idx] = int(rng.integers(0, 256))
    target_bytes = bytes(target_list)

    # Encode delta
    delta_container = encode_delta(base_bytes, target_bytes, level=6)
    assert len(delta_container) > 0

    # Decode and verify exact reconstruction
    reconstructed = decode_delta(base_bytes, delta_container)
    assert reconstructed == target_bytes
    assert hashlib.sha256(reconstructed).digest() == hashlib.sha256(target_bytes).digest()


def test_delta_empty_and_identical_chunks():
    """Verify safe handling of empty and identical chunks."""
    # 1. Identical chunks
    data = b"DCLC identical test chunk " * 100
    res_ident = evaluate_and_encode_delta(data, data)
    assert res_ident.is_accepted is True
    assert res_ident.reconstruction_verified is True
    assert res_ident.net_stored_size == 0

    # 2. Empty target
    base = b"Non-empty base chunk"
    target = b""
    delta_empty = encode_delta(base, target)
    rec_empty = decode_delta(base, delta_empty)
    assert rec_empty == b""


def test_delta_corruption_and_mismatch_exceptions():
    """Verify that corrupt headers, corrupted payload, and wrong base raise appropriate exceptions."""
    base_a = b"Reference Base Chunk A " * 50
    base_b = b"Unrelated Base Chunk B " * 50
    target = b"Reference Base Chunk A modified target" * 50

    delta = encode_delta(base_a, target)

    # 1. Wrong base provided -> BaseMismatchError
    with pytest.raises(BaseMismatchError):
        decode_delta(base_b, delta)

    # 2. Truncated container (< 76 bytes) -> DeltaCorruptionError
    with pytest.raises(DeltaCorruptionError):
        decode_delta(base_a, delta[:40])

    # 3. Invalid magic identifier -> DeltaCorruptionError
    corrupt_magic = b"BADMAGIC" + delta[8:]
    with pytest.raises(DeltaCorruptionError):
        decode_delta(base_a, corrupt_magic)

    # 4. Corrupted payload -> DeltaCorruptionError
    corrupt_payload = delta[:80] + b"\xff" * 20 + delta[100:]
    with pytest.raises(DeltaCorruptionError):
        decode_delta(base_a, corrupt_payload)


def test_unhelpful_delta_rejection_logic():
    """Verify that an unhelpful delta (unrelated chunks) is rejected."""
    # Two completely random independent chunks (0% overlap)
    rng = np.random.default_rng(999)
    base = bytes(rng.integers(0, 256, size=4096, dtype=np.uint8))
    target = bytes(rng.integers(0, 256, size=4096, dtype=np.uint8))

    res = evaluate_and_encode_delta(base, target, compression_level=6)

    # With completely random data, delta overhead makes total delta size >= standalone compressed size
    assert res.reconstruction_verified is True
    assert res.is_accepted is False
    assert res.rejection_reason is not None
    assert res.net_stored_size == res.standalone_compressed_size


def test_candidate_indexer_random_projection_lsh():
    """Verify that RandomProjectionLSHIndexer correctly buckets vectors without oracle labels."""
    indexer = RandomProjectionLSHIndexer(dim=128, num_tables=4, num_bits=4, seed=42)

    # Two nearly identical vectors (cosine ~ 0.99)
    v1 = np.ones(128, dtype=np.float32)
    v2 = np.ones(128, dtype=np.float32)
    v2[0] = 0.95
    # An orthogonal vector
    v3 = np.zeros(128, dtype=np.float32)
    v3[50:] = -1.0

    dummy_emb = np.zeros(128, dtype=np.float32)

    indexer.add_chunk(chunk_id=101, initial_feature=v1, embedding=dummy_emb, sha256="fp1")
    indexer.add_chunk(chunk_id=102, initial_feature=v3, embedding=dummy_emb, sha256="fp3")

    candidates = indexer.query_candidates(query_id=999, initial_feature=v2, embedding=dummy_emb, sha256="fp2")

    # Near-identical v1 should be retrieved as candidate for v2
    assert 101 in candidates


def test_candidate_indexer_embedding_knn():
    """Verify that EmbeddingCosineKNNIndexer retrieves nearest embeddings by cosine dot product."""
    indexer = EmbeddingCosineKNNIndexer(top_k=2)

    dummy_feat = np.zeros(128, dtype=np.float32)

    e1 = np.zeros(128, dtype=np.float32)
    e1[0] = 1.0  # Unit vector along dim 0

    e2 = np.zeros(128, dtype=np.float32)
    e2[0] = 0.9
    e2[1] = 0.435  # Unit vector close to e1

    e3 = np.zeros(128, dtype=np.float32)
    e3[10] = 1.0  # Orthogonal vector

    indexer.add_chunk(chunk_id=1, initial_feature=dummy_feat, embedding=e1, sha256="fp1")
    indexer.add_chunk(chunk_id=2, initial_feature=dummy_feat, embedding=e3, sha256="fp3")

    candidates = indexer.query_candidates(query_id=999, initial_feature=dummy_feat, embedding=e2, sha256="fp2")

    assert len(candidates) <= 2
    assert candidates[0] == 1  # e1 is closest to e2


def test_stream_locality_indexer_window_eviction():
    """Verify that StreamLocalityIndexer evicts chunks beyond window size."""
    indexer = StreamLocalityIndexer(window_size=3)

    dummy_feat = np.zeros(128, dtype=np.float32)
    dummy_emb = np.zeros(128, dtype=np.float32)

    indexer.add_chunk(1, dummy_feat, dummy_emb, "fp1")
    indexer.add_chunk(2, dummy_feat, dummy_emb, "fp2")
    indexer.add_chunk(3, dummy_feat, dummy_emb, "fp3")
    indexer.add_chunk(4, dummy_feat, dummy_emb, "fp4")  # 1 should be evicted

    candidates = indexer.query_candidates(999, dummy_feat, dummy_emb, "fp_query")
    assert candidates == [2, 3, 4]
    assert 1 not in candidates
