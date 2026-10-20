"""Tests for Phase B: Multi-Reference Delta Encoding (MRD).

Covers Section 3.4 and Section 4.4 of Xiao et al. (2026):
  - MRD encode/decode round trip (group-level restore: MG mode).
  - Bit-for-bit identical reconstruction across multiple targets.
  - Cross-boundary redundancy: CDC boundary shift where content is split
    across adjacent reference blocks, proving MRD succeeds where SR fails.
  - Multi-reference joint compression (e.g. 3 base chunks + 3 target chunks).
  - Single-target restoration from an MRD group (MS mode).
  - Corruption and safety:
      * Truncated container -> DeltaCorruptionError
      * Corrupted magic header -> DeltaCorruptionError
      * Payload corruption -> DeltaCorruptionError
      * Mismatched reference chunks (checksum failure) -> BaseMismatchError
      * Reference count mismatch -> BaseMismatchError
      * Out-of-bounds target index in single restore -> IndexError
      * Non-bytes chunk input -> TypeError
      * Empty input lists -> ValueError
  - Cases where MRD is worse than SR (economic rejection on unrelated data).
  - Throughput and latency trade-offs: MS restore overhead vs MG restore.
  - Regression compatibility: SR baseline remains unaffected.
"""

from __future__ import annotations

import hashlib
import time
import numpy as np
import pytest

from src.delta_encoder import (
    BaseMismatchError,
    DeltaCorruptionError,
    DeltaResult,
    MRDError,
    MRDReferenceMeta,
    MRDResult,
    MRDTargetMeta,
    compress_standalone,
    decode_delta,
    decode_mrd_group,
    decode_mrd_single,
    encode_delta,
    encode_mrd,
    evaluate_and_encode_delta,
    evaluate_and_encode_mrd,
    unpack_mrd_container,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_text_chunk(pattern: str, repeat: int = 50) -> bytes:
    """Generate deterministic byte chunk with repeating patterns."""
    return (pattern * repeat).encode("utf-8")


# ---------------------------------------------------------------------------
# 1. Basic MRD Round-Trip & Bit-Exact Reconstruction (MG Mode)
# ---------------------------------------------------------------------------

def test_mrd_encode_decode_round_trip_group():
    """Verify that encode_mrd followed by decode_mrd_group reconstructs identical bytes."""
    ref1 = b"Shared configuration header: version=1.2.0; cluster=east-us; timeout=500ms;\n" * 20
    ref2 = b"Database schema definition: table=users; columns=[id, name, email, role];\n" * 20
    refs = [ref1, ref2]

    # Target chunks have 85% overlap with refs
    target1 = b"Shared configuration header: version=1.3.0; cluster=east-us; timeout=750ms;\n" * 20
    target2 = b"Database schema definition: table=users; columns=[id, name, email, role, active];\n" * 20
    targets = [target1, target2]

    container = encode_mrd(refs, targets, level=6)
    assert len(container) > 0

    reconstructed = decode_mrd_group(refs, container)
    assert len(reconstructed) == len(targets)
    assert reconstructed[0] == target1
    assert reconstructed[1] == target2
    for orig, rec in zip(targets, reconstructed):
        assert hashlib.sha256(rec).digest() == hashlib.sha256(orig).digest()


# ---------------------------------------------------------------------------
# 2. Cross-Boundary Redundancy Elimination (Paper Figure 6 Example)
# ---------------------------------------------------------------------------

def test_mrd_cross_boundary_redundancy():
    """Demonstrate that MRD eliminates cross-block redundancy where SR fails.

    Paper Fig. 6 (Page 11):
      Old version F1 has:
        bf11 = "aa" + "ccc"
        bf12 = "ddd" + "fff"
      New version F2 has:
        bf21 = "aaee"
        bf22 = "cccdddf"  <-- Straddles bf11 and bf12!
    """
    rng = np.random.default_rng(42)
    words_alpha = [f"alpha_{w:04d}" for w in rng.integers(0, 500, size=500)]
    shared_part1 = (" ".join(words_alpha) + "\n").encode("utf-8")
    words_beta = [f"beta_{w:04d}" for w in rng.integers(0, 500, size=500)]
    shared_part2 = (" ".join(words_beta) + "\n").encode("utf-8")

    # Base chunks in version 1: shared_part1 in bf11, shared_part2 in bf12
    base_bf11 = b"PREFIX_V1_ONLY_CHUNK_11_" * 30 + shared_part1
    base_bf12 = shared_part2 + b"SUFFIX_V1_ONLY_CHUNK_12_" * 30
    refs = [base_bf11, base_bf12]

    # Target chunk in version 2 straddles the boundary: shared_part1 + shared_part2
    target_straddle = shared_part1 + shared_part2
    targets = [target_straddle]

    # 1. Single-Reference (SR) Delta: target_straddle can only pick ONE base
    res_sr_base1 = evaluate_and_encode_delta(base_bf11, target_straddle, compression_level=6)
    res_sr_base2 = evaluate_and_encode_delta(base_bf12, target_straddle, compression_level=6)
    best_sr_size = min(
        res_sr_base1.delta_total_size if res_sr_base1.is_accepted else 999999,
        res_sr_base2.delta_total_size if res_sr_base2.is_accepted else 999999,
    )

    # 2. Multi-Reference Delta (MRD): joins base_bf11 and base_bf12 into single reference
    mrd_res = evaluate_and_encode_mrd(refs, targets, compression_level=6)
    assert mrd_res.is_accepted is True
    assert mrd_res.reconstruction_verified is True

    # Standalone compressed size of target
    standalone_target = len(compress_standalone(target_straddle))

    # MRD achieves strong compression because target_straddle finds BOTH parts in dictionary
    assert mrd_res.mrd_total_size < standalone_target
    assert mrd_res.savings_bytes > 0

    # Key paper proof: MRD delta total size is significantly smaller than the best SR delta size!
    assert mrd_res.mrd_total_size < best_sr_size, (
        f"MRD size ({mrd_res.mrd_total_size}) should be smaller than best SR size ({best_sr_size})"
    )


# ---------------------------------------------------------------------------
# 3. Single-Target Restoration (MS Mode: Section 4.4)
# ---------------------------------------------------------------------------

def test_mrd_single_target_restore_ms_mode():
    """Verify that decode_mrd_single reconstructs individual chunks correctly."""
    ref_a = _generate_text_chunk("DATABASE_CONNECTION_POOL_PARAMS_HOST_PORT_", repeat=30)
    ref_b = _generate_text_chunk("AUTHENTICATION_TOKEN_SECRET_ROTATION_KEY_", repeat=30)
    refs = [ref_a, ref_b]

    target_0 = _generate_text_chunk("DATABASE_CONNECTION_POOL_PARAMS_HOST_PORT_v2_", repeat=30)
    target_1 = _generate_text_chunk("AUTHENTICATION_TOKEN_SECRET_ROTATION_KEY_v2_", repeat=30)
    target_2 = _generate_text_chunk("UNRELATED_APP_CACHE_LOGGING_METRIC_BUFFER_", repeat=30)
    targets = [target_0, target_1, target_2]

    container = encode_mrd(refs, targets, level=6)

    # Decode target 0 individually
    rec_0 = decode_mrd_single(refs, container, target_index=0)
    assert rec_0 == target_0

    # Decode target 1 individually
    rec_1 = decode_mrd_single(refs, container, target_index=1)
    assert rec_1 == target_1

    # Decode target 2 individually
    rec_2 = decode_mrd_single(refs, container, target_index=2)
    assert rec_2 == target_2

    # Out of range indices raise IndexError
    with pytest.raises(IndexError):
        decode_mrd_single(refs, container, target_index=3)
    with pytest.raises(IndexError):
        decode_mrd_single(refs, container, target_index=-1)


# ---------------------------------------------------------------------------
# 4. Multi-Reference Multiple Chunks (3 Refs + 3 Targets)
# ---------------------------------------------------------------------------

def test_mrd_three_refs_three_targets():
    """Verify joint compression with 3 reference chunks and 3 target chunks."""
    rng = np.random.default_rng(42)
    refs = []
    targets = []
    for i in range(3):
        words = [f"block_{i}_word_{w:04d}" for w in rng.integers(0, 200, size=500)]
        base = (" ".join(words) + "\n").encode("utf-8")
        refs.append(base)

        mod_words = list(words)
        for idx in rng.choice(len(words), size=30, replace=False):
            mod_words[idx] = f"modified_{i}_{idx}"
        target = (" ".join(mod_words) + "\n").encode("utf-8")
        targets.append(target)

    res = evaluate_and_encode_mrd(refs, targets, compression_level=6)
    assert res.is_accepted is True
    assert res.target_count == 3
    assert res.reference_count == 3
    assert res.reconstruction_verified is True
    assert len(res.target_metas) == 3
    assert len(res.reference_metas) == 3

    # Group restore check
    reconstructed = decode_mrd_group(refs, res.mrd_container)
    assert reconstructed == targets


# ---------------------------------------------------------------------------
# 5. Cases Where MRD is Worse Than SR / Standalone
# ---------------------------------------------------------------------------

def test_mrd_worse_than_standalone_unrelated_data_rejected():
    """Verify that MRD is rejected when data is unrelated and metadata adds overhead."""
    rng = np.random.default_rng(999)
    # Random bytes have 0 compressibility and 0 similarity
    ref1 = bytes(rng.integers(0, 256, size=2048, dtype=np.uint8))
    ref2 = bytes(rng.integers(0, 256, size=2048, dtype=np.uint8))
    target1 = bytes(rng.integers(0, 256, size=2048, dtype=np.uint8))
    target2 = bytes(rng.integers(0, 256, size=2048, dtype=np.uint8))

    res = evaluate_and_encode_mrd([ref1, ref2], [target1, target2], compression_level=6)

    # Standalone compressed size of random data + zlib overhead
    # MRD overhead includes 272 bytes header + combined dictionary overhead
    assert res.reconstruction_verified is True
    assert res.is_accepted is False
    assert res.rejection_reason is not None
    assert ">= standalone compressed size" in res.rejection_reason


def test_mrd_restore_throughput_overhead_ms_vs_mg():
    """Confirm the paper finding: MS mode incurs overhead compared to MG mode.

    Paper Section 4.4 (Page 18):
      'MG consistently achieves the highest restore throughput... For MS,
       the restore throughput shows an average reduction of 34.51% compared with SR.'
    """
    # Create an MRD group with 5 target chunks
    ref = b"REPEATING_TEMPLATE_CONTENT_BLOCK_SYSTEM_LOG_" * 80
    targets = [
        f"REPEATING_TEMPLATE_CONTENT_BLOCK_SYSTEM_LOG_ENTRY_{i}_".encode("utf-8") * 80
        for i in range(5)
    ]

    container = encode_mrd([ref], targets, level=6)

    # Benchmark MG mode: 50 iterations
    t0 = time.perf_counter()
    for _ in range(50):
        _ = decode_mrd_group([ref], container)
    mg_time = time.perf_counter() - t0

    # Benchmark MS mode: restore all 5 targets individually 50 times
    t1 = time.perf_counter()
    for _ in range(50):
        for i in range(5):
            _ = decode_mrd_single([ref], container, target_index=i)
    ms_time = time.perf_counter() - t1

    # Single-target restore decodes the group 5 times (once per target)
    # so ms_time must be strictly greater than mg_time
    assert ms_time > mg_time, "MS mode should take longer than MG mode for all targets"


# ---------------------------------------------------------------------------
# 6. Corruption, Tampering, and Safety Errors
# ---------------------------------------------------------------------------

def test_mrd_truncated_container():
    """Truncated container raises DeltaCorruptionError."""
    ref = b"BASE_REF_CHUNK"
    target = b"TARGET_CHUNK"
    container = encode_mrd([ref], [target])

    # Truncate at magic header
    with pytest.raises(DeltaCorruptionError, match="truncated"):
        decode_mrd_group([ref], container[:5])

    # Truncate halfway through header
    with pytest.raises(DeltaCorruptionError):
        decode_mrd_group([ref], container[:45])


def test_mrd_corrupt_magic_header():
    """Container with corrupted magic raises DeltaCorruptionError."""
    ref = b"BASE_REF_CHUNK"
    target = b"TARGET_CHUNK"
    container = encode_mrd([ref], [target])
    corrupt = b"BAD_M01!" + container[8:]

    with pytest.raises(DeltaCorruptionError, match="Invalid MRD magic"):
        decode_mrd_group([ref], corrupt)


def test_mrd_corrupted_payload():
    """Corrupted payload bytes trigger DeltaCorruptionError."""
    ref = b"BASE_REF_CHUNK" * 20
    target = b"TARGET_CHUNK" * 20
    container = encode_mrd([ref], [target])

    # Corrupt the payload at the tail
    corrupted_payload = container[:-15] + b"\xff" * 15
    with pytest.raises(DeltaCorruptionError):
        decode_mrd_group([ref], corrupted_payload)


def test_mrd_wrong_reference_bytes_mismatch():
    """Passing modified/wrong reference bytes raises BaseMismatchError."""
    ref1 = b"CORRECT_REFERENCE_CHUNK_ONE" * 20
    ref2 = b"CORRECT_REFERENCE_CHUNK_TWO" * 20
    target = b"TARGET_MODIFIED_CHUNK" * 20
    container = encode_mrd([ref1, ref2], [target])

    # Wrong reference chunk passed
    wrong_ref1 = b"TAMPERED_REFERENCE_CHUNK_ONE" * 20
    with pytest.raises(BaseMismatchError):
        decode_mrd_group([wrong_ref1, ref2], container)

    # Reference count mismatch
    with pytest.raises(BaseMismatchError, match="Reference count mismatch"):
        decode_mrd_group([ref1], container)


def test_mrd_empty_input_validation():
    """Empty target or reference list raises ValueError."""
    with pytest.raises(ValueError, match="reference_chunks must contain at least one"):
        encode_mrd([], [b"target"])

    with pytest.raises(ValueError, match="target_chunks must contain at least one"):
        encode_mrd([b"ref"], [])


def test_mrd_invalid_chunk_type():
    """Non-bytes element raises TypeError."""
    with pytest.raises(TypeError, match="must be bytes"):
        encode_mrd(["string_instead_of_bytes"], [b"target"])  # type: ignore

    with pytest.raises(TypeError, match="must be bytes"):
        encode_mrd([b"ref"], ["string_instead_of_bytes"])  # type: ignore


# ---------------------------------------------------------------------------
# 7. Comparison: SR vs MRD Identical Inputs
# ---------------------------------------------------------------------------

def test_sr_vs_mrd_comparison_identical_inputs():
    """Compare SR and MRD on the same multi-chunk revision scenario."""
    rng = np.random.default_rng(123)
    words_0 = [f"driver_int_{w:04d}" for w in rng.integers(0, 300, size=600)]
    ref_0 = (" ".join(words_0) + "\n").encode("utf-8")
    words_1 = [f"mem_page_{w:04d}" for w in rng.integers(0, 300, size=600)]
    ref_1 = (" ".join(words_1) + "\n").encode("utf-8")
    refs = [ref_0, ref_1]

    # 2 revised target chunks with small revisions (10% token mutations)
    mod_0 = list(words_0)
    for idx in rng.choice(len(words_0), size=30, replace=False):
        mod_0[idx] = f"driver_int_v2_{idx}"
    target_0 = (" ".join(mod_0) + "\n").encode("utf-8")

    mod_1 = list(words_1)
    for idx in rng.choice(len(words_1), size=30, replace=False):
        mod_1[idx] = f"mem_page_v2_{idx}"
    target_1 = (" ".join(mod_1) + "\n").encode("utf-8")
    targets = [target_0, target_1]

    raw_total = sum(len(t) for t in targets)

    # --- SR Baseline ---
    t_sr0 = time.perf_counter()
    sr_0 = evaluate_and_encode_delta(ref_0, target_0, compression_level=6)
    sr_1 = evaluate_and_encode_delta(ref_1, target_1, compression_level=6)
    sr_time = time.perf_counter() - t_sr0

    assert sr_0.is_accepted and sr_1.is_accepted
    sr_total_bytes = sr_0.net_stored_size + sr_1.net_stored_size
    sr_cr = float(raw_total / sr_total_bytes)

    # Benchmark SR restore
    t_srr0 = time.perf_counter()
    for _ in range(50):
        _ = decode_delta(ref_0, sr_0.delta_container)
        _ = decode_delta(ref_1, sr_1.delta_container)
    sr_restore_time = time.perf_counter() - t_srr0

    # --- MRD Mode ---
    t_mrd0 = time.perf_counter()
    mrd_res = evaluate_and_encode_mrd(refs, targets, compression_level=6)
    mrd_time = time.perf_counter() - t_mrd0

    assert mrd_res.is_accepted is True
    mrd_total_bytes = mrd_res.net_stored_size
    mrd_cr = mrd_res.savings_ratio

    # Benchmark MG restore
    t_mgr0 = time.perf_counter()
    for _ in range(50):
        _ = decode_mrd_group(refs, mrd_res.mrd_container)
    mg_restore_time = time.perf_counter() - t_mgr0

    # Benchmark MS restore
    t_msr0 = time.perf_counter()
    for _ in range(50):
        _ = decode_mrd_single(refs, mrd_res.mrd_container, 0)
        _ = decode_mrd_single(refs, mrd_res.mrd_container, 1)
    ms_restore_time = time.perf_counter() - t_msr0

    # Both must produce valid compression ratios > 1.0
    assert sr_cr > 1.0
    assert mrd_cr > 1.0

    # Both restore bit-exact data
    assert decode_mrd_group(refs, mrd_res.mrd_container) == targets


# ---------------------------------------------------------------------------
# 8. Regression Compatibility: Existing SR Delta Functions Untouched
# ---------------------------------------------------------------------------

def test_mrd_regression_sr_functions_unaffected():
    """Verify that existing Single-Reference functions continue to work normally."""
    base = b"HELLO_WORLD_BASE_CHUNK_DATA_" * 20
    target = b"HELLO_WORLD_BASE_CHUNK_DATA_MODIFIED_" * 20

    res = evaluate_and_encode_delta(base, target, compression_level=6)
    assert isinstance(res, DeltaResult)
    assert res.reconstruction_verified is True

    if res.is_accepted:
        rec = decode_delta(base, res.delta_container)
        assert rec == target
