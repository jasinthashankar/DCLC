"""Byte-Level Delta Encoding and Verification for DCLC (Phase 4C + Phase B).

Implements byte-level delta compression between reference chunks (bases)
and incoming candidate chunks (targets):

1. Single-Reference (SR) Delta Compression (Baseline):
   - Engine: Standard sliding-window preset-dictionary compression (RFC 1950 / Deflate).
     The reference chunk is loaded as a preset dictionary into zlib (LZ77 history window).
   - Header Format (76 bytes metadata overhead):
       [0:8]   Magic identifier: b"DCLC_D01" (8 bytes)
       [8:40]  Base chunk SHA-256 binary digest (32 bytes)
       [40:44] Target uncompressed byte length, uint32 big-endian (4 bytes)
       [44:76] Target SHA-256 binary digest (32 bytes)
       [76:]   Compressed delta payload

2. Multi-Reference Delta (MRD) Compression (Phase B — Section 3.4 & 4.4):
   - Solves boundary drift where CDC splits common content across adjacent chunks.
   - Conceptualizes the input-side neighborhood window N(B_i) and candidate base
     chunks phi_i as a cohesive encoding unit (Eq. 21).
   - Concatenates multiple candidate base blocks into a single logical reference unit
     (bf_11 || bf_12 || ...) and adjacent target chunks into an input fragment
     (bf_21 || bf_22 || ...).
   - Container format (Magic: b"DCLC_M01"):
       [0:8]   Magic identifier: b"DCLC_M01" (8 bytes)
       [8:10]  Number of reference chunks R (uint16)
       Followed by R * (32-byte sha256 + 4-byte length)
       Followed by 32-byte combined reference SHA-256 digest
       Followed by Number of target chunks T (uint16)
       Followed by T * (32-byte sha256 + 4-byte offset + 4-byte length)
       Followed by 4-byte combined target length + 32-byte combined target SHA-256
       Followed by compressed delta payload
   - Restoration Modes:
       * MRD Group-Level Restore (MG): Decompresses the whole target group at once.
       * MRD Single-Target Restore (MS): Restores a single target chunk by slicing
         the decompressed group with stored offset and length.

Integrity & Safety:
   * Every accepted delta is verified by full byte-for-byte reconstruction:
     reconstructed == target_bytes.
   * Deltas are REJECTED when total container size is not smaller than standalone compression.
   * Corrupted headers, checksum mismatches, and missing references raise dedicated exceptions.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from typing import Optional

MAGIC_HEADER = b"DCLC_D01"  # 8 bytes
HEADER_SIZE = 76           # 8 + 32 + 4 + 32 bytes
MAGIC_MRD_HEADER = b"DCLC_M01"  # 8 bytes


class DeltaError(Exception):
    """Base exception for delta encoding/decoding errors."""
    pass


class DeltaCorruptionError(DeltaError):
    """Raised when delta container header or payload is corrupted."""
    pass


class BaseMismatchError(DeltaError):
    """Raised when the base chunk provided does not match the delta's reference digest."""
    pass


@dataclass
class DeltaResult:
    """Summary of a delta encoding operation."""
    is_accepted: bool
    delta_container: Optional[bytes]
    original_size: int
    standalone_compressed_size: int
    delta_total_size: int
    net_stored_size: int
    savings_bytes: int
    savings_ratio: float
    reconstruction_verified: bool
    rejection_reason: Optional[str] = None


def compress_standalone(data: bytes, level: int = 6) -> bytes:
    """Independently compress data using standard zlib."""
    return zlib.compress(data, level=level)


def encode_delta(base_bytes: bytes, target_bytes: bytes, level: int = 6) -> bytes:
    """Compress target_bytes with base_bytes as preset dictionary and package into container."""
    base_digest = hashlib.sha256(base_bytes).digest()
    target_digest = hashlib.sha256(target_bytes).digest()
    target_len = len(target_bytes)

    # Dictionary compression
    comp = zlib.compressobj(level=level, zdict=base_bytes)
    payload = comp.compress(target_bytes) + comp.flush()

    header = struct.pack(">8s32sI32s", MAGIC_HEADER, base_digest, target_len, target_digest)
    return header + payload


def decode_delta(base_bytes: bytes, delta_container: bytes) -> bytes:
    """Decompress target bytes from delta container using base_bytes."""
    if len(delta_container) < HEADER_SIZE:
        raise DeltaCorruptionError(f"Delta container truncated: {len(delta_container)} < {HEADER_SIZE} bytes.")

    magic, expected_base_digest, target_len, expected_target_digest = struct.unpack(
        ">8s32sI32s", delta_container[:HEADER_SIZE]
    )

    if magic != MAGIC_HEADER:
        raise DeltaCorruptionError(f"Invalid magic header: {magic!r}, expected {MAGIC_HEADER!r}.")

    actual_base_digest = hashlib.sha256(base_bytes).digest()
    if actual_base_digest != expected_base_digest:
        raise BaseMismatchError("Provided base chunk SHA-256 does not match delta reference digest.")

    payload = delta_container[HEADER_SIZE:]
    try:
        decomp = zlib.decompressobj(zdict=base_bytes)
        reconstructed = decomp.decompress(payload) + decomp.flush()
    except Exception as e:
        raise DeltaCorruptionError(f"Decompression failed: {e}") from e

    if len(reconstructed) != target_len:
        raise DeltaCorruptionError(
            f"Decompressed length mismatch: got {len(reconstructed)}, expected {target_len}."
        )

    actual_target_digest = hashlib.sha256(reconstructed).digest()
    if actual_target_digest != expected_target_digest:
        raise DeltaCorruptionError("Decompressed data SHA-256 checksum mismatch.")

    return reconstructed


def evaluate_and_encode_delta(
    base_bytes: bytes,
    target_bytes: bytes,
    compression_level: int = 6,
) -> DeltaResult:
    """Encode delta, verify reconstruction, and decide whether to accept or reject."""
    orig_size = len(target_bytes)
    standalone_comp = compress_standalone(target_bytes, level=compression_level)
    standalone_size = len(standalone_comp)

    # Identical chunks
    if base_bytes == target_bytes:
        # Exact duplicate: stored as 0 extra bytes in dedup systems
        return DeltaResult(
            is_accepted=True,
            delta_container=b"",
            original_size=orig_size,
            standalone_compressed_size=standalone_size,
            delta_total_size=0,
            net_stored_size=0,
            savings_bytes=orig_size,
            savings_ratio=1.0,
            reconstruction_verified=True,
            rejection_reason=None,
        )

    delta_container = encode_delta(base_bytes, target_bytes, level=compression_level)
    delta_size = len(delta_container)

    # 1. Verify byte-for-byte reconstruction
    try:
        reconstructed = decode_delta(base_bytes, delta_container)
        verified = (reconstructed == target_bytes)
    except Exception:
        verified = False

    if not verified:
        return DeltaResult(
            is_accepted=False,
            delta_container=None,
            original_size=orig_size,
            standalone_compressed_size=standalone_size,
            delta_total_size=delta_size,
            net_stored_size=standalone_size,
            savings_bytes=0,
            savings_ratio=0.0,
            reconstruction_verified=False,
            rejection_reason="Reconstruction verification failed.",
        )

    # 2. Economic benefit check: is total delta size smaller than standalone compressed target?
    if delta_size < standalone_size:
        savings = standalone_size - delta_size
        ratio = float(orig_size / delta_size) if delta_size > 0 else float("inf")
        return DeltaResult(
            is_accepted=True,
            delta_container=delta_container,
            original_size=orig_size,
            standalone_compressed_size=standalone_size,
            delta_total_size=delta_size,
            net_stored_size=delta_size,
            savings_bytes=savings,
            savings_ratio=ratio,
            reconstruction_verified=True,
            rejection_reason=None,
        )
    else:
        return DeltaResult(
            is_accepted=False,
            delta_container=None,
            original_size=orig_size,
            standalone_compressed_size=standalone_size,
            delta_total_size=delta_size,
            net_stored_size=standalone_size,
            savings_bytes=0,
            savings_ratio=0.0,
            reconstruction_verified=True,
            rejection_reason=f"Delta size ({delta_size} B) >= standalone compressed size ({standalone_size} B).",
        )


# ===========================================================================
# Phase B — Multi-Reference Delta Encoding (MRD)
# Implements Section 3.4 & Section 4.4 of Xiao et al. (2026)
# ===========================================================================

class MRDError(DeltaError):
    """Base exception for multi-reference delta encoding/decoding errors."""
    pass


@dataclass(frozen=True)
class MRDTargetMeta:
    """Metadata for an individual target chunk within an MRD group."""
    chunk_index: int
    sha256: str
    offset: int
    length: int


@dataclass(frozen=True)
class MRDReferenceMeta:
    """Metadata for a candidate reference chunk within an MRD group."""
    ref_index: int
    sha256: str
    length: int


@dataclass
class MRDResult:
    """Summary of a multi-reference delta encoding operation."""
    is_accepted: bool
    mrd_container: Optional[bytes]
    target_count: int
    reference_count: int
    total_target_raw_bytes: int
    standalone_compressed_size: int
    mrd_total_size: int
    net_stored_size: int
    savings_bytes: int
    savings_ratio: float
    reconstruction_verified: bool
    rejection_reason: Optional[str] = None
    target_metas: list[MRDTargetMeta] = field(default_factory=list)
    reference_metas: list[MRDReferenceMeta] = field(default_factory=list)


def encode_mrd(
    reference_chunks: list[bytes],
    target_chunks: list[bytes],
    level: int = 6,
) -> bytes:
    """Encode a group of target chunks using multiple candidate reference chunks.

    Paper Section 3.4 (Eq. 21):
      Adjacent reference chunks are concatenated into a single logical reference
      unit (bf_11 || bf_12 || ...), and adjacent target chunks within the input
      neighborhood window are concatenated into a single logical target unit
      (bf_21 || bf_22 || ...). Joint delta compression is performed over the
      concatenated units using preset dictionary compression.

    Args:
        reference_chunks: Ordered list of candidate base chunks (phi_i).
        target_chunks: Ordered list of target chunks within neighborhood window N(B_i).
        level: zlib compression level (default 6).

    Returns:
        Packed bytes containing DCLC_M01 container header and compressed payload.
    """
    if not reference_chunks:
        raise ValueError("reference_chunks must contain at least one chunk.")
    if not target_chunks:
        raise ValueError("target_chunks must contain at least one chunk.")

    # 1. Assemble logical reference super-block
    ref_metas_packed: list[tuple[bytes, int]] = []
    for r_idx, r_chunk in enumerate(reference_chunks):
        if not isinstance(r_chunk, (bytes, bytearray)):
            raise TypeError(f"reference chunk {r_idx} must be bytes, got {type(r_chunk)}")
        r_b = bytes(r_chunk)
        r_digest = hashlib.sha256(r_b).digest()
        r_len = len(r_b)
        ref_metas_packed.append((r_digest, r_len))

    combined_ref = b"".join(reference_chunks)
    combined_ref_digest = hashlib.sha256(combined_ref).digest()

    # 2. Assemble logical target super-block with byte offsets and lengths
    target_metas_packed: list[tuple[bytes, int, int]] = []
    curr_offset = 0
    for t_idx, t_chunk in enumerate(target_chunks):
        if not isinstance(t_chunk, (bytes, bytearray)):
            raise TypeError(f"target chunk {t_idx} must be bytes, got {type(t_chunk)}")
        t_b = bytes(t_chunk)
        t_digest = hashlib.sha256(t_b).digest()
        t_len = len(t_b)
        target_metas_packed.append((t_digest, curr_offset, t_len))
        curr_offset += t_len

    combined_target = b"".join(target_chunks)
    combined_target_len = len(combined_target)
    combined_target_digest = hashlib.sha256(combined_target).digest()

    # 3. Perform joint delta compression using combined reference as preset dictionary
    comp = zlib.compressobj(level=level, zdict=combined_ref)
    payload = comp.compress(combined_target) + comp.flush()

    # 4. Pack structured header:
    #   [0:8]   MAGIC_MRD_HEADER: b"DCLC_M01" (8 bytes)
    #   [8:10]  num_refs: uint16 (2 bytes)
    #   Followed by num_refs * (32 bytes sha256 + 4 bytes length = 36 bytes)
    #   Followed by 32 bytes combined_ref_digest
    #   Followed by num_targets: uint16 (2 bytes)
    #   Followed by num_targets * (32 bytes sha256 + 4 bytes offset + 4 bytes length = 40 bytes)
    #   Followed by 4 bytes combined_target_len + 32 bytes combined_target_digest (36 bytes)
    #   Followed by compressed payload
    header_parts = [
        struct.pack(">8sH", MAGIC_MRD_HEADER, len(reference_chunks))
    ]
    for r_digest, r_len in ref_metas_packed:
        header_parts.append(struct.pack(">32sI", r_digest, r_len))
    header_parts.append(combined_ref_digest)

    header_parts.append(struct.pack(">H", len(target_chunks)))
    for t_digest, t_offset, t_len in target_metas_packed:
        header_parts.append(struct.pack(">32sII", t_digest, t_offset, t_len))
    header_parts.append(struct.pack(">I32s", combined_target_len, combined_target_digest))

    header = b"".join(header_parts)
    return header + payload


def unpack_mrd_container(
    mrd_container: bytes,
) -> tuple[list[MRDReferenceMeta], bytes, list[MRDTargetMeta], int, bytes, bytes]:
    """Unpack and validate the header of an MRD container.

    Returns:
        Tuple of:
          - ref_metas: list of MRDReferenceMeta
          - expected_combined_ref_digest: 32 bytes
          - target_metas: list of MRDTargetMeta
          - expected_target_len: int
          - expected_combined_target_digest: 32 bytes
          - payload: bytes
    """
    if len(mrd_container) < 10:
        raise DeltaCorruptionError(f"MRD container truncated: {len(mrd_container)} < 10 bytes.")

    magic, num_refs = struct.unpack(">8sH", mrd_container[:10])
    if magic != MAGIC_MRD_HEADER:
        raise DeltaCorruptionError(f"Invalid MRD magic header: {magic!r}, expected {MAGIC_MRD_HEADER!r}.")

    offset = 10
    ref_metas: list[MRDReferenceMeta] = []
    for r_idx in range(num_refs):
        if len(mrd_container) < offset + 36:
            raise DeltaCorruptionError("MRD container truncated in reference metadata table.")
        r_digest, r_len = struct.unpack(">32sI", mrd_container[offset : offset + 36])
        ref_metas.append(MRDReferenceMeta(
            ref_index=r_idx,
            sha256=r_digest.hex(),
            length=r_len,
        ))
        offset += 36

    if len(mrd_container) < offset + 32 + 2:
        raise DeltaCorruptionError("MRD container truncated before target metadata table.")

    expected_combined_ref_digest = mrd_container[offset : offset + 32]
    offset += 32

    num_targets = struct.unpack(">H", mrd_container[offset : offset + 2])[0]
    offset += 2

    target_metas: list[MRDTargetMeta] = []
    for t_idx in range(num_targets):
        if len(mrd_container) < offset + 40:
            raise DeltaCorruptionError("MRD container truncated in target metadata table.")
        t_digest, t_offset, t_len = struct.unpack(">32sII", mrd_container[offset : offset + 40])
        target_metas.append(MRDTargetMeta(
            chunk_index=t_idx,
            sha256=t_digest.hex(),
            offset=t_offset,
            length=t_len,
        ))
        offset += 40

    if len(mrd_container) < offset + 4 + 32:
        raise DeltaCorruptionError("MRD container truncated before payload.")

    expected_target_len, expected_combined_target_digest = struct.unpack(
        ">I32s", mrd_container[offset : offset + 36]
    )
    offset += 36

    payload = mrd_container[offset:]
    return (
        ref_metas,
        expected_combined_ref_digest,
        target_metas,
        expected_target_len,
        expected_combined_target_digest,
        payload,
    )


def decode_mrd_group(
    reference_chunks: list[bytes],
    mrd_container: bytes,
) -> list[bytes]:
    """Reconstruct all target chunks in an MRD group (MG mode: Section 4.4).

    Reconstructs the full target super-block within a single restore operation
    and slices out individual target chunks according to stored offsets.

    Args:
        reference_chunks: Ordered list of all reference chunks required by this group.
        mrd_container: Packed MRD container bytes.

    Returns:
        List of reconstructed target chunks matching original input order and content.
    """
    (
        ref_metas,
        expected_combined_ref_digest,
        target_metas,
        expected_target_len,
        expected_combined_target_digest,
        payload,
    ) = unpack_mrd_container(mrd_container)

    # 1. Validate reference chunk count and integrity
    if len(reference_chunks) != len(ref_metas):
        raise BaseMismatchError(
            f"Reference count mismatch: container expects {len(ref_metas)}, but {len(reference_chunks)} provided."
        )

    for r_idx, (r_chunk, r_meta) in enumerate(zip(reference_chunks, ref_metas)):
        if len(r_chunk) != r_meta.length:
            raise BaseMismatchError(
                f"Reference chunk {r_idx} length mismatch: got {len(r_chunk)}, expected {r_meta.length}."
            )
        if hashlib.sha256(r_chunk).hexdigest() != r_meta.sha256:
            raise BaseMismatchError(
                f"Reference chunk {r_idx} SHA-256 mismatch."
            )

    combined_ref = b"".join(reference_chunks)
    if hashlib.sha256(combined_ref).digest() != expected_combined_ref_digest:
        raise BaseMismatchError("Combined reference buffer SHA-256 digest mismatch.")

    # 2. Decompress target super-block
    try:
        decomp = zlib.decompressobj(zdict=combined_ref)
        combined_target = decomp.decompress(payload) + decomp.flush()
    except Exception as e:
        raise DeltaCorruptionError(f"MRD decompression failed: {e}") from e

    if len(combined_target) != expected_target_len:
        raise DeltaCorruptionError(
            f"MRD decompressed length mismatch: got {len(combined_target)}, expected {expected_target_len}."
        )
    if hashlib.sha256(combined_target).digest() != expected_combined_target_digest:
        raise DeltaCorruptionError("MRD combined target SHA-256 digest mismatch.")

    # 3. Slice and verify individual target chunks
    reconstructed_chunks: list[bytes] = []
    for t_meta in target_metas:
        chunk_slice = combined_target[t_meta.offset : t_meta.offset + t_meta.length]
        if len(chunk_slice) != t_meta.length:
            raise DeltaCorruptionError(
                f"Target chunk {t_meta.chunk_index} sliced length {len(chunk_slice)} != expected {t_meta.length}."
            )
        if hashlib.sha256(chunk_slice).hexdigest() != t_meta.sha256:
            raise DeltaCorruptionError(
                f"Target chunk {t_meta.chunk_index} checksum mismatch."
            )
        reconstructed_chunks.append(chunk_slice)

    return reconstructed_chunks


def decode_mrd_single(
    reference_chunks: list[bytes],
    mrd_container: bytes,
    target_index: int,
) -> bytes:
    """Reconstruct a single target chunk from an MRD group (MS mode: Section 4.4).

    Reconstructs one target block from within an MRD group without requiring the
    caller to manage the group-level super-delta.

    Args:
        reference_chunks: All reference chunks required by the MRD group.
        mrd_container: Packed MRD container bytes.
        target_index: Index of the target chunk within the group (0 <= target_index < target_count).

    Returns:
        Exact byte content of the requested target chunk.
    """
    (
        ref_metas,
        expected_combined_ref_digest,
        target_metas,
        expected_target_len,
        expected_combined_target_digest,
        payload,
    ) = unpack_mrd_container(mrd_container)

    if not (0 <= target_index < len(target_metas)):
        raise IndexError(
            f"target_index {target_index} out of range [0, {len(target_metas)})"
        )

    # Validate reference chunks
    if len(reference_chunks) != len(ref_metas):
        raise BaseMismatchError(
            f"Reference count mismatch: container expects {len(ref_metas)}, but {len(reference_chunks)} provided."
        )
    for r_idx, (r_chunk, r_meta) in enumerate(zip(reference_chunks, ref_metas)):
        if len(r_chunk) != r_meta.length or hashlib.sha256(r_chunk).hexdigest() != r_meta.sha256:
            raise BaseMismatchError(f"Reference chunk {r_idx} mismatch.")

    combined_ref = b"".join(reference_chunks)
    if hashlib.sha256(combined_ref).digest() != expected_combined_ref_digest:
        raise BaseMismatchError("Combined reference buffer SHA-256 digest mismatch.")

    # Decompress full super-block to access slice
    try:
        decomp = zlib.decompressobj(zdict=combined_ref)
        combined_target = decomp.decompress(payload) + decomp.flush()
    except Exception as e:
        raise DeltaCorruptionError(f"MRD decompression failed: {e}") from e

    if len(combined_target) != expected_target_len:
        raise DeltaCorruptionError("MRD decompressed length mismatch.")

    t_meta = target_metas[target_index]
    chunk_slice = combined_target[t_meta.offset : t_meta.offset + t_meta.length]
    if hashlib.sha256(chunk_slice).hexdigest() != t_meta.sha256:
        raise DeltaCorruptionError(f"Target chunk {target_index} checksum mismatch.")

    return chunk_slice


def evaluate_and_encode_mrd(
    reference_chunks: list[bytes],
    target_chunks: list[bytes],
    compression_level: int = 6,
) -> MRDResult:
    """Encode an MRD group, verify reconstruction (MG mode), and evaluate economic benefit.

    Economic check:
      Total MRD container size must be smaller than the sum of independent
      standalone compressed chunks (sum_i len(compress_standalone(C_i))).
      If not, the MRD delta is rejected.

    Args:
        reference_chunks: Candidate base chunks (phi_i).
        target_chunks: Target window chunks N(B_i).
        compression_level: zlib compression level (default 6).

    Returns:
        MRDResult with detailed telemetry, verified flag, and rejection reason if rejected.
    """
    if not target_chunks:
        raise ValueError("target_chunks must not be empty.")
    if not reference_chunks:
        raise ValueError("reference_chunks must not be empty.")

    total_raw = sum(len(c) for c in target_chunks)
    standalone_sizes = [len(compress_standalone(c, level=compression_level)) for c in target_chunks]
    total_standalone = sum(standalone_sizes)

    try:
        container = encode_mrd(reference_chunks, target_chunks, level=compression_level)
    except Exception as e:
        return MRDResult(
            is_accepted=False,
            mrd_container=None,
            target_count=len(target_chunks),
            reference_count=len(reference_chunks),
            total_target_raw_bytes=total_raw,
            standalone_compressed_size=total_standalone,
            mrd_total_size=0,
            net_stored_size=total_standalone,
            savings_bytes=0,
            savings_ratio=0.0,
            reconstruction_verified=False,
            rejection_reason=f"Encoding failed: {e}",
        )

    mrd_size = len(container)

    # 1. Verify byte-for-byte reconstruction via MG
    try:
        reconstructed_group = decode_mrd_group(reference_chunks, container)
        verified = (reconstructed_group == target_chunks)
    except Exception:
        verified = False

    if not verified:
        return MRDResult(
            is_accepted=False,
            mrd_container=None,
            target_count=len(target_chunks),
            reference_count=len(reference_chunks),
            total_target_raw_bytes=total_raw,
            standalone_compressed_size=total_standalone,
            mrd_total_size=mrd_size,
            net_stored_size=total_standalone,
            savings_bytes=0,
            savings_ratio=0.0,
            reconstruction_verified=False,
            rejection_reason="Reconstruction verification failed.",
        )

    # Extract parsed metadata for inspection
    ref_metas, _, target_metas, _, _, _ = unpack_mrd_container(container)

    # 2. Economic benefit check: is MRD container smaller than sum of standalone compressed chunks?
    if mrd_size < total_standalone:
        savings = total_standalone - mrd_size
        ratio = float(total_raw / mrd_size) if mrd_size > 0 else float("inf")
        return MRDResult(
            is_accepted=True,
            mrd_container=container,
            target_count=len(target_chunks),
            reference_count=len(reference_chunks),
            total_target_raw_bytes=total_raw,
            standalone_compressed_size=total_standalone,
            mrd_total_size=mrd_size,
            net_stored_size=mrd_size,
            savings_bytes=savings,
            savings_ratio=ratio,
            reconstruction_verified=True,
            rejection_reason=None,
            target_metas=target_metas,
            reference_metas=ref_metas,
        )
    else:
        return MRDResult(
            is_accepted=False,
            mrd_container=None,
            target_count=len(target_chunks),
            reference_count=len(reference_chunks),
            total_target_raw_bytes=total_raw,
            standalone_compressed_size=total_standalone,
            mrd_total_size=mrd_size,
            net_stored_size=total_standalone,
            savings_bytes=0,
            savings_ratio=0.0,
            reconstruction_verified=True,
            rejection_reason=f"MRD size ({mrd_size} B) >= standalone compressed size ({total_standalone} B).",
            target_metas=target_metas,
            reference_metas=ref_metas,
        )
