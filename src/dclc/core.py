"""Deterministic fixed-size chunk baseline for DCLC experiments.

This module intentionally does not claim to implement content-defined chunking,
learned resemblance detection, or delta encoding. It provides a simple baseline
that can be measured and later compared with the research implementation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256


@dataclass(frozen=True)
class PairAnalysis:
    source_size_bytes: int
    target_size_bytes: int
    chunk_size_bytes: int
    source_chunk_count: int
    target_chunk_count: int
    matched_target_chunk_count: int
    matched_target_bytes: int
    target_chunk_match_ratio: float
    target_byte_savings_ratio: float
    source_sha256: str
    target_sha256: str
    files_identical: bool

    def to_dict(self) -> dict[str, int | float | str | bool]:
        """Return a JSON-friendly representation."""
        return asdict(self)


def split_chunks(data: bytes, chunk_size: int) -> list[bytes]:
    """Split bytes into deterministic, fixed-size chunks.

    The final chunk may be shorter than ``chunk_size``. Empty input produces
    zero chunks. Fixed-size chunking is a baseline, not content-defined chunking.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    return [data[offset : offset + chunk_size] for offset in range(0, len(data), chunk_size)]


def _chunk_counter(chunks: list[bytes]) -> Counter[tuple[str, int]]:
    return Counter((sha256(chunk).hexdigest(), len(chunk)) for chunk in chunks)


def analyze_pair(source: bytes, target: bytes, chunk_size: int = 4096) -> PairAnalysis:
    """Estimate exact-chunk reuse from source to target.

    Repeated chunks are counted as a multiset, so a chunk occurrence cannot be
    reused more times than it appears in the source. Hash matches are additionally
    keyed by length. This metric only recognizes aligned fixed-size chunks.
    """
    if not isinstance(source, bytes) or not isinstance(target, bytes):
        raise TypeError("source and target must be bytes")
    source_chunks = split_chunks(source, chunk_size)
    target_chunks = split_chunks(target, chunk_size)
    source_counts = _chunk_counter(source_chunks)
    target_counts = _chunk_counter(target_chunks)

    matched_count = 0
    matched_bytes = 0
    for key, target_count in target_counts.items():
        reused = min(source_counts.get(key, 0), target_count)
        matched_count += reused
        matched_bytes += reused * key[1]

    target_count = len(target_chunks)
    return PairAnalysis(
        source_size_bytes=len(source),
        target_size_bytes=len(target),
        chunk_size_bytes=chunk_size,
        source_chunk_count=len(source_chunks),
        target_chunk_count=target_count,
        matched_target_chunk_count=matched_count,
        matched_target_bytes=matched_bytes,
        target_chunk_match_ratio=(matched_count / target_count if target_count else 0.0),
        target_byte_savings_ratio=(matched_bytes / len(target) if target else 0.0),
        source_sha256=sha256(source).hexdigest(),
        target_sha256=sha256(target).hexdigest(),
        files_identical=(source == target),
    )
