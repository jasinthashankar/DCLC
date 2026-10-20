
"""FastCDC chunking using the installed reference implementation."""

import fastcdc
from typing import Iterator


def fastcdc_chunks(
    data: bytes,
    min_size: int = 8 * 1024,
    avg_size: int = 16 * 1024,
    max_size: int = 32 * 1024,
) -> Iterator[bytes]:
    """Split bytes into variable-sized FastCDC chunks."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")

    if not (0 < min_size <= avg_size <= max_size):
        raise ValueError(
            "Require 0 < min_size <= avg_size <= max_size"
        )

    if not data:
        return

    for chunk in fastcdc.fastcdc(
        data,
        min_size=min_size,
        avg_size=avg_size,
        max_size=max_size,
        fat=True,
    ):
        yield chunk.data


def chunk_stats(chunks: list[bytes]) -> dict:
    """Calculate basic chunk statistics."""
    if not chunks:
        return {
            "count": 0,
            "total_bytes": 0,
            "min_bytes": 0,
            "max_bytes": 0,
            "average_bytes": 0.0,
        }

    sizes = [len(chunk) for chunk in chunks]

    return {
        "count": len(chunks),
        "total_bytes": sum(sizes),
        "min_bytes": min(sizes),
        "max_bytes": max(sizes),
        "average_bytes": sum(sizes) / len(sizes),
    }
