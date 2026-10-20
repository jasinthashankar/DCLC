
"""Initial feature extraction inspired by DCLC Algorithm 1.

Research note:
The paper excerpt does not specify the exact LSH family or hash
configurations. This implementation uses a documented, deterministic
random-projection LSH and deterministic shingle hashing as a prototype.
"""

import hashlib
import numpy as np


def split_into_subchunks(data: bytes, k: int) -> list[bytes]:
    """Split bytes into k nearly equal, ordered sub-chunks."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if k <= 0:
        raise ValueError("k must be positive")
    if not data:
        return [b""] * k

    # Divide the byte positions into k nearly equal ranges.
    boundaries = [(i * len(data)) // k for i in range(k + 1)]
    return [
        data[boundaries[i]:boundaries[i + 1]]
        for i in range(k)
    ]


def _byte_features(data: bytes, dimensions: int) -> np.ndarray:
    """Create a fixed-size numeric representation of bytes."""
    vector = np.zeros(dimensions, dtype=np.float32)

    for index, value in enumerate(data):
        vector[index % dimensions] += value / 255.0

    if data:
        vector /= len(data)

    return vector


def _lsh_signature(
    data: bytes,
    dimensions: int,
    seed: int,
) -> tuple[int, ...]:
    """Generate a deterministic random-projection LSH signature."""
    features = _byte_features(data, dimensions)
    rng = np.random.default_rng(seed)
    projections = rng.standard_normal((dimensions, dimensions))
    scores = projections @ features

    return tuple(int(score >= 0) for score in scores)


def _hash_shingle(shingle: tuple, dimensions: int) -> np.ndarray:
    """Map a shingle to a deterministic M-dimensional vector."""
    encoded = repr(shingle).encode("utf-8")
    digest = hashlib.shake_256(encoded).digest(dimensions * 4)

    values = np.frombuffer(digest, dtype=np.uint32).astype(np.float32)
    vector = (values / np.float32(2**32 - 1)) * 2.0 - 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm

    return vector


def build_shingles(
    signatures: list[tuple[int, ...]],
    n: int,
) -> set[tuple[tuple[int, ...], ...]]:
    """Build a set of multi-scale shingles following DCLC Algorithm 1 (lines 6-13).

    For each scale r in 1...N and chunk position t in 1...K:
        left = max(1, t - r)
        right = min(K, t + r)
        s = (h_{i, left}, ..., h_{i, right})
        S_i = S_i U {s}

    Using a set automatically eliminates duplicate shingles across scales
    and positions.
    """
    k = len(signatures)
    if k == 0 or n <= 0:
        return set()

    shingles: set[tuple[tuple[int, ...], ...]] = set()

    # Algorithm 1 lines 6-13
    for r in range(1, n + 1):
        for t in range(1, k + 1):
            left = max(1, t - r)
            right = min(k, t + r)
            # Converting 1-based [left, right] to 0-based slice [left - 1 : right]
            shingle = tuple(signatures[left - 1 : right])
            shingles.add(shingle)

    return shingles


def extract_initial_feature(
    chunk: bytes,
    k: int = 16,
    n: int = 5,
    m: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Return an M-dimensional initial feature vector for a chunk.

    Follows DCLC Algorithm 1:
      1. Splits chunk into K sub-chunks.
      2. Computes LSH signature for each sub-chunk using a consistent projection.
      3. Constructs multi-scale shingle set S_i over r in 1...N and t in 1...K.
      4. Maps each unique shingle s to an M-dimensional vector V(s),
         normalizes V(s) by its L2 norm, and aggregates:
         e_i = (1 / |S_i|) * sum_{s in S_i} (V(s) / ||V(s)||).
    """
    if not isinstance(chunk, bytes):
        raise TypeError("chunk must be bytes")
    if k <= 0 or m <= 0:
        raise ValueError("k and m must be positive")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    subchunks = split_into_subchunks(chunk, k)

    # Use the same seed for all sub-chunks so LSH projections are consistent
    signatures = [
        _lsh_signature(part, m, seed)
        for part in subchunks
    ]

    shingle_set = build_shingles(signatures, n)

    if not shingle_set:
        return np.zeros(m, dtype=np.float32)

    # Algorithm 1 lines 14-18:
    # sum_vec <- sum_vec + V(s) / ||V(s)||
    # ei <- (1 / |Si|) * sum_vec
    sum_vec = np.zeros(m, dtype=np.float32)
    for shingle in shingle_set:
        # _hash_shingle returns L2-normalized vector V(s) / ||V(s)||
        v_s = _hash_shingle(shingle, m)
        sum_vec += v_s

    return (sum_vec / float(len(shingle_set))).astype(np.float32)

