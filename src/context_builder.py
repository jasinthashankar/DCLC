
"""Build context sequences from initial DCLC feature vectors."""

import numpy as np


def build_context_sequence(
    features: list[np.ndarray],
    center_index: int,
    radius: int = 4,
) -> np.ndarray:
    """Return a context window centered on one chunk.

    Missing neighbors at sequence boundaries are zero-padded.
    The center chunk is included in the returned sequence.

    The paper (Section 3.2) specifies K = 4, giving a window of
    2K + 1 = 9 chunks.  The default ``radius=4`` matches this setting.

    Returns:
        Array of shape (2 * radius + 1, feature_dimension).
    """
    if not features:
        raise ValueError("features must not be empty")

    if radius < 0:
        raise ValueError("radius must be non-negative")

    if not 0 <= center_index < len(features):
        raise IndexError("center_index is outside the feature sequence")

    arrays = [np.asarray(feature, dtype=np.float32) for feature in features]

    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Every feature must be a one-dimensional vector")

    dimension = arrays[0].shape[0]

    if dimension == 0:
        raise ValueError("Feature vectors must not be empty")

    if any(array.shape[0] != dimension for array in arrays):
        raise ValueError("All feature vectors must have the same dimension")

    sequence = []

    for index in range(center_index - radius, center_index + radius + 1):
        if 0 <= index < len(arrays):
            sequence.append(arrays[index])
        else:
            sequence.append(np.zeros(dimension, dtype=np.float32))

    return np.stack(sequence).astype(np.float32)


def build_all_context_sequences(
    features: list[np.ndarray],
    radius: int = 4,
) -> np.ndarray:
    """Build one context sequence for every chunk.

    The paper (Section 3.2) specifies K = 4 (window of 9).
    The default ``radius=4`` matches this setting.
    """
    if not features:
        raise ValueError("features must not be empty")

    return np.stack([
        build_context_sequence(features, index, radius)
        for index in range(len(features))
    ])
