
import numpy as np
import pytest

from src.context_builder import (
    build_context_sequence,
    build_all_context_sequences,
)


def test_context_shape():
    features = [np.array([i, i + 1], dtype=np.float32) for i in range(5)]
    result = build_context_sequence(features, center_index=2, radius=2)
    assert result.shape == (5, 2)


def test_center_chunk_is_included():
    features = [np.array([i], dtype=np.float32) for i in range(5)]
    result = build_context_sequence(features, center_index=2, radius=1)
    np.testing.assert_array_equal(result[:, 0], [1, 2, 3])


def test_left_boundary_is_zero_padded():
    features = [np.array([1, 2]), np.array([3, 4])]
    result = build_context_sequence(features, center_index=0, radius=1)
    np.testing.assert_array_equal(result[0], [0, 0])
    np.testing.assert_array_equal(result[1], [1, 2])
    np.testing.assert_array_equal(result[2], [3, 4])


def test_all_contexts_shape():
    features = [np.array([i, i + 1]) for i in range(4)]
    result = build_all_context_sequences(features, radius=1)
    assert result.shape == (4, 3, 2)


def test_invalid_center_index():
    with pytest.raises(IndexError):
        build_context_sequence([np.array([1.0])], center_index=2)


def test_empty_features():
    with pytest.raises(ValueError):
        build_all_context_sequences([])


def test_inconsistent_dimensions():
    features = [np.array([1, 2]), np.array([3, 4, 5])]
    with pytest.raises(ValueError):
        build_context_sequence(features, center_index=0)


def test_negative_radius():
    with pytest.raises(ValueError):
        build_context_sequence([np.array([1.0])], center_index=0, radius=-1)


# ── Paper alignment tests ──────────────────────────────────────────────────────

def test_default_radius_is_four_single():
    """Paper Section 3.2 sets K=4 → window 2K+1=9.

    build_context_sequence() called without an explicit radius must produce
    a sequence of length 9.
    """
    features = [np.ones(128, dtype=np.float32) for _ in range(20)]
    result = build_context_sequence(features, center_index=10)
    assert result.shape == (9, 128), (
        f"Expected window of 9 (radius=4), got shape {result.shape}"
    )


def test_default_radius_is_four_all():
    """build_all_context_sequences() called without an explicit radius must
    produce context windows of length 9 (paper K=4, Section 3.2).
    """
    features = [np.ones(128, dtype=np.float32) for _ in range(10)]
    result = build_all_context_sequences(features)
    assert result.shape == (10, 9, 128), (
        f"Expected shape (10, 9, 128), got {result.shape}"
    )
