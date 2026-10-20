
import numpy as np
import pytest

from src.initial_feature_extractor import (
    split_into_subchunks,
    extract_initial_feature,
    build_shingles,
)


def test_split_preserves_original_data():
    data = b"hello DCLC project"
    parts = split_into_subchunks(data, 4)
    assert b"".join(parts) == data
    assert len(parts) == 4


def test_empty_input():
    parts = split_into_subchunks(b"", 4)
    assert len(parts) == 4
    assert all(part == b"" for part in parts)


def test_feature_has_expected_dimensions():
    feature = extract_initial_feature(
        b"sample chunk content", k=4, n=3, m=16
    )
    assert feature.shape == (16,)


def test_feature_values_are_finite():
    feature = extract_initial_feature(
        b"sample chunk content", k=4, n=3, m=16
    )
    assert np.isfinite(feature).all()


def test_extraction_is_deterministic():
    data = b"repeatable DCLC test"
    first = extract_initial_feature(data, k=4, n=3, m=16)
    second = extract_initial_feature(data, k=4, n=3, m=16)
    np.testing.assert_array_equal(first, second)


def test_invalid_window_size():
    with pytest.raises(ValueError):
        extract_initial_feature(b"test", k=4, n=0, m=16)
    with pytest.raises(ValueError):
        extract_initial_feature(b"test", k=4, n=-2, m=16)


def test_even_window_size_supported():
    feature = extract_initial_feature(b"test data for even n", k=4, n=4, m=16)
    assert feature.shape == (16,)
    assert np.isfinite(feature).all()


def test_invalid_k():
    with pytest.raises(ValueError):
        extract_initial_feature(b"test", k=0, n=3, m=16)


def test_invalid_input_type():
    with pytest.raises(TypeError):
        extract_initial_feature("not bytes", k=4, n=3, m=16)


def test_multiscale_shingles_nested_loops():
    # 4 distinct signatures
    signatures = [(1,), (2,), (3,), (4,)]
    # N = 2: scales r = 1 and r = 2
    shingles = build_shingles(signatures, n=2)

    # For r = 1:
    # t=1: left=1, right=2 -> ( (1,), (2,) )
    # t=2: left=1, right=3 -> ( (1,), (2,), (3,) )
    # t=3: left=2, right=4 -> ( (2,), (3,), (4,) )
    # t=4: left=3, right=4 -> ( (3,), (4,) )
    # For r = 2:
    # t=1: left=1, right=3 -> ( (1,), (2,), (3,) )  (already generated at r=1, t=2)
    # t=2: left=1, right=4 -> ( (1,), (2,), (3,), (4,) )
    # t=3: left=1, right=4 -> ( (1,), (2,), (3,), (4,) )  (duplicate)
    # t=4: left=2, right=4 -> ( (2,), (3,), (4,) )  (already generated at r=1, t=3)

    assert isinstance(shingles, set)
    # Check that shingles of different lengths exist (multi-scale)
    lengths = {len(s) for s in shingles}
    assert lengths == {2, 3, 4}


def test_duplicate_shingle_removal():
    # When sub-chunks have identical signatures, multiple combinations produce identical shingles
    identical_signatures = [(1, 0), (1, 0), (1, 0), (1, 0)]
    shingles = build_shingles(identical_signatures, n=2)

    # All shingles will consist of tuples of (1, 0) of lengths 2, 3, and 4
    assert isinstance(shingles, set)
    # Exactly 3 unique shingles can exist: length 2, length 3, length 4
    assert len(shingles) == 3


def test_shingle_boundary_computation():
    signatures = [(10,), (20,), (30,)]
    # For K=3, N=1:
    # r=1, t=1: left=max(1, 0)=1, right=min(3, 2)=2 -> (signatures[0], signatures[1])
    # r=1, t=2: left=max(1, 1)=1, right=min(3, 3)=3 -> (signatures[0], signatures[1], signatures[2])
    # r=1, t=3: left=max(1, 2)=2, right=min(3, 4)=3 -> (signatures[1], signatures[2])
    shingles = build_shingles(signatures, n=1)
    expected = {
        ((10,), (20,)),
        ((10,), (20,), (30,)),
        ((20,), (30,)),
    }
    assert shingles == expected

