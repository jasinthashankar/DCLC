import pytest

from src.dclc.core import analyze_pair, split_chunks


def test_split_chunks_and_short_final_chunk():
    assert split_chunks(b"abcdefghij", 4) == [b"abcd", b"efgh", b"ij"]


def test_empty_input_has_no_chunks():
    assert split_chunks(b"", 4) == []
    result = analyze_pair(b"", b"", 4)
    assert result.target_chunk_count == 0
    assert result.target_byte_savings_ratio == 0.0
    assert result.files_identical is True


def test_identical_files_have_full_reuse():
    data = b"abcdefghabcdefgh"
    result = analyze_pair(data, data, 4)
    assert result.files_identical is True
    assert result.matched_target_chunk_count == result.target_chunk_count
    assert result.matched_target_bytes == len(data)
    assert result.target_byte_savings_ratio == 1.0


def test_changed_chunk_is_not_counted_as_reused():
    result = analyze_pair(b"AAAABBBBCCCC", b"AAAAXXXXCCCC", 4)
    assert result.matched_target_chunk_count == 2
    assert result.matched_target_bytes == 8
    assert result.target_byte_savings_ratio == pytest.approx(8 / 12)
    assert result.files_identical is False


def test_chunk_reuse_respects_duplicate_occurrence_counts():
    result = analyze_pair(b"AAAA", b"AAAAAAAA", 4)
    assert result.matched_target_chunk_count == 1
    assert result.matched_target_bytes == 4


def test_invalid_chunk_size_rejected():
    with pytest.raises(ValueError):
        split_chunks(b"abc", 0)


def test_chunk_size_must_be_integer():
    with pytest.raises(ValueError):
        split_chunks(b"abc", True)
