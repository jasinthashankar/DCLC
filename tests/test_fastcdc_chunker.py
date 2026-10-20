
import pytest

from src.fastcdc_chunker import fastcdc_chunks, chunk_stats


def test_empty_input():
    assert list(fastcdc_chunks(b"")) == []


def test_chunks_reconstruct_original_data():
    data = b"DCLC sample data\n" * 5000
    chunks = list(fastcdc_chunks(data, 256, 512, 1024))
    assert b"".join(chunks) == data


def test_repeated_runs_are_deterministic():
    data = bytes(range(256)) * 100
    first = list(fastcdc_chunks(data, 256, 512, 1024))
    second = list(fastcdc_chunks(data, 256, 512, 1024))
    assert first == second


def test_maximum_chunk_size():
    data = b"x" * 10000
    chunks = list(fastcdc_chunks(data, 256, 512, 1024))
    assert all(len(chunk) <= 1024 for chunk in chunks)
    assert b"".join(chunks) == data


def test_invalid_parameters():
    with pytest.raises(ValueError):
        list(fastcdc_chunks(b"hello", 100, 50, 200))


def test_non_bytes_input():
    with pytest.raises(TypeError):
        list(fastcdc_chunks("not bytes"))


def test_chunk_statistics():
    stats = chunk_stats([b"abc", b"12345"])
    assert stats["count"] == 2
    assert stats["total_bytes"] == 8
    assert stats["min_bytes"] == 3
    assert stats["max_bytes"] == 5


def test_chunk_sizes_respect_minimum_except_final_chunk():
    data = bytes(range(256)) * 100
    chunks = list(fastcdc_chunks(data, 256, 512, 1024))
    assert all(len(chunk) >= 256 for chunk in chunks[:-1])


def test_default_parameters():
    data = b"DCLC test data " * 10000
    chunks = list(fastcdc_chunks(data))
    assert b"".join(chunks) == data
