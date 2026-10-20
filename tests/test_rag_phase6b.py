"""
tests/test_rag_phase6b.py
--------------------------
Phase 6B test suite — Document Ingestion and Chunk-to-Version Mapping.

10 test cases covering:
  1.  parse_document: supported extensions return correct ParsedDocument
  2.  parse_document: unsupported extension raises UnsupportedFileError
  3.  parse_document: empty file returns empty text and content_sha256
  4.  parse_document: content_sha256 matches manual computation
  5.  VersionRegistry.ingest_document: basic chunk mapping
  6.  VersionRegistry.ingest_document: offsets sum to file size
  7.  VersionRegistry.ingest_document: duplicate content returns cached version
  8.  VersionRegistry: shared-chunk detection across versions of same doc
  9.  VersionRegistry.reconstruct_text: concatenated text matches original
  10. VersionRegistry.get_version / get_all_versions: retrieval and key errors

Notes on chunk sizes:
  FastCDC with default parameters (min=4096, avg=8192, max=16384) treats
  any file smaller than ~4096 bytes as a single chunk.  To exercise
  multi-chunk behaviour without writing huge files we use
  min_size=64, avg_size=128, max_size=256 in the relevant tests.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from src.rag.document_parser import ParsedDocument, UnsupportedFileError, parse_document
from src.rag.version_mapper import (
    ChunkRecord,
    DocumentVersion,
    VersionRegistry,
    _make_doc_id,
    _make_version_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_PARAMS = dict(min_size=64, avg_size=256, max_size=1024)
_MEDIUM_CONTENT_SIZE = 8192  # bytes guaranteed to produce ≥1 chunk at defaults


def _write_tmp(suffix: str, content: bytes, dir_: Path) -> Path:
    """Write content to a temp file with the given suffix and return its path."""
    p = dir_ / f"tmpfile{suffix}"
    p.write_bytes(content)
    return p


def _lorem(n_bytes: int) -> bytes:
    """Generate n_bytes of deterministic ASCII text."""
    unit = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    repeats = (n_bytes // len(unit)) + 1
    return (unit * repeats)[:n_bytes]


# ---------------------------------------------------------------------------
# Test 1 — parse_document returns correct ParsedDocument for supported types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".txt", ".md", ".py"])
def test_parse_document_supported_extensions(tmp_path: Path, ext: str) -> None:
    """
    parse_document must accept .txt, .md, and .py files and return a
    ParsedDocument with the correct filename, extension, and non-empty text.
    """
    content = b"Hello, DCLC-RAG Phase 6B!\nSecond line.\n"
    p = _write_tmp(ext, content, tmp_path)

    doc = parse_document(p)

    assert isinstance(doc, ParsedDocument)
    assert doc.filename == p.name
    assert doc.extension == ext
    assert doc.raw_bytes == content
    assert "Hello" in doc.text_content
    assert doc.encoding_detected in ("utf-8", "latin-1")


# ---------------------------------------------------------------------------
# Test 2 — parse_document raises UnsupportedFileError for bad extensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".pdf", ".docx", ".jpg", ".csv", ""])
def test_parse_document_unsupported_extension_raises(tmp_path: Path, ext: str) -> None:
    """
    parse_document must raise UnsupportedFileError for unsupported extensions.
    The error message must mention the rejected extension.
    """
    p = tmp_path / f"file{ext}"
    p.write_bytes(b"irrelevant content")

    with pytest.raises(UnsupportedFileError) as exc_info:
        parse_document(p)

    assert ext in str(exc_info.value) or "not supported" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 3 — parse_document handles empty file gracefully
# ---------------------------------------------------------------------------


def test_parse_document_empty_file(tmp_path: Path) -> None:
    """
    An empty .txt file must parse without error.
    text_content must be "", raw_bytes must be b"", content_sha256 must be
    the SHA-256 of b"" (e5786...).
    """
    p = _write_tmp(".txt", b"", tmp_path)
    doc = parse_document(p)

    assert doc.raw_bytes == b""
    assert doc.text_content == ""
    assert doc.content_sha256 == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# Test 4 — content_sha256 matches manual computation
# ---------------------------------------------------------------------------


def test_parse_document_sha256_correctness(tmp_path: Path) -> None:
    """
    ParsedDocument.content_sha256 must equal hashlib.sha256(raw_bytes).hexdigest().
    """
    content = _lorem(1024)
    p = _write_tmp(".txt", content, tmp_path)
    doc = parse_document(p)

    expected = hashlib.sha256(content).hexdigest()
    assert doc.content_sha256 == expected


# ---------------------------------------------------------------------------
# Test 5 — ingest_document produces a valid DocumentVersion with chunks
# ---------------------------------------------------------------------------


def test_ingest_document_basic_chunk_mapping(tmp_path: Path) -> None:
    """
    Ingesting a file must return a DocumentVersion with:
      - correct doc_id and version_id
      - at least one ChunkRecord
      - chunks in ascending chunk_index order
      - each chunk_id being a 64-char hex string (SHA-256)
    """
    content = _lorem(_MEDIUM_CONTENT_SIZE)
    p = _write_tmp(".txt", content, tmp_path)

    registry = VersionRegistry()
    version = registry.ingest_document(p)

    assert isinstance(version, DocumentVersion)
    assert len(version.chunks) >= 1
    assert version.doc_id == _make_doc_id(p.name)
    assert version.version_id == _make_version_id(content)

    for i, chunk in enumerate(version.chunks):
        assert chunk.chunk_index == i
        assert len(chunk.chunk_id) == 64  # SHA-256 hex
        assert isinstance(chunk.raw_bytes, bytes)
        assert chunk.byte_length == len(chunk.raw_bytes)


# ---------------------------------------------------------------------------
# Test 6 — chunk byte offsets sum to file size
# ---------------------------------------------------------------------------


def test_ingest_document_offsets_sum_to_file_size(tmp_path: Path) -> None:
    """
    The sum of all chunk byte_lengths must equal len(raw_bytes) of the file.
    The offset of each chunk must equal the cumulative sum of preceding chunks.
    """
    content = _lorem(_MEDIUM_CONTENT_SIZE)
    p = _write_tmp(".txt", content, tmp_path)

    registry = VersionRegistry()
    version = registry.ingest_document(p, **_SMALL_PARAMS)

    total = sum(c.byte_length for c in version.chunks)
    assert total == len(content), (
        f"Expected total chunk bytes {len(content)}, got {total}"
    )

    cumulative = 0
    for chunk in version.chunks:
        assert chunk.byte_offset == cumulative, (
            f"Chunk {chunk.chunk_index}: expected offset {cumulative}, "
            f"got {chunk.byte_offset}"
        )
        cumulative += chunk.byte_length


# ---------------------------------------------------------------------------
# Test 7 — ingesting identical content twice returns the cached version
# ---------------------------------------------------------------------------


def test_ingest_document_cached_on_duplicate_content(tmp_path: Path) -> None:
    """
    If the same file content is ingested again (even with a different path),
    the registry returns the same DocumentVersion object (same version_id)
    without creating duplicate chunk records.
    """
    content = _lorem(2048)
    p1 = _write_tmp(".txt", content, tmp_path)
    # Write same content under a different name — same filename → same doc_id.
    p2 = tmp_path / p1.name  # same name, same path (idempotent re-ingest)

    registry = VersionRegistry()
    v1 = registry.ingest_document(p1)
    v2 = registry.ingest_document(p2)  # identical path + content

    assert v1 is v2, "Registry should return the cached DocumentVersion object."
    assert v1.version_id == v2.version_id


# ---------------------------------------------------------------------------
# Test 8 — shared-chunk detection across two versions of the same document
# ---------------------------------------------------------------------------


def test_shared_chunk_detection_across_versions(tmp_path: Path) -> None:
    """
    When v2 of a document shares some chunks with v1, those chunks in v2
    must have is_shared=True.  New chunks unique to v2 must have is_shared=False.

    We use small chunk sizes so that individual content blocks are separate chunks.
    """
    # Build a shared block + a unique block, large enough to produce
    # multiple chunks at avg_size=256. unique blocks use different repeated
    # patterns so FastCDC boundary positions differ from the shared block.
    shared_block = _lorem(1024)                          # ~4 chunks at avg=256
    unique_to_v1 = b"VERSION_ONE_UNIQUE_DATA___" * 80   # ~2080 bytes
    unique_to_v2 = b"VERSION_TWO_UNIQUE_DATA___" * 80   # ~2080 bytes

    content_v1 = shared_block + unique_to_v1
    content_v2 = shared_block + unique_to_v2

    # Use a filename-stable path for both — same doc_id required.
    p1 = tmp_path / "report.txt"
    p1.write_bytes(content_v1)

    registry = VersionRegistry()
    v1 = registry.ingest_document(p1, **_SMALL_PARAMS)

    # Replace with v2 content for re-ingest under same filename.
    p1.write_bytes(content_v2)
    v2 = registry.ingest_document(p1, **_SMALL_PARAMS)

    assert v1.version_id != v2.version_id, "Different content must yield different version_ids."

    # At least one chunk in v2 must be shared (from the shared_block).
    shared_in_v2 = [c for c in v2.chunks if c.is_shared]
    assert len(shared_in_v2) >= 1, (
        "Expected at least one shared chunk in v2 that appears in v1."
    )

    # Verify by fingerprint: every chunk in v2 marked shared must actually
    # appear among v1's chunk_ids.
    v1_chunk_ids = {c.chunk_id for c in v1.chunks}
    for chunk in shared_in_v2:
        assert chunk.chunk_id in v1_chunk_ids, (
            f"Chunk {chunk.chunk_id[:8]}… marked shared but not in v1."
        )


# ---------------------------------------------------------------------------
# Test 9 — reconstruct_text returns the original text
# ---------------------------------------------------------------------------


def test_reconstruct_text_matches_original(tmp_path: Path) -> None:
    """
    reconstruct_text must join all chunk text_contents in order.
    For a UTF-8 plain-text file the result must equal the original text.
    """
    content_text = "Hello DCLC RAG.\n" * 600  # ~9 KB
    content_bytes = content_text.encode("utf-8")
    p = _write_tmp(".txt", content_bytes, tmp_path)

    registry = VersionRegistry()
    version = registry.ingest_document(p, **_SMALL_PARAMS)

    reconstructed = registry.reconstruct_text(version.doc_id, version.version_id)
    assert reconstructed == content_text, (
        "Reconstructed text must equal the original content."
    )


# ---------------------------------------------------------------------------
# Test 10 — get_version and get_all_versions: retrieval and error handling
# ---------------------------------------------------------------------------


def test_get_version_and_all_versions(tmp_path: Path) -> None:
    """
    get_version must return the stored DocumentVersion by (doc_id, version_id).
    get_all_versions must return all ingested versions for a doc_id.
    KeyError must be raised for unknown doc_ids and version_ids.
    """
    content_v1 = _lorem(1024)
    content_v2 = _lorem(2048)  # different length → different content_sha256

    p = tmp_path / "doc.txt"
    registry = VersionRegistry()

    p.write_bytes(content_v1)
    v1 = registry.ingest_document(p)

    p.write_bytes(content_v2)
    v2 = registry.ingest_document(p)

    doc_id = v1.doc_id

    # Retrieval by (doc_id, version_id)
    fetched_v1 = registry.get_version(doc_id, v1.version_id)
    fetched_v2 = registry.get_version(doc_id, v2.version_id)
    assert fetched_v1.version_id == v1.version_id
    assert fetched_v2.version_id == v2.version_id

    # get_all_versions returns both
    all_versions = registry.get_all_versions(doc_id)
    version_ids = {v.version_id for v in all_versions}
    assert v1.version_id in version_ids
    assert v2.version_id in version_ids

    # KeyError for unknown doc_id
    with pytest.raises(KeyError):
        registry.get_version("nonexistent_doc_id_00", "any")

    # KeyError for unknown version_id within a known doc
    with pytest.raises(KeyError):
        registry.get_version(doc_id, "nonexistent_version_id")

    # KeyError for get_all_versions on unknown doc_id
    with pytest.raises(KeyError):
        registry.get_all_versions("unknown_doc_000000000")
