"""
src/rag/version_mapper.py
--------------------------
DCLC-RAG Phase 6B — Version Mapper.

Maps logical documents and their versions to an ordered sequence of DCLC
content-defined chunks (FastCDC).

Design decisions:
  - Chunking is performed via the existing fastcdc_chunks() interface which
    returns list[bytes] (raw chunk payloads, no offsets).  Byte offsets are
    computed here with a running cumulative sum.
  - A chunk is an "exact duplicate" iff sha256(raw_bytes) matches a chunk
    already seen across ANY previously ingested document or version.
    is_shared is set to True for such chunks.
  - The registry is in-memory only for Phase 6B (no database dependency).
  - doc_id  = sha256(filename.encode("utf-8"))[:16]  — stable across runs.
  - version_id = sha256(raw_file_bytes)[:16]          — content-addressable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.fastcdc_chunker import fastcdc_chunks
from src.rag.document_parser import ParsedDocument, parse_document


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChunkRecord:
    """
    A single FastCDC chunk within a document version.

    Attributes
    ----------
    chunk_id         : SHA-256 hex digest of raw_bytes (content address).
    chunk_index      : Zero-based position of this chunk within the version.
    byte_offset      : Byte offset of the chunk start within the original file.
    byte_length      : Number of bytes in this chunk.
    raw_bytes        : Exact bytes of this chunk (used for DCLC delta/dedup).
    text_content     : Text representation of the chunk (may be empty for binary-ish segments).
    encoding_detected: Codec used to decode text_content.
    is_shared        : True if another document/version has the same chunk_id.
    """

    chunk_id: str
    chunk_index: int
    byte_offset: int
    byte_length: int
    raw_bytes: bytes
    text_content: str
    encoding_detected: str
    is_shared: bool = False


@dataclass
class DocumentVersion:
    """
    One ingested version of a logical document.

    Attributes
    ----------
    doc_id         : Stable identifier derived from the filename.
    version_id     : Content-derived identifier (sha256 of file bytes)[:16].
    filename       : Basename of the file.
    extension      : Lowercase extension, e.g. ".txt".
    content_sha256 : Full SHA-256 of the raw file bytes.
    chunks         : Ordered list of ChunkRecords for this version.
    """

    doc_id: str
    version_id: str
    filename: str
    extension: str
    content_sha256: str
    chunks: List[ChunkRecord] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        """Sum of all chunk byte lengths (equals file size)."""
        return sum(c.byte_length for c in self.chunks)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class VersionRegistry:
    """
    In-memory store for document versions and their DCLC chunk maps.

    Storage layout::

        _registry : { doc_id: { version_id: DocumentVersion } }
        _chunk_seen : { chunk_id }   — global set of known chunk fingerprints

    Thread safety: not guaranteed (single-threaded use in Phase 6B).
    """

    def __init__(self) -> None:
        # { doc_id: { version_id: DocumentVersion } }
        self._registry: Dict[str, Dict[str, DocumentVersion]] = {}
        # Global chunk fingerprint pool — used to detect shared chunks
        # across ALL documents and versions.
        self._chunk_seen: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        path: "Path | str",
        min_size: int = 4096,
        avg_size: int = 8192,
        max_size: int = 16384,
    ) -> DocumentVersion:
        """
        Parse a file, chunk it with FastCDC, and register the version.

        If the identical content (same version_id) has already been ingested,
        the existing DocumentVersion is returned without re-chunking.

        Parameters
        ----------
        path      : Path to the document file.
        min_size  : FastCDC minimum chunk size in bytes.
        avg_size  : FastCDC average (target) chunk size in bytes.
        max_size  : FastCDC maximum chunk size in bytes.

        Returns
        -------
        DocumentVersion
        """
        parsed: ParsedDocument = parse_document(path)

        doc_id = _make_doc_id(parsed.filename)
        version_id = _make_version_id(parsed.raw_bytes)

        # Return cached version if already ingested.
        if doc_id in self._registry and version_id in self._registry[doc_id]:
            return self._registry[doc_id][version_id]

        # Chunk the raw file bytes with FastCDC.
        raw_chunks: List[bytes] = fastcdc_chunks(
            parsed.raw_bytes, min_size=min_size, avg_size=avg_size, max_size=max_size
        )

        chunk_records: List[ChunkRecord] = []
        byte_offset: int = 0

        for idx, chunk_bytes in enumerate(raw_chunks):
            chunk_id = hashlib.sha256(chunk_bytes).hexdigest()

            # Mark as shared if another document/version already holds this chunk.
            is_shared = chunk_id in self._chunk_seen
            # Register so future versions can detect sharing.
            self._chunk_seen.add(chunk_id)

            # Decode text from the chunk bytes.
            text_content, encoding_detected = _decode_chunk_text(
                chunk_bytes, parsed.encoding_detected
            )

            record = ChunkRecord(
                chunk_id=chunk_id,
                chunk_index=idx,
                byte_offset=byte_offset,
                byte_length=len(chunk_bytes),
                raw_bytes=chunk_bytes,
                text_content=text_content,
                encoding_detected=encoding_detected,
                is_shared=is_shared,
            )
            chunk_records.append(record)
            byte_offset += len(chunk_bytes)

        version = DocumentVersion(
            doc_id=doc_id,
            version_id=version_id,
            filename=parsed.filename,
            extension=parsed.extension,
            content_sha256=parsed.content_sha256,
            chunks=chunk_records,
        )

        # Store in registry.
        self._registry.setdefault(doc_id, {})[version_id] = version
        return version

    def get_version(self, doc_id: str, version_id: str) -> DocumentVersion:
        """
        Retrieve a specific version of a document.

        Raises
        ------
        KeyError
            If the doc_id or version_id is not found.
        """
        if doc_id not in self._registry:
            raise KeyError(f"Document '{doc_id}' not found in registry.")
        if version_id not in self._registry[doc_id]:
            raise KeyError(
                f"Version '{version_id}' not found for document '{doc_id}'."
            )
        return self._registry[doc_id][version_id]

    def get_all_versions(self, doc_id: str) -> List[DocumentVersion]:
        """
        Return all ingested versions for a given doc_id, in ingestion order.

        Raises
        ------
        KeyError
            If the doc_id is not found.
        """
        if doc_id not in self._registry:
            raise KeyError(f"Document '{doc_id}' not found in registry.")
        return list(self._registry[doc_id].values())

    def reconstruct_text(self, doc_id: str, version_id: str) -> str:
        """
        Reconstruct the full text of a version by concatenating chunk text.

        This is a best-effort reconstruction: it works correctly when the
        document is plain text and the chunks decode cleanly.  It is NOT a
        byte-exact reconstruction guarantee (use raw_bytes for that).

        Raises
        ------
        KeyError
            If the doc_id or version_id is not found.
        """
        version = self.get_version(doc_id, version_id)
        return "".join(c.text_content for c in version.chunks)

    def list_doc_ids(self) -> List[str]:
        """Return all registered doc_ids."""
        return list(self._registry.keys())

    def shared_chunk_count(self) -> int:
        """
        Return the total number of chunk slots (across all versions) where
        is_shared is True.
        """
        count = 0
        for versions in self._registry.values():
            for version in versions.values():
                count += sum(1 for c in version.chunks if c.is_shared)
        return count


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_doc_id(filename: str) -> str:
    """Stable 16-char doc identifier from filename bytes."""
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]


def _make_version_id(raw_bytes: bytes) -> str:
    """Content-addressed 16-char version identifier."""
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def _decode_chunk_text(
    chunk_bytes: bytes, preferred_codec: str
) -> Tuple[str, str]:
    """
    Attempt to decode chunk bytes to text.

    Tries the preferred codec first (inherited from ParsedDocument), then
    falls back to latin-1 (which never raises UnicodeDecodeError).

    Returns
    -------
    (text_content, encoding_detected)
    """
    codecs_to_try = [preferred_codec, "utf-8", "latin-1"]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: List[str] = []
    for c in codecs_to_try:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for codec in ordered:
        try:
            return chunk_bytes.decode(codec), codec
        except (UnicodeDecodeError, LookupError):
            continue

    # latin-1 is the last fallback and should never fail, but satisfy the
    # type checker with an explicit error.
    raise RuntimeError(
        "Unreachable: latin-1 decode should never fail on byte data."
    )
