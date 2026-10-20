"""
src/rag/retrieval/chunk_version_index.py
-----------------------------------------
DCLC-RAG Phase 6C — Reverse chunk-to-version index.

Builds a reverse map from chunk_id → all (doc_id, version_id, chunk_index)
tuples that contain that chunk, across all ingested documents and versions.

This is the structural foundation for shared-embedding deduplication:
  - A chunk_id with N version references → one embedding, N provenance records.
  - Exact identity is determined solely by chunk_id = sha256(raw_bytes).
  - Two chunks with different raw_bytes always have different chunk_ids,
    regardless of semantic similarity.

No external dependencies beyond stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from src.rag.version_mapper import VersionRegistry


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionRef:
    """
    One occurrence of a chunk within a specific document version.

    Attributes
    ----------
    doc_id      : Stable document identifier (sha256(filename)[:16]).
    version_id  : Content-derived version identifier (sha256(raw_bytes)[:16]).
    chunk_index : Zero-based position of the chunk within this version.
    filename    : Human-readable filename for reporting.
    """
    doc_id: str
    version_id: str
    chunk_index: int
    filename: str


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class ChunkVersionIndex:
    """
    Reverse map: chunk_id → frozenset[VersionRef].

    Built from an existing VersionRegistry. Read-only after build().

    Design notes
    ------------
    - chunk_id is the SHA-256 hex digest of the raw bytes of the chunk as
      stored in ChunkRecord.chunk_id.  Exact byte identity is required;
      semantic similarity is not considered.
    - A chunk that appears in N versions → N VersionRef entries under one
      chunk_id key.  Its text_content is deduplicated (all occurrences are
      identical, since the bytes are identical).
    - Unique text is stored once: chunk_id → text_content.
    """

    def __init__(self) -> None:
        # chunk_id → frozenset[VersionRef]
        self._index: Dict[str, Set[VersionRef]] = {}
        # chunk_id → canonical text_content (from first occurrence)
        self._text: Dict[str, str] = {}

    def build(self, registry: VersionRegistry) -> None:
        """
        Populate the index from all versions in the registry.

        Can be called multiple times to rebuild; clears previous state.
        """
        self._index.clear()
        self._text.clear()

        for doc_id in registry.list_doc_ids():
            for version in registry.get_all_versions(doc_id):
                for chunk in version.chunks:
                    ref = VersionRef(
                        doc_id=doc_id,
                        version_id=version.version_id,
                        chunk_index=chunk.chunk_index,
                        filename=version.filename,
                    )
                    self._index.setdefault(chunk.chunk_id, set()).add(ref)
                    # Store text_content once (byte-identical chunks have
                    # identical text after the same decode path).
                    if chunk.chunk_id not in self._text:
                        self._text[chunk.chunk_id] = chunk.text_content

    def get_refs(self, chunk_id: str) -> List[VersionRef]:
        """
        Return all VersionRef objects for a given chunk_id.

        Returns an empty list for unknown chunk_ids (no KeyError).
        """
        return list(self._index.get(chunk_id, set()))

    def get_text(self, chunk_id: str) -> str:
        """
        Return the canonical text_content for a chunk_id.

        Raises KeyError if the chunk_id is not indexed.
        """
        return self._text[chunk_id]

    def all_chunk_ids(self) -> List[str]:
        """Return all indexed chunk_ids in insertion-stable order."""
        return list(self._index.keys())

    def unique_chunk_count(self) -> int:
        """Number of distinct chunk_ids (unique content fingerprints)."""
        return len(self._index)

    def total_slots(self) -> int:
        """
        Total number of (version, chunk_index) slots across all versions.
        This equals the sum of len(version.chunks) for every version.
        unique_chunk_count() <= total_slots() always holds.
        """
        return sum(len(refs) for refs in self._index.values())

    def dedup_ratio(self) -> float:
        """
        Fraction of slots that are exact duplicates.

        dedup_ratio = 1 - unique_chunk_count / total_slots.
        0.0 = no duplicates; approaching 1.0 = almost everything is duplicated.
        Returns 0.0 if there are no slots.
        """
        total = self.total_slots()
        if total == 0:
            return 0.0
        return 1.0 - self.unique_chunk_count() / total

    def filter_refs(
        self,
        chunk_id: str,
        doc_id: str | None = None,
        version_id: str | None = None,
    ) -> List[VersionRef]:
        """
        Return VersionRef objects for chunk_id that pass the scope filter.

        Parameters
        ----------
        doc_id      : If set, only refs where ref.doc_id == doc_id are returned.
        version_id  : If set (requires doc_id), only refs where both match.

        This method is the enforcement point for version isolation:
        a chunk that exists only in version B cannot appear in a
        version A query, because its VersionRef set contains no A entry.
        """
        refs = self.get_refs(chunk_id)
        if doc_id is not None:
            refs = [r for r in refs if r.doc_id == doc_id]
        if version_id is not None:
            refs = [r for r in refs if r.version_id == version_id]
        return refs
