"""
src/rag/retrieval/bm25_index.py
---------------------------------
DCLC-RAG Phase 6C — BM25 Okapi inverted index.

Implemented using only stdlib + numpy.  No external BM25 library.

Two build modes
---------------
dedup_aware  : One BM25 document per unique chunk_id. Shared chunks are
               indexed once.  Retrieval scores are keyed by chunk_id.
baseline     : One BM25 document per (doc_id, version_id, chunk_index)
               slot.  Duplicate content is indexed N times.  Slot keys
               have the form  "<doc_id>::<version_id>::<chunk_index>".

Both modes use the same tokenizer, same IDF computation, and the same
BM25 Okapi formula with identical default parameters (k1=1.5, b=0.75).

Tokenization
------------
Lowercase.  Split on whitespace and punctuation via re.  No stemming.
Preprocessing is deterministic and applied identically in both modes.

BM25 Okapi formula
------------------
  score(d, q) = sum_t [ IDF(t) * tf(t,d) * (k1+1) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl)) ]
  IDF(t)      = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.version_mapper import VersionRegistry


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """
    Lowercase, split on non-alphanumeric characters.
    Returns a list of token strings (may be empty for whitespace-only text).
    """
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------


class BM25Index:
    """
    BM25 Okapi inverted index.

    After build_dedup_aware() or build_baseline():
      - self.doc_keys   : list[str] — ordered list of document keys
      - self.vocab      : dict[str, int] — term → column index
      - self._tf        : list[Counter] — per-document term frequencies
      - self._idf       : np.ndarray — IDF vector, shape (V,)
      - self._dl        : np.ndarray — document lengths, shape (D,)
      - self._avgdl     : float

    Parameters
    ----------
    k1 : float
        Term saturation parameter. Default 1.5.
    b  : float
        Length normalisation parameter. Default 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_keys: List[str] = []
        self.vocab: Dict[str, int] = {}
        self._tokens_per_doc: List[List[str]] = []
        self._idf: np.ndarray = np.array([])
        self._dl: np.ndarray = np.array([])
        self._avgdl: float = 0.0
        self._built: bool = False

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def build_dedup_aware(
        self,
        chunk_index: ChunkVersionIndex,
    ) -> None:
        """
        Build BM25 index with one document per unique chunk_id.

        Text is taken from ChunkVersionIndex.get_text(), which stores the
        canonical text_content for each unique chunk_id.  Because chunk_ids
        are SHA-256 fingerprints of raw bytes, only byte-identical chunks
        share a key — semantic similarity is never merged.
        """
        keys = chunk_index.all_chunk_ids()
        texts = {cid: chunk_index.get_text(cid) for cid in keys}
        self._build(keys, texts)

    def build_baseline(
        self,
        registry: VersionRegistry,
    ) -> None:
        """
        Build BM25 index with one document per (doc_id, version_id, chunk_index)
        slot.  Duplicate content is indexed N times — one per version that
        contains it.  This is the standard-RAG baseline behavior.

        Slot key format: "<doc_id>::<version_id>::<chunk_index>"
        """
        keys: List[str] = []
        texts: Dict[str, str] = {}
        for doc_id in registry.list_doc_ids():
            for version in registry.get_all_versions(doc_id):
                for chunk in version.chunks:
                    key = f"{doc_id}::{version.version_id}::{chunk.chunk_index}"
                    keys.append(key)
                    texts[key] = chunk.text_content
        self._build(keys, texts)

    def _build(self, keys: List[str], texts: Dict[str, str]) -> None:
        """Internal: build IDF table and store per-doc tokens."""
        self.doc_keys = list(keys)
        self._tokens_per_doc = [tokenize(texts[k]) for k in self.doc_keys]

        # Document lengths
        self._dl = np.array([len(toks) for toks in self._tokens_per_doc], dtype=np.float64)
        self._avgdl = float(self._dl.mean()) if len(self._dl) > 0 else 1.0

        # Build vocabulary
        all_terms: set[str] = set()
        for toks in self._tokens_per_doc:
            all_terms.update(toks)
        self.vocab = {term: i for i, term in enumerate(sorted(all_terms))}

        # Document frequency for each term
        N = len(self.doc_keys)
        df = np.zeros(len(self.vocab), dtype=np.float64)
        for toks in self._tokens_per_doc:
            for term in set(toks):
                if term in self.vocab:
                    df[self.vocab[term]] += 1

        # IDF (smoothed)
        self._idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        self._built = True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        query: str,
        scope: Optional[set[str]] = None,
    ) -> Dict[str, float]:
        """
        Score all documents against the query using BM25 Okapi.

        Parameters
        ----------
        query : str
            Raw query text; tokenized internally.
        scope : set[str] | None
            If provided, only document keys in this set are scored.
            Documents outside the scope receive a score of 0.0 (not returned).

        Returns
        -------
        dict[str, float]
            {doc_key: bm25_score} for all documents with score > 0.
            Documents with score 0 are omitted.
        """
        if not self._built:
            raise RuntimeError("BM25Index has not been built yet. Call build_*() first.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return {}

        N = len(self.doc_keys)
        scores: Dict[str, float] = {}

        for i, key in enumerate(self.doc_keys):
            if scope is not None and key not in scope:
                continue

            doc_tokens = self._tokens_per_doc[i]
            if not doc_tokens:
                continue

            dl = self._dl[i]
            tf_counter = Counter(doc_tokens)
            score = 0.0

            for term in query_tokens:
                if term not in self.vocab:
                    continue
                t_idx = self.vocab[term]
                tf = tf_counter.get(term, 0)
                idf = self._idf[t_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                score += idf * (numerator / denominator)

            if score > 0.0:
                scores[key] = score

        return scores

    def top_k(
        self,
        query: str,
        k: int = 10,
        scope: Optional[set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k (doc_key, score) pairs sorted by descending score.
        """
        raw = self.score(query, scope=scope)
        sorted_items = sorted(raw.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """Number of documents in this index."""
        return len(self.doc_keys)

    @property
    def vocabulary_size(self) -> int:
        """Number of unique terms in the vocabulary."""
        return len(self.vocab)

    @property
    def average_doc_length(self) -> float:
        """Average document length in tokens."""
        return self._avgdl
