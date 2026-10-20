"""
src/rag/retrieval/hybrid_retriever.py
--------------------------------------
DCLC-RAG Phase 6C — Hybrid Retriever (BM25 + Dense Cosine).

Combines BM25 keyword matching and dense vector similarity with
deduplication-aware shared embedding resolution and version filtering.

Score Normalization and Fusion
------------------------------
BM25 scores and Cosine similarities are normalized to [0, 1] per query
over all candidate chunks scored:
    norm(s) = (s - min_s) / (max_s - min_s + 1e-9)

Fused score:
    score = alpha * norm(bm25) + (1.0 - alpha) * norm(cosine)

Version Filtering
-----------------
If doc_id and/or version_id are provided:
- Results are strictly restricted to chunks that have at least one VersionRef
  matching the requested doc_id / version_id scope.
- Chunks whose references do not intersect the scope are completely excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex, VersionRef
from src.rag.retrieval.embedding_store import EmbeddingStore


@dataclass
class RetrievalResult:
    """
    Result returned by HybridRetriever.

    Attributes
    ----------
    chunk_id      : SHA-256 fingerprint of the chunk bytes.
    score         : Fused hybrid score in [0, 1].
    bm25_score    : Raw BM25 score.
    cosine_score  : Raw cosine similarity score.
    text_content  : Canonical text content of the chunk.
    version_refs  : All versions that share this chunk.
    filtered_refs : Subset of version_refs matching the query scope.
    """
    chunk_id: str
    score: float
    bm25_score: float
    cosine_score: float
    text_content: str
    version_refs: List[VersionRef]
    filtered_refs: List[VersionRef]


class HybridRetriever:
    """
    Deduplication-aware Hybrid Retriever combining BM25 and dense embeddings.
    """

    def __init__(
        self,
        bm25: BM25Index,
        store: EmbeddingStore,
        version_index: ChunkVersionIndex,
        default_alpha: float = 0.5,
    ) -> None:
        self.bm25 = bm25
        self.store = store
        self.version_index = version_index
        self.default_alpha = default_alpha

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
        doc_id: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Execute hybrid retrieval with version filtering.

        Parameters
        ----------
        query      : Query text string.
        top_k      : Number of top results to return.
        alpha      : Weight for BM25 (1 - alpha for dense). Defaults to default_alpha.
        doc_id     : If set, scope results to this document.
        version_id : If set, scope results to this specific version.

        Returns
        -------
        list[RetrievalResult] sorted by descending fused score.
        """
        if alpha is None:
            alpha = self.default_alpha

        # 1. Determine candidate chunk scope if filtering is requested
        all_candidate_ids = self.version_index.all_chunk_ids()
        valid_chunk_ids: Set[str] = set()

        if doc_id is not None or version_id is not None:
            for cid in all_candidate_ids:
                matching_refs = self.version_index.filter_refs(
                    cid, doc_id=doc_id, version_id=version_id
                )
                if matching_refs:
                    valid_chunk_ids.add(cid)
        else:
            valid_chunk_ids = set(all_candidate_ids)

        if not valid_chunk_ids:
            return []

        # 2. Compute BM25 scores
        raw_bm25 = self.bm25.score(query, scope=valid_chunk_ids)

        # 3. Compute Dense Cosine scores
        q_vec = self.store.query_vector(query)
        raw_cosine = self.store.cosine_scores(q_vec, scope=valid_chunk_ids)

        # 4. Normalization and Fusion
        # Union of chunks that scored > 0 in BM25 or have a cosine score
        scored_chunk_ids = set(raw_bm25.keys()) | set(raw_cosine.keys())
        if not scored_chunk_ids:
            return []

        bm25_vals = [raw_bm25.get(cid, 0.0) for cid in scored_chunk_ids]
        cos_vals = [raw_cosine.get(cid, 0.0) for cid in scored_chunk_ids]

        min_bm25, max_bm25 = min(bm25_vals), max(bm25_vals)
        min_cos, max_cos = min(cos_vals), max(cos_vals)

        diff_bm25 = (max_bm25 - min_bm25) if (max_bm25 > min_bm25) else 1.0
        diff_cos = (max_cos - min_cos) if (max_cos > min_cos) else 1.0

        results: List[RetrievalResult] = []
        for cid in scored_chunk_ids:
            b_raw = raw_bm25.get(cid, 0.0)
            c_raw = raw_cosine.get(cid, 0.0)

            # Min-max normalization
            b_norm = (b_raw - min_bm25) / diff_bm25 if (max_bm25 > min_bm25) else (1.0 if b_raw > 0 else 0.0)
            c_norm = (c_raw - min_cos) / diff_cos if (max_cos > min_cos) else (1.0 if c_raw > 0 else 0.0)

            fused_score = alpha * b_norm + (1.0 - alpha) * c_norm

            all_refs = self.version_index.get_refs(cid)
            filtered_refs = self.version_index.filter_refs(cid, doc_id=doc_id, version_id=version_id)

            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    score=float(fused_score),
                    bm25_score=float(b_raw),
                    cosine_score=float(c_raw),
                    text_content=self.version_index.get_text(cid),
                    version_refs=all_refs,
                    filtered_refs=filtered_refs,
                )
            )

        # Sort descending by fused score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
