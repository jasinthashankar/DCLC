"""
src/rag/retrieval/baseline_retriever.py
----------------------------------------
DCLC-RAG Phase 6C — Baseline Standard-RAG Retriever.

Controlled comparison baseline:
- Represents standard RAG indexing where each document version has its
  own independent chunk slots.
- Duplicate chunks between versions are NOT deduplicated; each slot
  receives its own embedding vector and its own BM25 posting entry.
- Uses identical tokenization, BM25 scoring formula, embedding model,
  and similarity metric as the DCLC-aware retriever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.chunk_version_index import VersionRef
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.version_mapper import VersionRegistry


@dataclass
class BaselineResult:
    """
    Result returned by BaselineRetriever.

    Attributes
    ----------
    slot_key     : Identifies the specific slot "<doc_id>::<version_id>::<chunk_index>".
    score        : Fused hybrid score in [0, 1].
    bm25_score   : Raw BM25 score.
    cosine_score : Raw cosine similarity score.
    text_content : Text content of the slot.
    ref          : VersionRef describing this slot.
    """
    slot_key: str
    score: float
    bm25_score: float
    cosine_score: float
    text_content: str
    ref: VersionRef


class BaselineRetriever:
    """
    Standard (non-deduplicated) Hybrid Retriever.
    """

    def __init__(self, default_alpha: float = 0.5, precision: Optional[str] = None) -> None:
        self.bm25 = BM25Index()
        self.store = EmbeddingStore(precision=precision)
        self.default_alpha = default_alpha
        self._slots: Dict[str, Tuple[VersionRef, str]] = {}  # slot_key -> (ref, text)

    def build(
        self,
        registry: VersionRegistry,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        precision: Optional[str] = None,
    ) -> None:
        """
        Build baseline indexes over all version-chunk slots.
        """
        if precision is not None:
            self.store.precision = precision

        self._slots.clear()
        slot_keys: List[str] = []
        texts: List[str] = []

        for doc_id in registry.list_doc_ids():
            for version in registry.get_all_versions(doc_id):
                for chunk in version.chunks:
                    key = f"{doc_id}::{version.version_id}::{chunk.chunk_index}"
                    ref = VersionRef(
                        doc_id=doc_id,
                        version_id=version.version_id,
                        chunk_index=chunk.chunk_index,
                        filename=version.filename,
                    )
                    self._slots[key] = (ref, chunk.text_content)
                    slot_keys.append(key)
                    texts.append(chunk.text_content)

        # Build baseline BM25 index
        self.bm25.build_baseline(registry)

        # Build baseline embedding store (embeds each slot independently)
        self.store.build_baseline(
            chunk_ids=slot_keys,
            texts=texts,
            model_name=model_name,
            batch_size=batch_size,
            precision=precision,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
        doc_id: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> List[BaselineResult]:
        """
        Execute baseline retrieval with slot-level version filtering.
        """
        if alpha is None:
            alpha = self.default_alpha

        valid_slots: Set[str] = set()
        for key, (ref, _) in self._slots.items():
            if doc_id is not None and ref.doc_id != doc_id:
                continue
            if version_id is not None and ref.version_id != version_id:
                continue
            valid_slots.add(key)

        if not valid_slots:
            return []

        raw_bm25 = self.bm25.score(query, scope=valid_slots)
        q_vec = self.store.query_vector(query)
        raw_cosine = self.store.cosine_scores(q_vec, scope=valid_slots)

        scored_keys = set(raw_bm25.keys()) | set(raw_cosine.keys())
        if not scored_keys:
            return []

        bm25_vals = [raw_bm25.get(k, 0.0) for k in scored_keys]
        cos_vals = [raw_cosine.get(k, 0.0) for k in scored_keys]

        min_bm25, max_bm25 = min(bm25_vals), max(bm25_vals)
        min_cos, max_cos = min(cos_vals), max(cos_vals)

        diff_bm25 = (max_bm25 - min_bm25) if (max_bm25 > min_bm25) else 1.0
        diff_cos = (max_cos - min_cos) if (max_cos > min_cos) else 1.0

        results: List[BaselineResult] = []
        for k in scored_keys:
            b_raw = raw_bm25.get(k, 0.0)
            c_raw = raw_cosine.get(k, 0.0)

            b_norm = (b_raw - min_bm25) / diff_bm25 if (max_bm25 > min_bm25) else (1.0 if b_raw > 0 else 0.0)
            c_norm = (c_raw - min_cos) / diff_cos if (max_cos > min_cos) else (1.0 if c_raw > 0 else 0.0)

            fused_score = alpha * b_norm + (1.0 - alpha) * c_norm
            ref, text = self._slots[k]

            results.append(
                BaselineResult(
                    slot_key=k,
                    score=float(fused_score),
                    bm25_score=float(b_raw),
                    cosine_score=float(c_raw),
                    text_content=text,
                    ref=ref,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    @property
    def total_slots(self) -> int:
        return len(self._slots)
