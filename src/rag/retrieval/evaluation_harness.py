"""
src/rag/retrieval/evaluation_harness.py
----------------------------------------
DCLC-RAG Phase 6C — Retrieval Evaluation Harness.

Computes comparative metrics between DCLC-Aware Hybrid Retrieval
and Standard Non-Deduplicated Baseline Retrieval on identical corpora,
queries, and ground-truth relevant passages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.rag.retrieval.baseline_retriever import BaselineRetriever
from src.rag.retrieval.hybrid_retriever import HybridRetriever


@dataclass
class EvalQuery:
    """
    Evaluation query with ground truth relevant passages and version scope.

    Attributes
    ----------
    query_id         : Unique query identifier.
    query_text       : The natural language search query.
    relevant_chunks  : Set of relevant chunk_ids (or unique substring fingerprints)
                       representing ground-truth answers.
    scope_doc_id     : Optional doc_id scope restriction.
    scope_version_id : Optional version_id scope restriction.
    expected_answer_keywords : Keywords expected in correctly retrieved passages.
    prohibited_keywords      : Keywords that belong strictly to another version
                               (to test version isolation).
    """
    query_id: str
    query_text: str
    relevant_chunks: Set[str] = field(default_factory=set)
    scope_doc_id: Optional[str] = None
    scope_version_id: Optional[str] = None
    expected_answer_keywords: List[str] = field(default_factory=list)
    prohibited_keywords: List[str] = field(default_factory=list)


@dataclass
class MetricReport:
    """
    Comparative metric report for one system.
    """
    system_name: str
    embedding_computations: int
    unique_vectors: int
    vector_payload_bytes: int
    embedding_build_time_seconds: float
    mean_retrieval_latency_ms: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    version_isolation_violations: int


class EvaluationHarness:
    """
    Runs evaluation queries across DCLC-aware and baseline retrievers.
    """

    def __init__(self, ks: Optional[List[int]] = None) -> None:
        self.ks = ks or [1, 3, 5]

    def evaluate_dclc(
        self,
        retriever: HybridRetriever,
        queries: List[EvalQuery],
        alpha: float = 0.5,
    ) -> MetricReport:
        """
        Evaluate DCLC-aware hybrid retriever.
        """
        p_at_k: Dict[int, List[float]] = {k: [] for k in self.ks}
        r_at_k: Dict[int, List[float]] = {k: [] for k in self.ks}
        latencies_ms: List[float] = []
        violations = 0

        for q in queries:
            t0 = time.perf_counter()
            results = retriever.retrieve(
                query=q.query_text,
                top_k=max(self.ks),
                alpha=alpha,
                doc_id=q.scope_doc_id,
                version_id=q.scope_version_id,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            # Check version isolation violations
            for res in results:
                # If prohibited keywords appear in retrieved passage, violation!
                text_lower = res.text_content.lower()
                for bad in q.prohibited_keywords:
                    if bad.lower() in text_lower:
                        violations += 1

                # If scope was set, ensure filtered_refs match
                if q.scope_version_id is not None:
                    if not any(r.version_id == q.scope_version_id for r in res.filtered_refs):
                        violations += 1

            # Precision & Recall @ K
            retrieved_chunk_ids = [r.chunk_id for r in results]
            for k in self.ks:
                top_k_ids = set(retrieved_chunk_ids[:k])
                hits = 0
                if q.relevant_chunks:
                    hits = len(top_k_ids & q.relevant_chunks)
                else:
                    # Fallback to expected keywords matching
                    for r in results[:k]:
                        if any(kw.lower() in r.text_content.lower() for kw in q.expected_answer_keywords):
                            hits += 1
                
                prec = hits / k
                total_rel = len(q.relevant_chunks) if q.relevant_chunks else len(q.expected_answer_keywords) or 1
                rec = min(1.0, hits / total_rel)
                p_at_k[k].append(prec)
                r_at_k[k].append(rec)

        mean_p = {k: float(sum(vals) / len(vals)) if vals else 0.0 for k, vals in p_at_k.items()}
        mean_r = {k: float(sum(vals) / len(vals)) if vals else 0.0 for k, vals in r_at_k.items()}
        mean_lat = float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0

        return MetricReport(
            system_name="DCLC-Aware Hybrid",
            embedding_computations=retriever.store.embedding_count,
            unique_vectors=retriever.store.embedding_count,
            vector_payload_bytes=retriever.store.storage_bytes,
            embedding_build_time_seconds=retriever.store.build_time_seconds,
            mean_retrieval_latency_ms=mean_lat,
            precision_at_k=mean_p,
            recall_at_k=mean_r,
            version_isolation_violations=violations,
        )

    def evaluate_baseline(
        self,
        retriever: BaselineRetriever,
        queries: List[EvalQuery],
        alpha: float = 0.5,
    ) -> MetricReport:
        """
        Evaluate baseline non-deduplicated retriever.
        """
        p_at_k: Dict[int, List[float]] = {k: [] for k in self.ks}
        r_at_k: Dict[int, List[float]] = {k: [] for k in self.ks}
        latencies_ms: List[float] = []
        violations = 0

        for q in queries:
            t0 = time.perf_counter()
            results = retriever.retrieve(
                query=q.query_text,
                top_k=max(self.ks),
                alpha=alpha,
                doc_id=q.scope_doc_id,
                version_id=q.scope_version_id,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            # Check version isolation violations
            for res in results:
                text_lower = res.text_content.lower()
                for bad in q.prohibited_keywords:
                    if bad.lower() in text_lower:
                        violations += 1

                if q.scope_version_id is not None:
                    if res.ref.version_id != q.scope_version_id:
                        violations += 1

            # Precision & Recall @ K
            for k in self.ks:
                hits = 0
                for r in results[:k]:
                    if any(kw.lower() in r.text_content.lower() for kw in q.expected_answer_keywords):
                        hits += 1
                
                prec = hits / k
                total_rel = len(q.expected_answer_keywords) or 1
                rec = min(1.0, hits / total_rel)
                p_at_k[k].append(prec)
                r_at_k[k].append(rec)

        mean_p = {k: float(sum(vals) / len(vals)) if vals else 0.0 for k, vals in p_at_k.items()}
        mean_r = {k: float(sum(vals) / len(vals)) if vals else 0.0 for k, vals in r_at_k.items()}
        mean_lat = float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0

        return MetricReport(
            system_name="Standard Baseline (Non-Dedup)",
            embedding_computations=retriever.store.embedding_count,
            unique_vectors=retriever.store.embedding_count,
            vector_payload_bytes=retriever.store.storage_bytes,
            embedding_build_time_seconds=retriever.store.build_time_seconds,
            mean_retrieval_latency_ms=mean_lat,
            precision_at_k=mean_p,
            recall_at_k=mean_r,
            version_isolation_violations=violations,
        )
