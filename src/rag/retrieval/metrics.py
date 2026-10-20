"""
src/rag/retrieval/metrics.py
----------------------------
Standard Information Retrieval (IR) Evaluation Metrics for DCLC-RAG (Phase 7E).

Provides robust, mathematically verified implementations of:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@K)
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Compute Precision@K: fraction of retrieved items in top-K that are relevant.

    Parameters
    ----------
    retrieved : Sequence[str]
        Ranked list of retrieved chunk/document identifiers.
    relevant  : Set[str]
        Set of ground-truth relevant chunk/document identifiers.
    k         : int
        Cutoff rank (1-indexed).

    Returns
    -------
    float in [0.0, 1.0]. Returns 0.0 if k <= 0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / float(k)


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Compute Recall@K: fraction of relevant items retrieved in top-K.

    Parameters
    ----------
    retrieved : Sequence[str]
        Ranked list of retrieved chunk/document identifiers.
    relevant  : Set[str]
        Set of ground-truth relevant chunk/document identifiers.
    k         : int
        Cutoff rank (1-indexed).

    Returns
    -------
    float in [0.0, 1.0]. Returns 0.0 if relevant set is empty or k <= 0.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / float(len(relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """
    Compute Reciprocal Rank (RR): reciprocal of the 1-based rank of the first relevant item.

    Parameters
    ----------
    retrieved : Sequence[str]
        Ranked list of retrieved chunk/document identifiers.
    relevant  : Set[str]
        Set of ground-truth relevant chunk/document identifiers.

    Returns
    -------
    float in [0.0, 1.0]. Returns 0.0 if no relevant item is found.
    """
    if not relevant:
        return 0.0
    for rank_idx, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / float(rank_idx)
    return 0.0


def dcg_at_k(retrieved: Sequence[str], graded_relevance: Dict[str, float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain (DCG@K) using standard logarithmic discounting:
        DCG@K = sum_{i=1}^K (2^{r_i} - 1) / log2(i + 1)
    where r_i is the relevance grade of the item at 1-based rank i.
    """
    if k <= 0:
        return 0.0
    dcg = 0.0
    for rank_idx, item in enumerate(retrieved[:k], start=1):
        rel = graded_relevance.get(item, 0.0)
        if rel > 0.0:
            dcg += (2.0 ** rel - 1.0) / math.log2(rank_idx + 1.0)
    return dcg


def ndcg_at_k(retrieved: Sequence[str], graded_relevance: Dict[str, float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (nDCG@K):
        nDCG@K = DCG@K / IDCG@K
    where IDCG@K is the maximum possible DCG@K with ideal sorting of all known relevant items.

    Returns
    -------
    float in [0.0, 1.0]. Returns 0.0 if IDCG is 0.0.
    """
    if k <= 0 or not graded_relevance:
        return 0.0

    actual_dcg = dcg_at_k(retrieved, graded_relevance, k)

    # Sort all known relevant grades in descending order for ideal DCG
    ideal_grades = sorted([g for g in graded_relevance.values() if g > 0.0], reverse=True)
    if not ideal_grades:
        return 0.0

    idcg = 0.0
    for rank_idx, rel in enumerate(ideal_grades[:k], start=1):
        idcg += (2.0 ** rel - 1.0) / math.log2(rank_idx + 1.0)

    if idcg <= 0.0:
        return 0.0

    return actual_dcg / idcg
