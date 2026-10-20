"""
tests/test_rag_phase7e_metrics.py
---------------------------------
Unit tests verifying IR evaluation metric implementations against
manually computed mathematical benchmark values.
"""

from __future__ import annotations

import math
import pytest

from src.rag.retrieval.metrics import (
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k_hand_calculated():
    retrieved = ["docA", "docB", "docC", "docD", "docE"]
    relevant = {"docB", "docD"}

    # k = 1: docA not relevant -> 0/1 = 0.0
    assert precision_at_k(retrieved, relevant, k=1) == 0.0

    # k = 2: docB relevant -> 1/2 = 0.5
    assert precision_at_k(retrieved, relevant, k=2) == 0.5

    # k = 3: docB relevant -> 1/3
    assert abs(precision_at_k(retrieved, relevant, k=3) - (1.0 / 3.0)) < 1e-6

    # k = 5: docB, docD relevant -> 2/5 = 0.4
    assert precision_at_k(retrieved, relevant, k=5) == 0.4


def test_recall_at_k_hand_calculated():
    retrieved = ["docA", "docB", "docC", "docD", "docE"]
    relevant = {"docB", "docD"}

    # k = 1: 0 hits / 2 relevant = 0.0
    assert recall_at_k(retrieved, relevant, k=1) == 0.0

    # k = 2: 1 hit / 2 relevant = 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 0.5

    # k = 5: 2 hits / 2 relevant = 1.0
    assert recall_at_k(retrieved, relevant, k=5) == 1.0


def test_reciprocal_rank_hand_calculated():
    retrieved = ["docA", "docB", "docC"]
    relevant = {"docB"}
    # docB is at 1-based rank 2 -> RR = 1/2 = 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5

    # When first item is relevant -> RR = 1.0
    assert reciprocal_rank(["docB", "docA"], relevant) == 1.0

    # When no relevant item is found -> RR = 0.0
    assert reciprocal_rank(["docA", "docC"], relevant) == 0.0


def test_ndcg_at_k_hand_calculated():
    retrieved = ["docA", "docB", "docC", "docD", "docE"]
    graded_relevance = {"docB": 3.0, "docD": 2.0}

    # Hand calculation:
    # rank 1: docA (rel 0) -> 0
    # rank 2: docB (rel 3) -> (2^3 - 1) / log2(3) = 7 / 1.584962500721156 = 4.41672904
    # rank 3: docC (rel 0) -> 0
    # rank 4: docD (rel 2) -> (2^2 - 1) / log2(5) = 3 / 2.321928094887362 = 1.29202967
    # rank 5: docE (rel 0) -> 0
    # DCG@5 = 4.41672904 + 1.29202967 = 5.70875871
    #
    # IDCG@5 (ideal ranking: docB at rank 1, docD at rank 2):
    # rank 1: (2^3 - 1) / log2(2) = 7.0
    # rank 2: (2^2 - 1) / log2(3) = 3 / 1.584962500721156 = 1.89278945
    # IDCG@5 = 7.0 + 1.89278945 = 8.89278945
    #
    # Exact calculation using math.log2:
    dcg = (7.0 / math.log2(3)) + (3.0 / math.log2(5))
    idcg = 7.0 + (3.0 / math.log2(3))
    expected_ndcg = dcg / idcg
    computed_ndcg = ndcg_at_k(retrieved, graded_relevance, k=5)

    assert abs(computed_ndcg - expected_ndcg) < 1e-9


def test_perfect_ndcg_is_one():
    ideal = ["docB", "docD", "docA"]
    graded = {"docB": 3.0, "docD": 2.0}
    assert abs(ndcg_at_k(ideal, graded, k=3) - 1.0) < 1e-6


def test_edge_cases():
    assert precision_at_k([], {"a"}, k=5) == 0.0
    assert recall_at_k([], {"a"}, k=5) == 0.0
    assert recall_at_k(["a"], set(), k=5) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0
    assert ndcg_at_k([], {"a": 1.0}, k=5) == 0.0
    assert ndcg_at_k(["a"], {}, k=5) == 0.0
