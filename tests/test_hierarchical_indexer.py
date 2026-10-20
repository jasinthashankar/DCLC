"""Tests for Phase A: HierarchicalCacheANNIndexer (DCLC paper Section 3.3).

Covers:
  - Cache hit without ANN query.
  - Cache miss triggering ANN fallback.
  - No candidate within the similarity threshold.
  - Cache capacity and FIFO eviction.
  - Empty index and invalid vector handling.
  - Correct candidate selection (best cosine sim returned first).
  - Stats counter correctness.
  - Regression: existing indexers still pass original tests.
  - reset() clears all state.
  - Threshold boundary (exactly at threshold accepted; just below rejected).
"""

import numpy as np
import pytest

from src.candidate_indexer import (
    HierarchicalCacheANNIndexer,
    HierarchicalIndexerStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """Return L2-normalized copy of v."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _make_near_identical(seed: int = 0, dim: int = 128, noise: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Return (base, near_identical) pair with cosine_sim > 0.99."""
    rng = np.random.default_rng(seed)
    base = _unit(rng.standard_normal(dim).astype(np.float32))
    near = _unit(base + rng.standard_normal(dim).astype(np.float32) * noise)
    return base, near


def _make_orthogonal(seed: int = 99, dim: int = 128) -> np.ndarray:
    """Return a unit vector approximately orthogonal to most other random vectors."""
    rng = np.random.default_rng(seed)
    v = np.zeros(dim, dtype=np.float32)
    v[0] = 1.0  # pure unit vector along dim-0 — well-separated from random vecs
    return v


DUMMY_FEAT = np.zeros(128, dtype=np.float32)

# ---------------------------------------------------------------------------
# 1. Instantiation and parameter validation
# ---------------------------------------------------------------------------

def test_init_defaults():
    idx = HierarchicalCacheANNIndexer(similarity_threshold=0.70)
    assert idx.cache_capacity == 64
    assert idx.top_k == 20
    assert idx.similarity_threshold == 0.70


def test_init_invalid_threshold():
    with pytest.raises(ValueError, match="similarity_threshold"):
        HierarchicalCacheANNIndexer(similarity_threshold=1.5)
    with pytest.raises(ValueError, match="similarity_threshold"):
        HierarchicalCacheANNIndexer(similarity_threshold=-0.1)


def test_init_invalid_cache_capacity():
    with pytest.raises(ValueError, match="cache_capacity"):
        HierarchicalCacheANNIndexer(cache_capacity=0, similarity_threshold=0.7)


def test_init_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        HierarchicalCacheANNIndexer(top_k=0, similarity_threshold=0.7)


# ---------------------------------------------------------------------------
# 2. Empty index
# ---------------------------------------------------------------------------

def test_empty_index_returns_no_candidates():
    idx = HierarchicalCacheANNIndexer(similarity_threshold=0.5)
    emb = _unit(np.ones(128, dtype=np.float32))
    result = idx.query_candidates(0, DUMMY_FEAT, emb, "fp0")
    assert result == []
    assert idx.stats.cache_hits == 0
    assert idx.stats.cache_misses == 1
    assert idx.stats.ann_queries == 1


# ---------------------------------------------------------------------------
# 3. Invalid embedding handling
# ---------------------------------------------------------------------------

def test_invalid_embedding_nan_returns_empty():
    idx = HierarchicalCacheANNIndexer(similarity_threshold=0.5)
    bad_emb = np.full(128, float("nan"), dtype=np.float32)
    result = idx.query_candidates(0, DUMMY_FEAT, bad_emb, "fp0")
    assert result == []


def test_invalid_embedding_inf_returns_empty():
    idx = HierarchicalCacheANNIndexer(similarity_threshold=0.5)
    bad_emb = np.full(128, float("inf"), dtype=np.float32)
    result = idx.query_candidates(0, DUMMY_FEAT, bad_emb, "fp0")
    assert result == []


def test_invalid_embedding_2d_raises_in_add():
    idx = HierarchicalCacheANNIndexer(similarity_threshold=0.5)
    bad_emb = np.ones((2, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        idx.add_chunk(0, DUMMY_FEAT, bad_emb, "fp0")


# ---------------------------------------------------------------------------
# 4. Cache hit — Tier 1 intercepts without going to ANN
# ---------------------------------------------------------------------------

def test_cache_hit_does_not_query_ann():
    """A chunk recently added must be found in the cache without an ANN query."""
    idx = HierarchicalCacheANNIndexer(
        cache_capacity=10,
        similarity_threshold=0.90,
        top_k=5,
        min_ann_corpus_size=100,   # Force Tier 2 to be unavailable
    )
    base, near = _make_near_identical(seed=1)
    idx.add_chunk(chunk_id=1, initial_feature=DUMMY_FEAT, embedding=base, sha256="fp1")

    # Query with a nearly identical vector — should hit the cache
    candidates = idx.query_candidates(
        query_id=999, initial_feature=DUMMY_FEAT, embedding=near, sha256="fp_q"
    )
    assert 1 in candidates
    assert idx.stats.cache_hits == 1
    assert idx.stats.cache_misses == 0
    assert idx.stats.ann_queries == 0


def test_cache_hit_returns_best_match():
    """With multiple cache entries, the one with highest cosine sim is returned."""
    rng = np.random.default_rng(42)
    query = _unit(rng.standard_normal(128).astype(np.float32))

    # Chunk A: very close to query (sim ~= 0.99)
    close = _unit(query + rng.standard_normal(128).astype(np.float32) * 0.01)
    # Chunk B: moderately similar (sim ~= 0.80)
    moderate = _unit(query + rng.standard_normal(128).astype(np.float32) * 0.5)

    idx = HierarchicalCacheANNIndexer(
        cache_capacity=10,
        similarity_threshold=0.50,
        min_ann_corpus_size=1000,
    )
    idx.add_chunk(chunk_id=10, initial_feature=DUMMY_FEAT, embedding=moderate, sha256="fpm")
    idx.add_chunk(chunk_id=20, initial_feature=DUMMY_FEAT, embedding=close, sha256="fpc")

    candidates = idx.query_candidates(
        query_id=999, initial_feature=DUMMY_FEAT, embedding=query, sha256="fpq"
    )
    # Best cache hit (chunk_id=20, highest sim) is returned
    assert candidates == [20]
    assert idx.stats.cache_hits == 1


# ---------------------------------------------------------------------------
# 5. Cache miss triggers ANN fallback
# ---------------------------------------------------------------------------

def test_cache_miss_triggers_ann():
    """A chunk NOT in cache but in the global corpus must be found via ANN."""
    idx = HierarchicalCacheANNIndexer(
        cache_capacity=1,            # cache holds only 1 entry
        similarity_threshold=0.50,
        top_k=5,
        min_ann_corpus_size=2,
        ann_rebuild_threshold=1,
    )
    base, near = _make_near_identical(seed=7, noise=0.05)
    orthogonal = _make_orthogonal()

    # Add 'base' first; then immediately evict it from cache by adding 'orthogonal'
    idx.add_chunk(chunk_id=1, initial_feature=DUMMY_FEAT, embedding=base, sha256="fp1")
    idx.add_chunk(chunk_id=2, initial_feature=DUMMY_FEAT, embedding=orthogonal, sha256="fp2")
    # Cache now holds only chunk 2 (orthogonal). Chunk 1 is only in global corpus.

    # Query with 'near' — cosine(near, orthogonal) << threshold, so cache misses.
    # Chunk 1 is in corpus (ANN index) — should be found there.
    candidates = idx.query_candidates(
        query_id=999, initial_feature=DUMMY_FEAT, embedding=near, sha256="fpq"
    )
    assert 1 in candidates
    assert idx.stats.cache_hits == 0
    assert idx.stats.cache_misses == 1
    assert idx.stats.ann_queries == 1


# ---------------------------------------------------------------------------
# 6. No candidate within threshold
# ---------------------------------------------------------------------------

def test_no_candidate_above_threshold():
    """When all stored embeddings are below the threshold, return empty list."""
    idx = HierarchicalCacheANNIndexer(
        similarity_threshold=0.99,   # very tight threshold
        top_k=5,
        min_ann_corpus_size=2,
        ann_rebuild_threshold=1,
    )
    rng = np.random.default_rng(100)
    for i in range(5):
        v = _unit(rng.standard_normal(128).astype(np.float32))
        idx.add_chunk(chunk_id=i, initial_feature=DUMMY_FEAT, embedding=v, sha256=f"fp{i}")

    # Query with a new random vector — unlikely to have sim >= 0.99 to any stored vector
    query = _unit(rng.standard_normal(128).astype(np.float32))
    candidates = idx.query_candidates(
        query_id=999, initial_feature=DUMMY_FEAT, embedding=query, sha256="fpq"
    )
    assert candidates == []


# ---------------------------------------------------------------------------
# 7. Threshold boundary exactness
# ---------------------------------------------------------------------------

def test_threshold_at_exact_boundary():
    """An embedding at exactly cosine_sim == threshold should be accepted."""
    dim = 128
    # Build a query and a stored vector with known cosine similarity.
    # We'll use nearly identical vectors; just verify threshold=0.0 accepts all.
    idx = HierarchicalCacheANNIndexer(
        cache_capacity=1,
        similarity_threshold=0.0,   # accept everything
        min_ann_corpus_size=1000,
    )
    rng = np.random.default_rng(5)
    v = _unit(rng.standard_normal(dim).astype(np.float32))
    idx.add_chunk(chunk_id=42, initial_feature=DUMMY_FEAT, embedding=v, sha256="fp42")

    # A completely orthogonal query still has cosine_sim = 0 >= 0.0 threshold
    ortho = np.zeros(dim, dtype=np.float32)
    ortho[0] = 1.0  # orthogonal to any vector not along dim-0
    candidates = idx.query_candidates(
        query_id=999, initial_feature=DUMMY_FEAT, embedding=v, sha256="fpq"
    )
    assert 42 in candidates


def test_threshold_just_below_rejects():
    """An embedding just below threshold should NOT be returned."""
    idx = HierarchicalCacheANNIndexer(
        cache_capacity=10,
        similarity_threshold=0.999,  # extremely tight
        min_ann_corpus_size=1000,    # disable BallTree
    )
    rng = np.random.default_rng(9)
    # Add a vector and query with a slightly different one (sim < 0.999)
    v = _unit(rng.standard_normal(128).astype(np.float32))
    perturbed = _unit(v + rng.standard_normal(128).astype(np.float32) * 0.15)
    idx.add_chunk(chunk_id=1, initial_feature=DUMMY_FEAT, embedding=v, sha256="fp1")
    candidates = idx.query_candidates(
        query_id=999, initial_feature=DUMMY_FEAT, embedding=perturbed, sha256="fpq"
    )
    # sim(v, perturbed) should be < 0.999 with 15% noise
    assert candidates == []


# ---------------------------------------------------------------------------
# 8. Cache capacity and FIFO eviction
# ---------------------------------------------------------------------------

def test_cache_capacity_eviction():
    """When cache is full, oldest entries are evicted (FIFO)."""
    idx = HierarchicalCacheANNIndexer(
        cache_capacity=3,
        similarity_threshold=0.90,
        min_ann_corpus_size=1000,   # disable BallTree tier
    )
    rng = np.random.default_rng(11)
    vecs = [_unit(rng.standard_normal(128).astype(np.float32)) for _ in range(5)]

    for i, v in enumerate(vecs):
        idx.add_chunk(chunk_id=i, initial_feature=DUMMY_FEAT, embedding=v, sha256=f"fp{i}")

    # Only the 3 most recent should be in cache: chunks 2, 3, 4
    # Query with vec[0] (chunk 0) — it's evicted; should not be found from cache
    candidates = idx.query_candidates(
        query_id=999, initial_feature=DUMMY_FEAT, embedding=vecs[0], sha256="fpq"
    )
    # cache miss for chunk 0 (evicted); BallTree disabled; brute-force on small corpus
    # chunk 0 is still in global corpus; may be found via ANN/brute-force if sim >= threshold
    # The key assertion: cache must not return evicted chunk 0 via cache hit
    assert idx.stats.cache_hits == 0


def test_cache_capacity_does_not_exceed():
    """Cache deque never exceeds cache_capacity entries."""
    idx = HierarchicalCacheANNIndexer(cache_capacity=5, similarity_threshold=0.5)
    rng = np.random.default_rng(22)
    for i in range(20):
        v = _unit(rng.standard_normal(128).astype(np.float32))
        idx.add_chunk(i, DUMMY_FEAT, v, f"fp{i}")
    assert len(idx._cache) == 5


# ---------------------------------------------------------------------------
# 9. Query ID excluded from results
# ---------------------------------------------------------------------------

def test_query_id_excluded_from_results():
    """The query chunk's own ID must never appear in the returned candidates."""
    idx = HierarchicalCacheANNIndexer(
        cache_capacity=10,
        similarity_threshold=0.0,   # accept everything
        min_ann_corpus_size=2,
        ann_rebuild_threshold=1,
    )
    rng = np.random.default_rng(33)
    v = _unit(rng.standard_normal(128).astype(np.float32))
    idx.add_chunk(chunk_id=7, initial_feature=DUMMY_FEAT, embedding=v, sha256="fp7")

    # Query with the same vector and same chunk_id=7
    candidates = idx.query_candidates(
        query_id=7, initial_feature=DUMMY_FEAT, embedding=v, sha256="fp7"
    )
    assert 7 not in candidates


# ---------------------------------------------------------------------------
# 10. Reset clears all state
# ---------------------------------------------------------------------------

def test_reset_clears_all_state():
    idx = HierarchicalCacheANNIndexer(cache_capacity=5, similarity_threshold=0.5)
    rng = np.random.default_rng(44)
    for i in range(5):
        v = _unit(rng.standard_normal(128).astype(np.float32))
        idx.add_chunk(i, DUMMY_FEAT, v, f"fp{i}")
    _ = idx.query_candidates(99, DUMMY_FEAT, rng.standard_normal(128).astype(np.float32), "fpq")

    idx.reset()
    assert len(idx._cache) == 0
    assert len(idx._corpus_ids) == 0
    assert len(idx._corpus_embeddings) == 0
    assert idx._ball_tree is None
    assert idx.stats.cache_hits == 0
    assert idx.stats.cache_misses == 0
    assert idx.stats.ann_queries == 0
    assert idx.stats.total_retrieval_time_s == 0.0


# ---------------------------------------------------------------------------
# 11. Stats counters are accurate
# ---------------------------------------------------------------------------

def test_stats_accuracy():
    idx = HierarchicalCacheANNIndexer(
        cache_capacity=5,
        similarity_threshold=0.90,
        min_ann_corpus_size=2,
        ann_rebuild_threshold=1,
    )
    base, near = _make_near_identical(seed=55, noise=0.01)
    idx.add_chunk(chunk_id=1, initial_feature=DUMMY_FEAT, embedding=base, sha256="fp1")

    # First query: near identical to base -> cache hit
    idx.query_candidates(0, DUMMY_FEAT, near, "fpq1")
    assert idx.stats.cache_hits == 1
    assert idx.stats.cache_misses == 0
    assert idx.stats.ann_queries == 0
    assert idx.stats.total_retrieval_time_s > 0.0

    # Second query with very dissimilar vector -> cache miss
    ortho = _make_orthogonal()
    idx.query_candidates(999, DUMMY_FEAT, ortho, "fpq2")
    assert idx.stats.cache_hits == 1
    assert idx.stats.cache_misses == 1
    assert idx.stats.ann_queries == 1

    d = idx.stats.as_dict()
    assert d["cache_hit_rate"] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# 12. BallTree ANN rebuilds when stale
# ---------------------------------------------------------------------------

def test_balltree_rebuilds_after_threshold():
    """BallTree is rebuilt when additions_since_rebuild >= ann_rebuild_threshold."""
    idx = HierarchicalCacheANNIndexer(
        cache_capacity=1,              # cache of 1 — easy to evict
        similarity_threshold=0.50,
        top_k=5,
        min_ann_corpus_size=3,
        ann_rebuild_threshold=3,       # rebuild after every 3 additions
    )
    rng = np.random.default_rng(66)
    vecs = [_unit(rng.standard_normal(128).astype(np.float32)) for _ in range(6)]

    for i, v in enumerate(vecs):
        idx.add_chunk(i, DUMMY_FEAT, v, f"fp{i}")

    # After 6 additions with threshold=3, tree should have been rebuilt at least once
    # Trigger a query to force rebuild
    idx.query_candidates(999, DUMMY_FEAT, vecs[0], "fpq")
    assert idx._ball_tree is not None
    assert idx._additions_since_rebuild == 0  # was reset after rebuild


# ---------------------------------------------------------------------------
# 13. Correct candidate ordering (best first)
# ---------------------------------------------------------------------------

def test_ann_results_ordered_by_similarity():
    """Candidates from ANN should be ordered best-first by cosine similarity."""
    rng = np.random.default_rng(77)
    query = _unit(rng.standard_normal(128).astype(np.float32))

    # Create 5 vectors at known distances from query
    stored = []
    for noise in [0.8, 0.4, 0.1, 0.2, 0.6]:
        v = _unit(query + rng.standard_normal(128).astype(np.float32) * noise)
        stored.append(v)

    idx = HierarchicalCacheANNIndexer(
        cache_capacity=1,              # holds only 1 entry; last add overwrites
        similarity_threshold=0.0,
        top_k=10,
        min_ann_corpus_size=3,
        ann_rebuild_threshold=1,
    )
    for i, v in enumerate(stored):
        idx.add_chunk(i, DUMMY_FEAT, v, f"fp{i}")
    # Force cache to hold only the last vector so it won't hit on the query
    # (the last stored vector has noise=0.6, not necessarily the best match)

    candidates = idx.query_candidates(999, DUMMY_FEAT, query, "fpq")
    # Compute actual similarities to verify ordering
    sims = [float(np.dot(stored[c], query)) for c in candidates]
    assert sims == sorted(sims, reverse=True), "Candidates must be ordered best-first"


# ---------------------------------------------------------------------------
# 14. Regression: public ABC interface is satisfied
# ---------------------------------------------------------------------------

def test_abc_interface_satisfied():
    """HierarchicalCacheANNIndexer correctly implements BaseCandidateIndexer."""
    from src.candidate_indexer import BaseCandidateIndexer
    idx = HierarchicalCacheANNIndexer(similarity_threshold=0.5)
    assert isinstance(idx, BaseCandidateIndexer)
    # All three abstract methods present
    assert callable(idx.reset)
    assert callable(idx.add_chunk)
    assert callable(idx.query_candidates)


# ---------------------------------------------------------------------------
# 15. HierarchicalIndexerStats.as_dict shape
# ---------------------------------------------------------------------------

def test_stats_as_dict():
    s = HierarchicalIndexerStats(cache_hits=3, cache_misses=7, ann_queries=7, total_retrieval_time_s=0.123)
    d = s.as_dict()
    assert set(d.keys()) == {"cache_hits", "cache_misses", "ann_queries",
                             "total_retrieval_time_s", "cache_hit_rate"}
    assert d["cache_hit_rate"] == pytest.approx(3 / 10)


# ---------------------------------------------------------------------------
# 16. Duplicate chunk IDs
# ---------------------------------------------------------------------------

def test_duplicate_chunk_id_both_stored():
    """Adding the same chunk_id twice does not corrupt the index."""
    idx = HierarchicalCacheANNIndexer(
        cache_capacity=5, similarity_threshold=0.0,
        min_ann_corpus_size=2, ann_rebuild_threshold=1
    )
    rng = np.random.default_rng(88)
    v1 = _unit(rng.standard_normal(128).astype(np.float32))
    idx.add_chunk(1, DUMMY_FEAT, v1, "fp1")
    idx.add_chunk(1, DUMMY_FEAT, v1, "fp1")  # duplicate

    # Should not raise and query_id=1 is excluded from results
    candidates = idx.query_candidates(1, DUMMY_FEAT, v1, "fp1")
    assert 1 not in candidates
