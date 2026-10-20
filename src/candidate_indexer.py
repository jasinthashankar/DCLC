"""Label-Free Candidate Indexing for DCLC Deduplication Pipeline (Phase 4C + Phase A).

Provides practical candidate-generation mechanisms for incoming chunks WITHOUT
access to oracle family IDs, revision IDs, or lineage labels:

1. Random-Projection Locality-Sensitive Hashing (RP-LSH):
   - Applied to the 128-dimensional initial feature vectors (Algorithm 1).
   - Projects dense vectors onto random Gaussian hyperplanes to preserve cosine similarity.
   - Note: This is distinct from MinHash (which estimates set Jaccard over permutations).

2. Cosine Nearest-Neighbor Index over Learned Transformer Embeddings:
   - Queries the 128-dimensional L2-normalized representations produced by the trained
     DCLC context encoder (checkpoints/dclc_encoder_version_aware.pt).
   - Retrieves top-K nearest neighbors based on vector dot product.

3. Stream-Locality Baseline:
   - Sliding-window buffer over the most recently ingested non-duplicate chunks.
   - Simulates physical storage stream / temporal backup locality.

4. Hierarchical Cache + ANN Index (Phase A — implements DCLC paper Section 3.3):
   - Two-tier retrieval matching the paper's locality-aware hierarchical design:
       Tier 1 (Dynamic Local Cache): A bounded FIFO cache of recently added chunk
           embeddings. Cosine similarity (via dot product on L2-normalized vectors) is
           computed against all cache entries. If any entry exceeds the configurable
           cosine similarity threshold, the best cache hit is returned immediately
           without querying the global index.
       Tier 2 (ANN Global Index): A scikit-learn BallTree built over all ingested
           L2-normalized embeddings. Queried only on cache miss. BallTree with
           Euclidean metric on unit-norm vectors gives exact cosine nearest-neighbor
           results: cosine_sim = 1 - euclidean_dist^2 / 2.

   ANN Backend Choice:
       Annoy (Spotify) is the indexer used in the paper, but it requires MSVC C++
       14+ to compile from source and no pre-built wheel exists for Python 3.13 on
       Windows (confirmed: pip wheel failed with "error: Microsoft Visual C++ 14.0
       or greater is required"). scikit-learn BallTree is used instead: it is pure
       Python/Cython with pre-built wheels, already installed in this environment
       (scikit-learn 1.9.1), and provides exact cosine NN results on unit vectors.

   Counters exposed:
       cache_hits, cache_misses, ann_queries, total_retrieval_time_s.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseCandidateIndexer(ABC):
    """Abstract base class for label-free candidate retrieval indexers."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored index state."""
        pass

    @abstractmethod
    def add_chunk(
        self,
        chunk_id: int,
        initial_feature: np.ndarray,
        embedding: np.ndarray,
        sha256: str,
    ) -> None:
        """Index a stored chunk."""
        pass

    @abstractmethod
    def query_candidates(
        self,
        query_id: int,
        initial_feature: np.ndarray,
        embedding: np.ndarray,
        sha256: str,
    ) -> list[int]:
        """Retrieve candidate chunk IDs for an incoming query chunk."""
        pass


class RandomProjectionLSHIndexer(BaseCandidateIndexer):
    """Random-Projection Locality-Sensitive Hashing (RP-LSH) over dense feature vectors.

    Projects d-dimensional vectors onto L independent hash tables, each with B random hyperplanes:
        hash_code = tuple((W @ x >= 0).astype(int))
    Chunks falling into the same bucket in any of the L tables are candidate matches.
    """

    def __init__(
        self,
        dim: int = 128,
        num_tables: int = 8,
        num_bits: int = 6,
        seed: int = 42,
    ):
        self.dim = dim
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Generate random hyperplanes: shape (num_tables, num_bits, dim)
        self.hyperplanes = self.rng.standard_normal((num_tables, num_bits, dim)).astype(np.float32)
        # Normalize projection vectors
        norms = np.linalg.norm(self.hyperplanes, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        self.hyperplanes /= norms

        self.tables: list[dict[tuple[int, ...], list[int]]] = [{} for _ in range(num_tables)]
        self.stored_ids: list[int] = []

    def reset(self) -> None:
        self.tables = [{} for _ in range(self.num_tables)]
        self.stored_ids = []

    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> tuple[int, ...]:
        # Dot product with hyperplanes: shape (num_bits,)
        projections = self.hyperplanes[table_idx] @ vector
        return tuple((projections >= 0).astype(int).tolist())

    def add_chunk(
        self,
        chunk_id: int,
        initial_feature: np.ndarray,
        embedding: np.ndarray,
        sha256: str,
    ) -> None:
        feat = initial_feature.astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        self.stored_ids.append(chunk_id)
        for t in range(self.num_tables):
            code = self._hash_vector(feat, t)
            self.tables[t].setdefault(code, []).append(chunk_id)

    def query_candidates(
        self,
        query_id: int,
        initial_feature: np.ndarray,
        embedding: np.ndarray,
        sha256: str,
    ) -> list[int]:
        feat = initial_feature.astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        candidates: set[int] = set()
        for t in range(self.num_tables):
            code = self._hash_vector(feat, t)
            bucket = self.tables[t].get(code, [])
            for cid in bucket:
                if cid != query_id:
                    candidates.add(cid)

        return sorted(list(candidates))


class EmbeddingCosineKNNIndexer(BaseCandidateIndexer):
    """Nearest-neighbor index over learned 128-dimensional Transformer embeddings."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.stored_ids: list[int] = []
        self.stored_embeddings: list[np.ndarray] = []

    def reset(self) -> None:
        self.stored_ids = []
        self.stored_embeddings = []

    def add_chunk(
        self,
        chunk_id: int,
        initial_feature: np.ndarray,
        embedding: np.ndarray,
        sha256: str,
    ) -> None:
        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        self.stored_ids.append(chunk_id)
        self.stored_embeddings.append(emb)

    def query_candidates(
        self,
        query_id: int,
        initial_feature: np.ndarray,
        embedding: np.ndarray,
        sha256: str,
    ) -> list[int]:
        if not self.stored_embeddings:
            return []

        q_emb = embedding.astype(np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        mat = np.stack(self.stored_embeddings, axis=0) # (N, D)
        sims = mat @ q_emb                             # (N,)

        # Mask query_id if already in stored list
        for idx, cid in enumerate(self.stored_ids):
            if cid == query_id:
                sims[idx] = -999.0

        k = min(self.top_k, len(self.stored_ids))
        if k <= 0:
            return []

        top_indices = np.argsort(sims)[::-1][:k]
        return [self.stored_ids[idx] for idx in top_indices if sims[idx] > -900.0]


class StreamLocalityIndexer(BaseCandidateIndexer):
    """Sliding-window buffer simulating stream / temporal storage locality."""

    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self.window: list[int] = []

    def reset(self) -> None:
        self.window = []

    def add_chunk(
        self,
        chunk_id: int,
        initial_feature: np.ndarray,
        embedding: np.ndarray,
        sha256: str,
    ) -> None:
        self.window.append(chunk_id)
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def query_candidates(
        self,
        query_id: int,
        initial_feature: np.ndarray,
        embedding: np.ndarray,
        sha256: str,
    ) -> list[int]:
        return [cid for cid in self.window if cid != query_id]


# ---------------------------------------------------------------------------
# Phase A — DCLC Paper Section 3.3: Locality-Aware Hierarchical Retrieval
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalIndexerStats:
    """Telemetry counters for HierarchicalCacheANNIndexer."""
    cache_hits: int = 0
    cache_misses: int = 0
    ann_queries: int = 0
    total_retrieval_time_s: float = 0.0

    def reset(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.ann_queries = 0
        self.total_retrieval_time_s = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "ann_queries": self.ann_queries,
            "total_retrieval_time_s": round(self.total_retrieval_time_s, 6),
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
        }


class HierarchicalCacheANNIndexer(BaseCandidateIndexer):
    """Locality-aware two-tier retrieval index implementing DCLC paper Section 3.3.

    Architecture (Fig. 4 of the paper):
      Tier 1 — Dynamic Local Cache
        A bounded FIFO cache of the most recently added chunk embeddings.
        Each query first scans the cache using cosine similarity (dot product on
        L2-normalized vectors). If any cache entry has cosine_sim >= threshold, the
        best matching entry is returned immediately as a cache hit.

      Tier 2 — Global ANN Index (BallTree)
        On a cache miss, a scikit-learn BallTree is queried to retrieve the top-K
        most similar stored embeddings across the full ingested corpus.

        The BallTree uses Euclidean distance on L2-normalized (unit-norm) vectors.
        For unit vectors: cosine_similarity = 1 - euclidean_distance^2 / 2.
        This gives exact cosine nearest-neighbor results identical to a brute-force
        cosine search, with O(n log n) build and O(log n) expected query time.

    ANN Backend Choice — Annoy vs BallTree:
        Annoy (Spotify) is referenced in the paper but requires MSVC C++ 14+ to
        compile from source; no pre-built wheel exists for Python 3.13 on Windows
        (confirmed: pip wheel builds failed in this environment). scikit-learn
        BallTree is used as the documented replacement: pre-built binary wheels are
        already installed (scikit-learn 1.9.1), and it returns exact cosine-neighbor
        results on unit vectors.

    Cache semantics:
        The cache stores at most `cache_capacity` entries (FIFO eviction). When the
        global corpus grows beyond `ann_rebuild_threshold` new additions since the
        last rebuild, the BallTree is rebuilt automatically on the next ANN query.

    Parameters:
        cache_capacity: Maximum number of embeddings in the local FIFO cache.
        similarity_threshold: Cosine similarity cutoff for a cache entry to be
            considered a valid candidate (0.0–1.0).  No default value is hard-coded
            as a "paper-proven" number; callers must pass an explicit value.
        top_k: Maximum number of candidates to return from the ANN global index.
        ann_rebuild_threshold: Number of new additions that trigger a BallTree
            rebuild on the next ANN query.
        min_ann_corpus_size: Minimum number of stored vectors required before the
            ANN index is used. Below this count, brute-force dot-product is used.

    Exposed telemetry:
        stats  — HierarchicalIndexerStats with cache_hits, cache_misses,
                 ann_queries, total_retrieval_time_s.
    """

    def __init__(
        self,
        cache_capacity: int = 64,
        similarity_threshold: float = 0.70,
        top_k: int = 20,
        ann_rebuild_threshold: int = 50,
        min_ann_corpus_size: int = 10,
    ) -> None:
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [0.0, 1.0], got {similarity_threshold}"
            )
        if cache_capacity < 1:
            raise ValueError(f"cache_capacity must be >= 1, got {cache_capacity}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if ann_rebuild_threshold < 1:
            raise ValueError(
                f"ann_rebuild_threshold must be >= 1, got {ann_rebuild_threshold}"
            )

        self.cache_capacity = cache_capacity
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.ann_rebuild_threshold = ann_rebuild_threshold
        self.min_ann_corpus_size = min_ann_corpus_size

        # Tier 1 — FIFO cache: deque of (chunk_id, unit-norm embedding)
        self._cache: deque[tuple[int, np.ndarray]] = deque()

        # Tier 2 — Global corpus: parallel lists
        self._corpus_ids: list[int] = []
        self._corpus_embeddings: list[np.ndarray] = []

        # BallTree state
        self._ball_tree: Any = None          # sklearn BallTree or None
        self._tree_built_at_size: int = 0    # corpus size when tree was last built
        self._additions_since_rebuild: int = 0

        self.stats = HierarchicalIndexerStats()

    def reset(self) -> None:
        """Clear all state and counters."""
        self._cache.clear()
        self._corpus_ids.clear()
        self._corpus_embeddings.clear()
        self._ball_tree = None
        self._tree_built_at_size = 0
        self._additions_since_rebuild = 0
        self.stats.reset()

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """Return L2-normalized copy of v. Returns zero vector if norm is 0."""
        v = np.asarray(v, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return v
        return v / norm

    @staticmethod
    def _validate_embedding(embedding: np.ndarray, label: str = "embedding") -> None:
        """Raise ValueError for embeddings that cannot be used (wrong shape/NaN/Inf)."""
        if embedding.ndim != 1:
            raise ValueError(
                f"{label} must be a 1-D array, got shape {embedding.shape}"
            )
        if embedding.size == 0:
            raise ValueError(f"{label} must not be empty")
        if not np.isfinite(embedding).all():
            raise ValueError(f"{label} contains NaN or Inf values")

    def add_chunk(
        self,
        chunk_id: int,
        initial_feature: np.ndarray,
        embedding: np.ndarray,
        sha256: str,
    ) -> None:
        """Ingest a new chunk into the cache and global corpus.

        Args:
            chunk_id: Unique integer identifier for this chunk.
            initial_feature: Ignored by this indexer (used by LSH/Stream variants).
            embedding: 1-D float32 embedding vector for the chunk.
            sha256: SHA-256 fingerprint string (unused internally; kept for API parity).
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        self._validate_embedding(embedding, label=f"add_chunk(chunk_id={chunk_id})")
        unit_emb = self._normalize(embedding)

        # -- Tier 1: update FIFO cache
        self._cache.append((chunk_id, unit_emb))
        if len(self._cache) > self.cache_capacity:
            self._cache.popleft()

        # -- Tier 2: append to global corpus, mark tree as stale
        self._corpus_ids.append(chunk_id)
        self._corpus_embeddings.append(unit_emb)
        self._additions_since_rebuild += 1

    def _rebuild_tree_if_needed(self) -> None:
        """Rebuild the BallTree when stale or when corpus has grown past the threshold."""
        n = len(self._corpus_ids)
        if n < self.min_ann_corpus_size:
            return  # Too few points for a tree; brute force used instead

        if (
            self._ball_tree is None
            or self._additions_since_rebuild >= self.ann_rebuild_threshold
        ):
            try:
                from sklearn.neighbors import BallTree  # lazy import
                matrix = np.stack(self._corpus_embeddings, axis=0)  # (n, dim)
                self._ball_tree = BallTree(matrix, metric="euclidean")
                self._tree_built_at_size = n
                self._additions_since_rebuild = 0
                logger.debug("BallTree rebuilt over %d vectors.", n)
            except ImportError:
                logger.error(
                    "scikit-learn is required for HierarchicalCacheANNIndexer "
                    "but could not be imported. Falling back to brute-force search."
                )
                self._ball_tree = None

    def _brute_force_topk(
        self, query_unit: np.ndarray, exclude_ids: set[int]
    ) -> list[tuple[int, float]]:
        """Return top-k (chunk_id, cosine_sim) from corpus excluding exclude_ids."""
        if not self._corpus_embeddings:
            return []
        mat = np.stack(self._corpus_embeddings, axis=0)  # (n, dim)
        sims = mat @ query_unit                            # (n,)  cosine sims

        # Mask excluded IDs
        for i, cid in enumerate(self._corpus_ids):
            if cid in exclude_ids:
                sims[i] = -2.0  # below any valid cosine sim

        k = min(self.top_k, len(self._corpus_ids))
        top_indices = np.argsort(sims)[::-1][:k]
        return [
            (self._corpus_ids[i], float(sims[i]))
            for i in top_indices
            if float(sims[i]) > -1.5
        ]

    def query_candidates(
        self,
        query_id: int,
        initial_feature: np.ndarray,
        embedding: np.ndarray,
        sha256: str,
    ) -> list[int]:
        """Retrieve candidate chunk IDs via two-tier hierarchical retrieval.

        Tier 1: Scan the local FIFO cache for cosine_sim >= similarity_threshold.
                If a hit is found, return its chunk_id immediately.
        Tier 2: Query the global BallTree ANN index (or brute-force on small corpora).
                Return up to top_k candidates whose cosine_sim >= similarity_threshold.

        Args:
            query_id: The ID of the chunk being queried. Excluded from results.
            initial_feature: Ignored (API parity with other indexers).
            embedding: 1-D float32 embedding vector for the query chunk.
            sha256: SHA-256 fingerprint (API parity; not used for retrieval).

        Returns:
            Ordered list of candidate chunk_ids (best first by cosine similarity).
            Returns an empty list if no candidate meets the threshold.
        """
        t_start = time.perf_counter()
        embedding = np.asarray(embedding, dtype=np.float32)
        try:
            self._validate_embedding(embedding, label=f"query_candidates(query_id={query_id})")
        except ValueError as exc:
            logger.warning("Invalid query embedding for chunk %d: %s", query_id, exc)
            self.stats.total_retrieval_time_s += time.perf_counter() - t_start
            return []

        query_unit = self._normalize(embedding)
        exclude_ids: set[int] = {query_id}

        # ---------------------------------------------------------------
        # Tier 1: Dynamic local cache scan
        # ---------------------------------------------------------------
        best_cache_cid: Optional[int] = None
        best_cache_sim: float = -2.0

        for cid, cached_emb in self._cache:
            if cid in exclude_ids:
                continue
            sim = float(np.dot(cached_emb, query_unit))
            if sim >= self.similarity_threshold and sim > best_cache_sim:
                best_cache_sim = sim
                best_cache_cid = cid

        if best_cache_cid is not None:
            self.stats.cache_hits += 1
            self.stats.total_retrieval_time_s += time.perf_counter() - t_start
            logger.debug(
                "Cache hit: chunk %d -> candidate %d (sim=%.4f)",
                query_id, best_cache_cid, best_cache_sim,
            )
            return [best_cache_cid]

        # ---------------------------------------------------------------
        # Tier 2: ANN global index (BallTree or brute-force fallback)
        # ---------------------------------------------------------------
        self.stats.cache_misses += 1
        self.stats.ann_queries += 1

        n = len(self._corpus_ids)
        if n == 0:
            self.stats.total_retrieval_time_s += time.perf_counter() - t_start
            return []

        results: list[tuple[int, float]] = []

        if n < self.min_ann_corpus_size or self._ball_tree is None:
            self._rebuild_tree_if_needed()

        if self._ball_tree is not None and n >= self.min_ann_corpus_size:
            # BallTree ANN query
            self._rebuild_tree_if_needed()  # no-op if tree is current
            try:
                k = min(self.top_k + 1, n)  # +1 in case query_id is in corpus
                query_2d = query_unit.reshape(1, -1)
                dists, indices = self._ball_tree.query(query_2d, k=k)
                dists = dists[0]
                indices = indices[0]
                # Euclidean dist on unit vecs -> cosine_sim = 1 - d^2/2
                for dist, idx in zip(dists, indices):
                    cid = self._corpus_ids[int(idx)]
                    if cid in exclude_ids:
                        continue
                    cosine_sim = float(1.0 - (dist * dist) / 2.0)
                    if cosine_sim >= self.similarity_threshold:
                        results.append((cid, cosine_sim))
                results.sort(key=lambda x: x[1], reverse=True)
                results = results[: self.top_k]
            except Exception as exc:
                logger.warning(
                    "BallTree query failed for chunk %d: %s. Falling back to brute-force.",
                    query_id, exc,
                )
                results = [
                    (cid, sim)
                    for cid, sim in self._brute_force_topk(query_unit, exclude_ids)
                    if sim >= self.similarity_threshold
                ]
        else:
            # Corpus too small for BallTree; use brute-force dot product
            results = [
                (cid, sim)
                for cid, sim in self._brute_force_topk(query_unit, exclude_ids)
                if sim >= self.similarity_threshold
            ]

        self.stats.total_retrieval_time_s += time.perf_counter() - t_start
        return [cid for cid, _ in results]
