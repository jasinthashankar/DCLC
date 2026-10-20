"""
src/rag/retrieval/embedding_store.py
--------------------------------------
DCLC-RAG Phase 6C — Dense embedding store with shared-embedding deduplication.

Design guarantee
----------------
One embedding vector is computed and stored per unique chunk_id.
chunk_id = sha256(raw_bytes).hexdigest() — exact byte identity only.
Two chunks with different bytes always have different chunk_ids, regardless
of how similar their text is.

This is the core contribution of DCLC-RAG relative to a standard baseline:
a chunk shared across N versions is embedded exactly once, not N times.

Embedding model
---------------
Default: sentence-transformers/all-MiniLM-L6-v2
  - Dimension: 384
  - Size on disk: ~91 MB (downloaded once, cached locally)
  - License: Apache 2.0
  - No paid API, no network call after first download

Offline / no-model fallback
---------------------------
If sentence-transformers is not installed or the model cannot be loaded,
EmbeddingStore falls back to TF-IDF cosine similarity over the vocabulary
built by the BM25 index.  This is documented and explicit; it does not
silently change ranking behavior — callers receive a warning and the
model_name attribute is set to "tfidf_fallback".

Similarity metric
-----------------
Cosine similarity via L2-normalised dot product (numpy matrix multiply).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.rag.config import (
    DEFAULT_PRECISION,
    SUPPORTED_PRECISIONS,
    get_embedding_precision,
    validate_precision,
)
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex

logger = logging.getLogger(__name__)

# Model configuration
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384  # dimension for all-MiniLM-L6-v2


_LOADED_MODELS: Dict[str, Any] = {}


def _get_sentence_transformer(model_name: str):
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        _LOADED_MODELS[model_name] = model
        return model
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Embedding Store
# ---------------------------------------------------------------------------


_VALID_PRECISIONS: frozenset[str] = SUPPORTED_PRECISIONS


class EmbeddingStore:
    """
    Dense embedding store: one vector per unique chunk_id.

    Supports configurable precision modes:
    - "float32": Standard full-precision IEEE float32 (4 bytes / coordinate, default).
    - "float16": Half-precision IEEE float16 (2 bytes / coordinate, 2x compression).
    - "int8" / "sq8": Symmetric 8-bit scalar quantization (1 byte / coordinate, 4x compression).

    Attributes (after build())
    --------------------------
    precision           : str  — active precision mode ("float32", "float16", "int8")
    model_name          : str  — active model name or "tfidf_fallback"
    embedding_dim       : int  — vector dimension (384 for MiniLM)
    embedding_count     : int  — number of stored vectors (= unique_chunk_count)
    build_time_seconds  : float — wall-clock seconds for embedding generation
    storage_bytes       : int  — total dense vector memory payload in bytes
    """

    def __init__(self, precision: Optional[str] = None) -> None:
        if precision is None:
            self.precision = get_embedding_precision()
        else:
            self.precision = validate_precision(precision)
        self._chunk_ids: List[str] = []         # ordered list of indexed chunk_ids
        self._matrix: Optional[np.ndarray] = None   # shape (N, D), float32, float16, or int8
        self._scales: Optional[np.ndarray] = None   # shape (N,), float32 quantization scales
        self._int_norms: Optional[np.ndarray] = None # shape (N,), float32 norms of integer vectors
        self.model_name: str = ""
        self.embedding_dim: int = _EMBEDDING_DIM
        self.build_time_seconds: float = 0.0
        self._vocab: Dict[str, int] = {}        # used by tfidf_fallback only
        self._idf_vec: Optional[np.ndarray] = None  # used by tfidf_fallback only

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _set_empty_matrix(self) -> None:
        """Initialise a zero-row matrix for empty document sets."""
        if self.precision == "float32":
            self._matrix = np.zeros((0, self.embedding_dim), dtype=np.float32)
        elif self.precision == "float16":
            self._matrix = np.zeros((0, self.embedding_dim), dtype=np.float16)
        elif self.precision == "int8":
            self._matrix = np.zeros((0, self.embedding_dim), dtype=np.int8)
            self._scales = np.zeros((0,), dtype=np.float32)
            self._int_norms = np.zeros((0,), dtype=np.float32)

    def _store_matrix(self, vecs: np.ndarray) -> None:
        """
        Store unit-normalized float32 vectors into the configured precision format.

        Parameters
        ----------
        vecs : np.ndarray
            Shape (N, D), float32, unit L2-normalised (or zero vectors).
        """
        if self.precision == "float32":
            self._matrix = vecs.astype(np.float32)
            self._scales = None
            self._int_norms = None
        elif self.precision == "float16":
            self._matrix = vecs.astype(np.float16)
            self._scales = None
            self._int_norms = None
        elif self.precision == "int8":
            # Symmetric 8-bit scalar quantization (SQ8)
            # Normalized vectors have components strictly within [-1.0, 1.0].
            # Quantization scale: S = 127.0
            scale = 127.0
            q = np.clip(np.round(vecs * scale), -127, 127).astype(np.int8)
            self._matrix = q
            self._scales = np.full((len(vecs),), scale, dtype=np.float32)

            # Store integer Euclidean norms to ensure cosine similarity is
            # mathematically exact without dequantizing the entire matrix into memory.
            raw_norms = np.linalg.norm(q.astype(np.float32), axis=1)
            self._int_norms = raw_norms.astype(np.float32)

    def build(
        self,
        chunk_index: ChunkVersionIndex,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 32,
        precision: Optional[str] = None,
    ) -> None:
        """
        Compute one embedding per unique chunk_id in chunk_index.

        DCLC-RAG guarantee: only unique content (by SHA-256 fingerprint)
        generates a new embedding. Shared chunks are embedded once.

        Parameters
        ----------
        chunk_index : ChunkVersionIndex
            Already-built reverse index.
        model_name  : str
            HuggingFace model identifier. Default: all-MiniLM-L6-v2.
        batch_size  : int
            Encoding batch size passed to SentenceTransformer.encode().
        precision   : str | None
            Optional precision override ("float32", "float16", "int8").
        """
        if precision is not None:
            self.precision = validate_precision(precision)

        chunk_ids = chunk_index.all_chunk_ids()
        texts = [chunk_index.get_text(cid) for cid in chunk_ids]
        self._chunk_ids = chunk_ids

        # Handle empty chunk set safely
        if not chunk_ids:
            self._set_empty_matrix()
            self.model_name = model_name
            self.build_time_seconds = 0.0
            return

        t0 = time.perf_counter()
        model = _get_sentence_transformer(model_name)

        if model is not None:
            try:
                vecs = model.encode(texts, batch_size=batch_size,
                                    convert_to_numpy=True,
                                    show_progress_bar=False)
                vecs = np.array(vecs, dtype=np.float32)
                self.model_name = model_name
                self.embedding_dim = vecs.shape[1] if vecs.ndim == 2 else _EMBEDDING_DIM
            except Exception as exc:
                warnings.warn(
                    f"Failed to encode with model '{model_name}': {exc}. "
                    "Falling back to TF-IDF cosine similarity.",
                    stacklevel=2,
                )
                vecs = self._tfidf_encode(texts)
                self.model_name = "tfidf_fallback"
        else:
            warnings.warn(
                f"Model '{model_name}' could not be loaded. "
                "Falling back to TF-IDF cosine similarity.",
                stacklevel=2,
            )
            vecs = self._tfidf_encode(texts)
            self.model_name = "tfidf_fallback"

        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        norm_vecs = vecs / norms

        self._store_matrix(norm_vecs)

        self.build_time_seconds = time.perf_counter() - t0
        logger.info(
            "EmbeddingStore built: %d unique vectors, model=%s, dim=%d, precision=%s, time=%.2fs",
            len(self._chunk_ids), self.model_name, self.embedding_dim, self.precision,
            self.build_time_seconds,
        )

    def build_baseline(
        self,
        chunk_ids: List[str],
        texts: List[str],
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 32,
        precision: Optional[str] = None,
    ) -> None:
        """
        Baseline variant: embed a list of slot texts without deduplication.

        chunk_ids here are slot keys (e.g. "doc::ver::idx") — not SHA-256
        content fingerprints.  Duplicate content is embedded N times.

        Parameters
        ----------
        chunk_ids : list[str]
            Ordered list of slot key strings.
        texts     : list[str]
            Corresponding text for each slot.
        precision : str | None
            Optional precision override.
        """
        if precision is not None:
            self.precision = validate_precision(precision)

        assert len(chunk_ids) == len(texts), "chunk_ids and texts must have the same length."
        self._chunk_ids = list(chunk_ids)

        if not chunk_ids:
            self._set_empty_matrix()
            self.model_name = model_name
            self.build_time_seconds = 0.0
            return

        t0 = time.perf_counter()
        model = _get_sentence_transformer(model_name)

        if model is not None:
            try:
                vecs = model.encode(texts, batch_size=batch_size,
                                    convert_to_numpy=True,
                                    show_progress_bar=False)
                vecs = np.array(vecs, dtype=np.float32)
                self.model_name = model_name
                self.embedding_dim = vecs.shape[1] if vecs.ndim == 2 else _EMBEDDING_DIM
            except Exception as exc:
                warnings.warn(f"Model load failed: {exc}. Using TF-IDF fallback.")
                vecs = self._tfidf_encode(texts)
                self.model_name = "tfidf_fallback"
        else:
            warnings.warn("sentence-transformers not installed. Using TF-IDF fallback.")
            vecs = self._tfidf_encode(texts)
            self.model_name = "tfidf_fallback"

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        norm_vecs = vecs / norms

        self._store_matrix(norm_vecs)

        self.build_time_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_vector(self, text: str) -> np.ndarray:
        """
        Encode a query string to a normalised vector.

        Uses the same model/fallback as was used during build().
        Returns a float32 array of shape (embedding_dim,).
        """
        if self._matrix is None:
            raise RuntimeError("EmbeddingStore has not been built yet.")

        if self.model_name != "tfidf_fallback":
            model = _get_sentence_transformer(self.model_name)
            if model is not None:
                try:
                    vec = model.encode([text], convert_to_numpy=True,
                                       show_progress_bar=False)[0]
                except Exception:
                    vec = self._tfidf_encode([text])[0]
            else:
                vec = self._tfidf_encode([text])[0]
        else:
            vec = self._tfidf_encode([text])[0]

        vec = vec.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def cosine_scores(
        self,
        query_vector: np.ndarray,
        scope: Optional[set[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute cosine similarity between query_vector and all stored vectors.

        Because vectors are L2-normalised, cosine = dot product.
        Supports float32, float16, and int8 quantized scoring.

        Parameters
        ----------
        query_vector : np.ndarray
            Shape (D,), already L2-normalised float32.
        scope : set[str] | None
            If provided, only keys in this set are scored and returned.

        Returns
        -------
        dict[str, float]
            {chunk_id_or_slot_key: cosine_score} for all entries.
        """
        if self._matrix is None:
            raise RuntimeError("EmbeddingStore has not been built yet.")
        if len(self._chunk_ids) == 0:
            return {}

        q_vec = query_vector.astype(np.float32)

        if self.precision == "float32":
            scores_vec = self._matrix @ q_vec
        elif self.precision == "float16":
            scores_vec = self._matrix.astype(np.float32) @ q_vec
        elif self.precision == "int8":
            # Efficient integer dot product with query vector
            dot = self._matrix.astype(np.float32) @ q_vec
            if self._int_norms is not None:
                safe_norms = np.where(self._int_norms == 0, 1.0, self._int_norms)
                scores_vec = np.where(self._int_norms == 0, 0.0, dot / safe_norms)
            else:
                scores_vec = dot / 127.0
        else:
            scores_vec = self._matrix.astype(np.float32) @ q_vec

        result: Dict[str, float] = {}
        for i, key in enumerate(self._chunk_ids):
            if scope is not None and key not in scope:
                continue
            result[key] = float(scores_vec[i])
        return result

    # ------------------------------------------------------------------
    # TF-IDF fallback encoder
    # ------------------------------------------------------------------

    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to TF-IDF vectors using the vocabulary from all provided
        texts.  Used as a pure-numpy offline fallback.

        Returns float32 array of shape (len(texts), V) where V = vocab size.
        """
        import math, re
        token_re = re.compile(r"[A-Za-z0-9]+")

        def tokenize(t: str):
            return token_re.findall(t.lower())

        tokenized = [tokenize(t) for t in texts]

        # Build or reuse vocab
        if not self._vocab:
            all_terms: set[str] = set()
            for toks in tokenized:
                all_terms.update(toks)
            self._vocab = {term: i for i, term in enumerate(sorted(all_terms))}

            N = len(texts)
            from collections import Counter
            df = np.zeros(len(self._vocab), dtype=np.float64)
            for toks in tokenized:
                for term in set(toks):
                    if term in self._vocab:
                        df[self._vocab[term]] += 1
            self._idf_vec = np.log((N + 1) / (df + 1)) + 1.0

        V = len(self._vocab)
        self.embedding_dim = V
        mat = np.zeros((len(texts), V), dtype=np.float32)
        from collections import Counter
        for i, toks in enumerate(tokenized):
            if not toks:
                continue
            cnt = Counter(toks)
            for term, freq in cnt.items():
                if term in self._vocab:
                    j = self._vocab[term]
                    idf = float(self._idf_vec[j]) if self._idf_vec is not None else 1.0
                    mat[i, j] = float(freq) / len(toks) * idf

        return mat

    # ------------------------------------------------------------------
    # Properties & Inspection
    # ------------------------------------------------------------------

    @property
    def embedding_count(self) -> int:
        """Number of stored embedding vectors."""
        return len(self._chunk_ids)

    @property
    def raw_vector_bytes(self) -> int:
        """Actual byte footprint of the raw embedding matrix buffer."""
        return self._matrix.nbytes if self._matrix is not None else 0

    @property
    def metadata_bytes(self) -> int:
        """Byte footprint of quantization scales and integer norm vectors."""
        total = 0
        if self._scales is not None:
            total += self._scales.nbytes
        if self._int_norms is not None:
            total += self._int_norms.nbytes
        return total

    @property
    def storage_bytes(self) -> int:
        """
        Total byte payload for stored embedding representations.
        - float32: embedding_count × embedding_dim × 4
        - float16: embedding_count × embedding_dim × 2
        - int8:    raw matrix bytes (1 B / coord) + norm metadata (4 B / vector)
        """
        if self._matrix is None or self.embedding_count == 0:
            return 0
        if self.precision == "float32":
            return self.embedding_count * self.embedding_dim * 4
        elif self.precision == "float16":
            return self.embedding_count * self.embedding_dim * 2
        elif self.precision == "int8":
            return self.raw_vector_bytes + self.metadata_bytes
        return self.raw_vector_bytes

    @property
    def compression_ratio_vs_fp32(self) -> float:
        """Theoretical storage reduction factor compared to uncompressed float32."""
        fp32_bytes = self.embedding_count * self.embedding_dim * 4
        if self.storage_bytes == 0:
            return 1.0
        return fp32_bytes / self.storage_bytes

    def get_chunk_ids(self) -> List[str]:
        """Return the ordered list of chunk_ids in this store."""
        return list(self._chunk_ids)

    def get_vector(self, chunk_id: str) -> np.ndarray:
        """
        Return the reconstructed, unit-normalized float32 vector for chunk_id.
        Handles dequantization from int8/sq8 and float16.
        """
        if self._matrix is None:
            raise RuntimeError("EmbeddingStore has not been built yet.")
        try:
            idx = self._chunk_ids.index(chunk_id)
        except ValueError:
            raise KeyError(f"Chunk ID '{chunk_id}' not found in EmbeddingStore.")

        if self.precision == "float32":
            return self._matrix[idx].copy()
        elif self.precision == "float16":
            vec = self._matrix[idx].astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        elif self.precision == "int8":
            raw_q = self._matrix[idx].astype(np.float32)
            norm = float(self._int_norms[idx]) if self._int_norms is not None else float(np.linalg.norm(raw_q))
            if norm == 0.0:
                return np.zeros(self.embedding_dim, dtype=np.float32)
            return raw_q / norm

        return self._matrix[idx].astype(np.float32)

    def dequantize(self, chunk_id: str) -> np.ndarray:
        """Alias for get_vector(chunk_id)."""
        return self.get_vector(chunk_id)

    # ------------------------------------------------------------------
    # Persistence & Serialization (Phase 7D)
    # ------------------------------------------------------------------

    def save(self, file_path: str | Path) -> None:
        """
        Persist the built embedding store and metadata to a compressed .npz archive.
        Preserves exact precision format (float32, float16, or int8), chunk IDs,
        scales, and integer norm metadata.
        """
        if self._matrix is None:
            raise RuntimeError("Cannot save an unbuilt EmbeddingStore.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "precision": self.precision,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "build_time_seconds": self.build_time_seconds,
            "has_scales": self._scales is not None,
            "has_int_norms": self._int_norms is not None,
            "is_tfidf": bool(self._vocab),
        }
        meta_json = json.dumps(meta)

        save_dict: Dict[str, Any] = {
            "matrix": self._matrix,
            "chunk_ids": np.array(self._chunk_ids, dtype=object),
            "meta_json": np.array(meta_json),
        }
        if self._scales is not None:
            save_dict["scales"] = self._scales
        if self._int_norms is not None:
            save_dict["int_norms"] = self._int_norms
        if self._vocab:
            save_dict["vocab_json"] = np.array(json.dumps(self._vocab))
            if self._idf_vec is not None:
                save_dict["idf_vec"] = self._idf_vec

        np.savez_compressed(path, **save_dict)
        logger.info(
            "Saved EmbeddingStore (%s, %d vectors) to %s",
            self.precision,
            len(self._chunk_ids),
            path,
        )

    @classmethod
    def load(cls, file_path: str | Path) -> EmbeddingStore:
        """
        Load an EmbeddingStore from a compressed .npz archive, restoring exact precision,
        quantization metadata, matrix buffer, and chunk index mapping.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Embedding store file not found: {path}")

        with np.load(path, allow_pickle=True) as data:
            meta_json = str(data["meta_json"])
            meta = json.loads(meta_json)

            store = cls(precision=meta["precision"])
            store.model_name = meta.get("model_name", "")
            store.embedding_dim = int(meta.get("embedding_dim", _EMBEDDING_DIM))
            store.build_time_seconds = float(meta.get("build_time_seconds", 0.0))

            store._matrix = data["matrix"]
            store._chunk_ids = list(data["chunk_ids"])

            if "scales" in data:
                store._scales = data["scales"]
            if "int_norms" in data:
                store._int_norms = data["int_norms"]
            if "vocab_json" in data:
                store._vocab = json.loads(str(data["vocab_json"]))
            if "idf_vec" in data:
                store._idf_vec = data["idf_vec"]

            return store
