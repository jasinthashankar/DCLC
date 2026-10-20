"""
src/rag/config.py
------------------
DCLC-RAG Phase 7D — Centralized Configuration Architecture for Embedding Precision
and Retrieval Parameters.

Environment variables:
- DCLC_EMBEDDING_PRECISION: Precision mode for dense embedding vectors.
  Supported values: "float32" (default), "float16", "int8", "sq8".
  Note: "sq8" is normalized to "int8" (symmetric 8-bit scalar quantization).
- DCLC_EMBEDDING_MODEL: Embedding model name (default: "all-MiniLM-L6-v2").
- DCLC_EMBEDDING_DIM: Embedding dimension (default: 384).
- DCLC_RAG_PROVIDER: LLM provider ("mock", "groq", "local").
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, FrozenSet, Optional

# Supported embedding precision modes
SUPPORTED_PRECISIONS: Final[FrozenSet[str]] = frozenset({"float32", "float16", "int8", "sq8"})
DEFAULT_PRECISION: Final[str] = "float32"
DEFAULT_MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM: Final[int] = 384


def normalize_precision(precision: str) -> str:
    """
    Normalize precision string: strip whitespace, lowercase, and map aliases.
    "sq8" maps to "int8" (symmetric 8-bit scalar quantization).
    """
    cleaned = precision.strip().lower()
    if cleaned == "sq8":
        return "int8"
    return cleaned


def validate_precision(precision: str) -> str:
    """
    Validate precision mode and return normalized canonical value ('float32', 'float16', or 'int8').

    Raises:
        ValueError: If precision is not in SUPPORTED_PRECISIONS.
    """
    cleaned = precision.strip().lower()
    if cleaned not in SUPPORTED_PRECISIONS:
        sorted_supported = sorted(list(SUPPORTED_PRECISIONS))
        raise ValueError(
            f"Invalid precision '{precision}'. Must be one of: {sorted_supported}"
        )
    return normalize_precision(cleaned)


def get_embedding_precision(override: Optional[str] = None) -> str:
    """
    Resolve active embedding precision with explicit priority:
    1. Explicit override argument (if provided and non-empty).
    2. Environment variable DCLC_EMBEDDING_PRECISION.
    3. Default value: "float32".

    Returns:
        Canonical precision mode ("float32", "float16", or "int8").

    Raises:
        ValueError: If configured precision is invalid.
    """
    if override is not None and str(override).strip():
        return validate_precision(str(override))

    env_val = os.environ.get("DCLC_EMBEDDING_PRECISION", "").strip()
    if env_val:
        return validate_precision(env_val)

    return DEFAULT_PRECISION


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration container for embedding storage and precision."""
    precision: str = DEFAULT_PRECISION
    model_name: str = DEFAULT_MODEL_NAME
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    batch_size: int = 32

    @classmethod
    def from_env(cls, precision_override: Optional[str] = None) -> EmbeddingConfig:
        """Construct EmbeddingConfig by resolving environment variables and defaults."""
        prec = get_embedding_precision(precision_override)
        model = os.environ.get("DCLC_EMBEDDING_MODEL", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        try:
            dim = int(os.environ.get("DCLC_EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)).strip())
        except ValueError:
            dim = DEFAULT_EMBEDDING_DIM
        return cls(precision=prec, model_name=model, embedding_dim=dim)
