# src/rag/__init__.py
# DCLC-RAG Phase 6B: Document ingestion and chunk-to-version mapping package.

from src.rag.config import (
    DEFAULT_PRECISION,
    SUPPORTED_PRECISIONS,
    EmbeddingConfig,
    get_embedding_precision,
    validate_precision,
)

__all__ = [
    "DEFAULT_PRECISION",
    "SUPPORTED_PRECISIONS",
    "EmbeddingConfig",
    "get_embedding_precision",
    "validate_precision",
]
