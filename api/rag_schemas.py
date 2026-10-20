"""
api/rag_schemas.py
------------------
Pydantic schemas for DCLC-RAG version-aware document ingestion, retrieval,
and grounded question answering with verified citations (Phase 6E).
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class RAGChunkInfo(BaseModel):
    """Metadata for a single chunk within a document version."""
    chunk_id: str
    chunk_index: int
    byte_offset: int
    byte_length: int
    is_shared: bool
    text_preview: str


class RAGVersionSummary(BaseModel):
    """Summary of a single ingested document version."""
    version_id: str
    doc_id: str
    filename: str
    extension: str
    content_sha256: str
    chunk_count: int
    total_bytes: int


class RAGDocumentSummary(BaseModel):
    """A logical document containing one or more version revisions."""
    doc_id: str
    filename: str
    extension: str
    version_count: int
    versions: List[RAGVersionSummary]


class RAGDocumentsResponse(BaseModel):
    """Response containing all registered documents and global chunk statistics."""
    documents: List[RAGDocumentSummary]
    total_documents: int
    total_versions: int
    total_unique_chunks: int
    shared_chunk_count: int


class RAGIngestResponse(BaseModel):
    """Response returned upon ingesting a document file."""
    doc_id: str
    version_id: str
    filename: str
    extension: str
    content_sha256: str
    chunk_count: int
    total_bytes: int
    shared_chunks_in_version: int
    is_new_document: bool
    is_duplicate_version: bool
    message: str


class RAGQueryRequest(BaseModel):
    """User query request with optional version and document scoping."""
    query: str = Field(..., min_length=1, description="Question to answer using retrieved evidence.")
    doc_id: Optional[str] = Field(None, description="Optional document ID to restrict retrieval scope.")
    version_id: Optional[str] = Field(None, description="Optional version ID to enforce strict version isolation.")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of evidence chunks to retrieve.")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Hybrid retrieval balance: 1.0 = BM25, 0.0 = Dense Cosine.")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Generation temperature (default 0.0 for deterministic factual QA).")


class RAGVerifiedCitation(BaseModel):
    """A citation verified by the citation validator against retrieved evidence."""
    tag: str
    chunk_id: str
    doc_id: str
    version_id: str
    filename: str
    chunk_index: int
    byte_offset: int
    byte_length: int
    snippet: str
    is_valid: bool = True


class RAGQueryResponse(BaseModel):
    """Structured response from the RAG Question Answering Engine."""
    status: str  # "SUCCESS", "ABSTAIN", "ERROR"
    query: str
    answer: str
    target_doc_id: Optional[str] = None
    target_version_id: Optional[str] = None
    citations: List[RAGVerifiedCitation] = []
    retrieved_chunk_count: int = 0
    used_chunk_count: int = 0
    provider_name: str
    model_name: str
    latency_retrieval_ms: float
    latency_generation_ms: float
    total_latency_ms: float
    abstention_reason: Optional[str] = None
    invalid_citations: List[str] = []


class RAGStatusResponse(BaseModel):
    """RAG system status, active LLM provider, and deduplication savings metrics."""
    status: str
    provider_name: str
    model_name: str
    total_documents: int
    total_versions: int
    total_unique_chunks: int
    shared_chunk_count: int
    dense_embeddings_count: int
    dense_dimension: int
    dedup_savings_pct: float
    embedding_precision: str = "float32"
    storage_bytes: int = 0
    compression_ratio_vs_fp32: float = 1.0
