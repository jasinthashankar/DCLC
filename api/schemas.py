"""Pydantic Request and Response Schemas for DCLC REST API (Phase 5A)."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health and model availability response."""
    status: str = Field(..., description="Service health status, e.g., 'healthy'")
    checkpoint_available: bool = Field(..., description="Whether the version-aware model checkpoint exists on disk")
    checkpoint_path: str = Field(..., description="Relative path to the model checkpoint")
    device: str = Field(..., description="Compute device currently configured, e.g., 'cpu'")
    active_features: list[str] = Field(..., description="List of active research pipeline components")


class ChunkDetail(BaseModel):
    """Metadata for an individual FastCDC chunk."""
    chunk_index: int = Field(..., description="Sequential index of the chunk in the file")
    byte_offset: int = Field(..., description="Starting byte offset in the source file")
    byte_length: int = Field(..., description="Length of the chunk in bytes")
    sha256: str = Field(..., description="Hex-encoded SHA-256 fingerprint")


class ChunkPreviewResponse(BaseModel):
    """Response schema for FastCDC chunk preview."""
    filename: str = Field(..., description="Name of the uploaded file")
    total_bytes: int = Field(..., description="Total size of the file in bytes")
    chunk_count: int = Field(..., description="Number of FastCDC chunks generated")
    min_chunk_bytes: int = Field(..., description="Minimum chunk size in bytes")
    avg_chunk_bytes: float = Field(..., description="Average chunk size in bytes")
    max_chunk_bytes: int = Field(..., description="Maximum chunk size in bytes")
    chunks: list[ChunkDetail] = Field(..., description="List of chunk metadata (without raw payload)")


class ChunkPairComparison(BaseModel):
    """Comparison detail between a target chunk and its selected candidate."""
    target_chunk_index: int
    target_sha256: str
    target_size_bytes: int
    is_exact_duplicate: bool
    candidate_strategy_used: str
    selected_reference_chunk_index: Optional[int] = None
    selected_reference_sha256: Optional[str] = None
    neural_cosine_similarity: Optional[float] = None
    similarity_threshold: float
    resemblance_detected: bool
    delta_accepted: bool
    delta_size_bytes: Optional[int] = None
    standalone_compressed_size_bytes: int
    reconstruction_verified: bool
    rejection_reason: Optional[str] = None
    # Phase C extensions (backward compatible)
    delta_mode: Optional[str] = "single"
    mrd_group_index: Optional[int] = None
    selected_reference_chunk_indices: Optional[list[int]] = None
    selected_reference_sha256s: Optional[list[str]] = None


class RevisionCompareResponse(BaseModel):
    """Response schema for comparing two document/file revisions."""
    base_filename: str
    target_filename: str
    base_file_size_bytes: int
    target_file_size_bytes: int
    base_chunk_count: int
    target_chunk_count: int
    candidate_strategy: str
    similarity_threshold: float
    exact_duplicates_count: int
    resemblance_pairs_count: int
    deltas_accepted_count: int
    deltas_rejected_count: int
    reconstruction_success_rate: float
    target_raw_bytes: int
    target_standalone_compressed_bytes: int
    target_dclc_stored_bytes: int
    target_storage_savings_pct: float
    target_compression_ratio: float
    detailed_chunk_comparisons: list[ChunkPairComparison]
    # Phase C extensions (backward compatible)
    delta_mode: str = "single"
    mrd_window_size: Optional[int] = None
    mrd_max_refs: Optional[int] = None
    mrd_groups_count: Optional[int] = 0
    mrd_groups_accepted_count: Optional[int] = 0
    mrd_groups_rejected_count: Optional[int] = 0
    indexer_stats: Optional[dict[str, Any]] = None


class SystemStorageBenchmark(BaseModel):
    """Storage statistics for a specific deduplication/compression architecture."""
    stored_bytes: int
    savings_bytes: int
    savings_pct: float
    compression_ratio: float
    exact_duplicates_eliminated: Optional[int] = None
    candidate_recall: Optional[float] = None
    recalled_pairs_count: Optional[int] = None
    avg_candidate_pool: Optional[float] = None
    deltas_accepted: Optional[int] = None
    reconstruction_success_rate: Optional[float] = None


class BenchmarkStorageResponse(BaseModel):
    """Response schema for comparative storage benchmarks."""
    split_evaluated: str
    total_chunks_evaluated: int
    total_logical_raw_bytes: int
    exact_duplicate_targets: int
    resemblance_targets: int
    system_0_raw: dict[str, Any]
    system_1_standalone_compression: dict[str, Any]
    system_2_exact_dedup: dict[str, Any]
    system_3_full_dclc: dict[str, Any]
    # Phase C extension: Multi-Reference Delta comparison
    system_4_dclc_mrd: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    error: str = Field(..., description="High-level error classification")
    detail: str = Field(..., description="Sanitized, user-facing error description")


# --- Phase E: Durable Persistence and Revision Management Schemas ---

class StoredRevisionSummary(BaseModel):
    """Summary metadata for a persisted revision in durable storage."""
    revision_id: str
    target_name: str
    base_name: str
    timestamp: str
    raw_bytes: int
    stored_bytes: int
    storage_savings_pct: float
    compression_ratio: float
    sha256: str
    delta_mode: str
    chunk_count: int


class StoredChunkEntry(BaseModel):
    """Chunk manifest item inside a revision recipe."""
    chunk_index: int
    type: str
    chunk_sha256: str
    chunk_size: int
    stored_size: int
    base_chunk_sha256: Optional[str] = None
    compressed_chunk_sha256: Optional[str] = None
    delta_container_sha256: Optional[str] = None
    subchunk_index: Optional[int] = None
    ref_chunk_sha256_list: Optional[list[str]] = None


class StoredRevisionDetail(BaseModel):
    """Full persisted revision recipe including all chunk mappings."""
    revision_id: str
    target_name: str
    base_name: str
    timestamp: str
    raw_bytes: int
    stored_bytes: int
    storage_savings_pct: float
    compression_ratio: float
    sha256: str
    base_sha256: str
    candidate_strategy: str
    delta_mode: str
    chunk_count: int
    exact_duplicates_count: int
    delta_chunks_count: int
    standalone_chunks_count: int
    chunks: list[dict[str, Any]]


class RestoreRevisionResponse(BaseModel):
    """Response verifying byte-exact restoration of a stored revision."""
    revision_id: str
    target_name: str
    sha256: str
    raw_bytes: int
    chunk_count: int
    verified: bool
    reconstructed_bytes_match: bool
    message: str

