"""FastAPI application entrypoint for DCLC Cloud Storage Redundancy Elimination (Phase 5A).

Exposes verified research pipeline operations:
- /health: Service status and model availability
- /chunk-preview: FastCDC chunking and SHA-256 fingerprinting
- /compare-revisions: Full DCLC redundancy elimination between revisions
- /benchmark-storage: End-to-end comparative benchmark across test split
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    BenchmarkStorageResponse,
    ChunkPreviewResponse,
    ErrorResponse,
    HealthResponse,
    RestoreRevisionResponse,
    RevisionCompareResponse,
    StoredRevisionDetail,
    StoredRevisionSummary,
)
from api.service import DCLCBackendService
from api.rag_routes import router as rag_router
from src.storage_manager import (
    MissingReferenceError,
    RevisionNotFoundError,
    StorageCorruptionError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dclc_api")

# Configuration via environment variables
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_UPLOAD_BYTES = int(os.environ.get("DCLC_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES))

raw_origins = os.environ.get(
    "DCLC_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
)
ALLOWED_ORIGINS = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

CHECKPOINT_PATH = os.environ.get("DCLC_CHECKPOINT_PATH", "checkpoints/dclc_encoder_version_aware.pt")
DEVICE = os.environ.get("DCLC_DEVICE", "cpu")

# Initialize backend service instance
service = DCLCBackendService(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)

app = FastAPI(
    title="DCLC Cloud Storage Deduplication API",
    description=(
        "Production-grade REST backend exposing Deep Contrastive Learning-based "
        "Redundancy Elimination for Similar Data in Cloud Storage Systems."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount DCLC-RAG Router
app.include_router(rag_router)


async def _read_and_validate_file(file: UploadFile, label: str = "Uploaded file") -> bytes:
    """Read upload content and enforce maximum payload boundary without saving to disk."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{label} has an empty or invalid filename.",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"{label} '{file.filename}' ({len(content)} bytes) exceeds the maximum allowed limit of {MAX_UPLOAD_BYTES} bytes.",
        )
    return content


# ---------------------------------------------------------
# Global Exception Handlers (Clean responses, no path leakage)
# ---------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTPException", "detail": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error processing request {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected server error occurred during pipeline execution.",
        },
    )


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get(
    "/",
    tags=["System"],
    summary="API Root and Documentation Index",
)
async def api_root():
    """Welcome index directing users to interactive docs and active endpoints."""
    return {
        "title": "DCLC Cloud Storage Deduplication API",
        "status": "online",
        "documentation": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "health_check": "GET /health",
            "chunk_preview": "POST /chunk-preview",
            "compare_revisions": "POST /compare-revisions",
            "benchmark_storage": "GET /benchmark-storage, POST /benchmark-storage",
            "rag_ingest": "POST /rag/ingest",
            "rag_documents": "GET /rag/documents",
            "rag_status": "GET /rag/status",
            "rag_query": "POST /rag/query",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check and model checkpoint verification",
)
async def health_check() -> HealthResponse:
    """Return system readiness, checkpoint presence, and active pipeline features."""
    return service.check_health()


@app.post(
    "/chunk-preview",
    response_model=ChunkPreviewResponse,
    responses={413: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
    tags=["Chunking"],
    summary="Preview FastCDC chunking and SHA-256 fingerprints",
)
async def preview_chunks(
    file: Annotated[UploadFile, File(description="File to chunk with FastCDC")],
    min_chunk_size: Annotated[int, Query(ge=64, le=65536, description="Minimum chunk size in bytes")] = 4096,
    avg_chunk_size: Annotated[int, Query(ge=128, le=131072, description="Target average chunk size in bytes")] = 8192,
    max_chunk_size: Annotated[int, Query(ge=256, le=262144, description="Maximum chunk size in bytes")] = 16384,
) -> ChunkPreviewResponse:
    """Chunk an uploaded file into content-defined chunks using FastCDC and return structural metadata."""
    if min_chunk_size > avg_chunk_size or avg_chunk_size > max_chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chunk size constraints violated: min_chunk_size <= avg_chunk_size <= max_chunk_size is required.",
        )

    file_bytes = await _read_and_validate_file(file, label="Preview file")
    return service.preview_chunks(
        file_bytes=file_bytes,
        filename=file.filename or "uploaded_file.bin",
        min_size=min_chunk_size,
        avg_size=avg_chunk_size,
        max_size=max_chunk_size,
    )


@app.post(
    "/compare-revisions",
    response_model=RevisionCompareResponse,
    responses={413: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
    tags=["Redundancy Elimination"],
    summary="Compare two revisions with DCLC neural redundancy elimination",
)
async def compare_revisions(
    base_file: Annotated[UploadFile, File(description="Base revision file")],
    target_file: Annotated[UploadFile, File(description="Target revision file to compress against base")],
    candidate_strategy: Annotated[str, Form(description="Candidate retrieval strategy: 'stream_locality', 'rp_lsh', 'embedding_knn', or 'hierarchical_cache'")] = "stream_locality",
    resemblance_threshold: Annotated[float, Form(ge=0.0, le=1.0, description="Resemblance cosine similarity threshold")] = 0.70,
    min_chunk_size: Annotated[int, Form(ge=64, le=65536)] = 4096,
    avg_chunk_size: Annotated[int, Form(ge=128, le=131072)] = 8192,
    max_chunk_size: Annotated[int, Form(ge=256, le=262144)] = 16384,
    delta_mode: Annotated[str, Form(description="Delta compression mode: 'single' (SR) or 'multi' (MRD)")] = "single",
    mrd_window_size: Annotated[int, Form(ge=1, le=16, description="Target chunk window size for Multi-Reference Delta")] = 3,
    mrd_max_refs: Annotated[int, Form(ge=1, le=8, description="Maximum candidate reference chunks for Multi-Reference Delta")] = 3,
) -> RevisionCompareResponse:
    """Execute end-to-end DCLC pipeline:

    1. FastCDC content-defined chunking.
    2. SHA-256 exact-duplicate detection.
    3. Algorithm 1 multi-scale shingle feature extraction.
    4. Radius-4 context sequence assembly.
    5. Transformer encoder embedding inference.
    6. Candidate retrieval (Stream Locality, RP-LSH, k-NN, or Hierarchical Cache + BallTree).
    7. Neural cosine resemblance filtering.
    8. Verified preset-dictionary delta compression (Single-Reference or Multi-Reference MRD).
    """
    valid_strats = {"stream_locality", "rp_lsh", "lsh", "embedding_knn", "knn", "hierarchical_cache", "hierarchical", "hierarchical_ann"}
    if candidate_strategy.lower() not in valid_strats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid candidate_strategy '{candidate_strategy}'. Must be one of: {sorted(list(valid_strats))}",
        )

    valid_delta_modes = {"single", "sr", "multi", "mrd"}
    if delta_mode.lower() not in valid_delta_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid delta_mode '{delta_mode}'. Must be one of: {sorted(list(valid_delta_modes))}",
        )

    if min_chunk_size > avg_chunk_size or avg_chunk_size > max_chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chunk size constraints violated: min_chunk_size <= avg_chunk_size <= max_chunk_size is required.",
        )

    base_bytes = await _read_and_validate_file(base_file, label="Base file")
    target_bytes = await _read_and_validate_file(target_file, label="Target file")

    return service.compare_revisions(
        base_bytes=base_bytes,
        target_bytes=target_bytes,
        base_filename=base_file.filename or "base.txt",
        target_filename=target_file.filename or "target.txt",
        candidate_strategy=candidate_strategy,
        resemblance_threshold=resemblance_threshold,
        min_chunk_size=min_chunk_size,
        avg_chunk_size=avg_chunk_size,
        max_chunk_size=max_chunk_size,
        delta_mode=delta_mode,
        mrd_window_size=mrd_window_size,
        mrd_max_refs=mrd_max_refs,
    )


@app.get(
    "/benchmark-storage",
    response_model=BenchmarkStorageResponse,
    tags=["Evaluation Benchmark"],
    summary="Benchmark storage efficiency across held-out test families (GET)",
)
async def get_benchmark_storage(
    resemblance_threshold: Annotated[float, Query(ge=0.0, le=1.0, description="Resemblance threshold for delta encoding")] = 0.70,
    enable_mrd: Annotated[bool, Query(description="Include Multi-Reference Delta evaluation")] = False,
) -> BenchmarkStorageResponse:
    """Run full comparative benchmark across test split comparing Raw, Standalone Compression, Exact Dedup, and Full DCLC."""
    return service.run_storage_benchmark(resemblance_threshold=resemblance_threshold, enable_mrd=enable_mrd)


@app.post(
    "/benchmark-storage",
    response_model=BenchmarkStorageResponse,
    tags=["Evaluation Benchmark"],
    summary="Benchmark storage efficiency across held-out test families (POST)",
)
async def post_benchmark_storage(
    resemblance_threshold: Annotated[float, Query(ge=0.0, le=1.0, description="Resemblance threshold for delta encoding")] = 0.70,
    enable_mrd: Annotated[bool, Query(description="Include Multi-Reference Delta evaluation")] = False,
) -> BenchmarkStorageResponse:
    """Run full comparative benchmark across test split comparing Raw, Standalone Compression, Exact Dedup, and Full DCLC."""
    return service.run_storage_benchmark(resemblance_threshold=resemblance_threshold, enable_mrd=enable_mrd)


# ==============================================================================
# Phase E: Durable Storage Lifecycle & Revision Restoration Endpoints
# ==============================================================================

async def _handle_store_revision(
    base_file: UploadFile,
    target_file: UploadFile,
    candidate_strategy: str,
    resemblance_threshold: float,
    min_chunk_size: int,
    avg_chunk_size: int,
    max_chunk_size: int,
    delta_mode: str,
    mrd_window_size: int,
    mrd_max_refs: int,
) -> StoredRevisionDetail:
    """Internal helper to process revision storage."""
    base_bytes = await base_file.read()
    target_bytes = await target_file.read()

    if len(base_bytes) > MAX_UPLOAD_BYTES or len(target_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum configured size of {MAX_UPLOAD_BYTES} bytes.",
        )

    try:
        recipe = service.store_revision(
            base_bytes=base_bytes,
            target_bytes=target_bytes,
            base_name=base_file.filename or "base.txt",
            target_name=target_file.filename or "target.txt",
            candidate_strategy=candidate_strategy,
            resemblance_threshold=resemblance_threshold,
            delta_mode=delta_mode,
            mrd_window_size=mrd_window_size,
            mrd_max_refs=mrd_max_refs,
            min_chunk_size=min_chunk_size,
            avg_chunk_size=avg_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        return StoredRevisionDetail(**recipe)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to store revision: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post(
    "/store-revision",
    response_model=StoredRevisionDetail,
    status_code=status.HTTP_201_CREATED,
    tags=["Durable Storage Lifecycle"],
    summary="Compress and durably persist a target revision against a base file",
)
async def store_revision_endpoint(
    base_file: UploadFile = File(..., description="Base reference file"),
    target_file: UploadFile = File(..., description="Target revision file"),
    candidate_strategy: Annotated[str, Form(description="Candidate strategy")] = "hierarchical_cache",
    resemblance_threshold: Annotated[float, Form(description="Similarity threshold")] = 0.50,
    min_chunk_size: Annotated[int, Form(description="FastCDC minimum chunk size")] = 2048,
    avg_chunk_size: Annotated[int, Form(description="FastCDC average chunk size")] = 4096,
    max_chunk_size: Annotated[int, Form(description="FastCDC maximum chunk size")] = 8192,
    delta_mode: Annotated[str, Form(description="Delta compression mode: 'single' or 'multi'")] = "single",
    mrd_window_size: Annotated[int, Form(description="Window size for MRD (1-16)")] = 3,
    mrd_max_refs: Annotated[int, Form(description="Maximum references for MRD (1-8)")] = 3,
) -> StoredRevisionDetail:
    """Store revision recipe and content-addressed chunks/deltas on disk."""
    return await _handle_store_revision(
        base_file=base_file,
        target_file=target_file,
        candidate_strategy=candidate_strategy,
        resemblance_threshold=resemblance_threshold,
        min_chunk_size=min_chunk_size,
        avg_chunk_size=avg_chunk_size,
        max_chunk_size=max_chunk_size,
        delta_mode=delta_mode,
        mrd_window_size=mrd_window_size,
        mrd_max_refs=mrd_max_refs,
    )


@app.post(
    "/revisions/store",
    response_model=StoredRevisionDetail,
    status_code=status.HTTP_201_CREATED,
    tags=["Durable Storage Lifecycle"],
    summary="Compress and durably persist a target revision (RESTful alias)",
)
async def store_revision_alias_endpoint(
    base_file: UploadFile = File(..., description="Base reference file"),
    target_file: UploadFile = File(..., description="Target revision file"),
    candidate_strategy: Annotated[str, Form(description="Candidate strategy")] = "hierarchical_cache",
    resemblance_threshold: Annotated[float, Form(description="Similarity threshold")] = 0.50,
    min_chunk_size: Annotated[int, Form(description="FastCDC minimum chunk size")] = 2048,
    avg_chunk_size: Annotated[int, Form(description="FastCDC average chunk size")] = 4096,
    max_chunk_size: Annotated[int, Form(description="FastCDC maximum chunk size")] = 8192,
    delta_mode: Annotated[str, Form(description="Delta compression mode: 'single' or 'multi'")] = "single",
    mrd_window_size: Annotated[int, Form(description="Window size for MRD (1-16)")] = 3,
    mrd_max_refs: Annotated[int, Form(description="Maximum references for MRD (1-8)")] = 3,
) -> StoredRevisionDetail:
    """RESTful alias for /store-revision."""
    return await _handle_store_revision(
        base_file=base_file,
        target_file=target_file,
        candidate_strategy=candidate_strategy,
        resemblance_threshold=resemblance_threshold,
        min_chunk_size=min_chunk_size,
        avg_chunk_size=avg_chunk_size,
        max_chunk_size=max_chunk_size,
        delta_mode=delta_mode,
        mrd_window_size=mrd_window_size,
        mrd_max_refs=mrd_max_refs,
    )


@app.get(
    "/revisions",
    response_model=list[StoredRevisionSummary],
    tags=["Durable Storage Lifecycle"],
    summary="List all persisted revision recipes in durable storage",
)
async def list_revisions_endpoint() -> list[StoredRevisionSummary]:
    """Retrieve summary metadata of all revisions stored on disk."""
    items = service.list_revisions()
    return [StoredRevisionSummary(**item) for item in items]


@app.get(
    "/revisions/{revision_id}",
    response_model=StoredRevisionDetail,
    tags=["Durable Storage Lifecycle"],
    summary="Retrieve full revision recipe manifest by revision ID",
)
async def get_revision_endpoint(revision_id: str) -> StoredRevisionDetail:
    """Fetch complete chunk recipe manifest for a stored revision."""
    try:
        recipe = service.get_revision(revision_id)
        return StoredRevisionDetail(**recipe)
    except RevisionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except StorageCorruptionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


async def _handle_restore_revision(revision_id: str) -> RestoreRevisionResponse:
    """Internal helper to restore and verify a revision."""
    try:
        restored_bytes, recipe = service.restore_revision(revision_id)
        return RestoreRevisionResponse(
            revision_id=revision_id,
            target_name=recipe.get("target_name", "restored.bin"),
            sha256=recipe.get("sha256", ""),
            raw_bytes=len(restored_bytes),
            chunk_count=recipe.get("chunk_count", 0),
            verified=True,
            reconstructed_bytes_match=True,
            message="Byte-for-byte exact restoration verified against stored SHA-256 fingerprint.",
        )
    except RevisionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except (MissingReferenceError, StorageCorruptionError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error restoring revision %s: %s", revision_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.get(
    "/restore-revision/{revision_id}",
    response_model=RestoreRevisionResponse,
    tags=["Durable Storage Lifecycle"],
    summary="Reconstruct target revision from disk chunks/deltas and verify SHA-256",
)
async def restore_revision_endpoint(revision_id: str) -> RestoreRevisionResponse:
    """Reconstruct revision from disk manifest and verify byte-exact integrity."""
    return await _handle_restore_revision(revision_id)


@app.get(
    "/revisions/{revision_id}/restore",
    response_model=RestoreRevisionResponse,
    tags=["Durable Storage Lifecycle"],
    summary="Reconstruct target revision from disk chunks/deltas (RESTful alias)",
)
async def restore_revision_alias_endpoint(revision_id: str) -> RestoreRevisionResponse:
    """RESTful alias for /restore-revision/{revision_id}."""
    return await _handle_restore_revision(revision_id)


async def _handle_download_revision(revision_id: str) -> Response:
    """Internal helper to stream restored file bytes."""
    try:
        restored_bytes, recipe = service.restore_revision(revision_id)
        filename = recipe.get("target_name", f"{revision_id}.bin")
        return Response(
            content=restored_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-DCLC-SHA256": recipe.get("sha256", ""),
                "X-DCLC-Original-Size": str(len(restored_bytes)),
            },
        )
    except RevisionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except (MissingReferenceError, StorageCorruptionError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.get(
    "/download-restored/{revision_id}",
    tags=["Durable Storage Lifecycle"],
    summary="Download reconstructed file bytes from disk manifest",
)
async def download_restored_endpoint(revision_id: str) -> Response:
    """Download restored binary file directly."""
    return await _handle_download_revision(revision_id)


@app.get(
    "/revisions/{revision_id}/download",
    tags=["Durable Storage Lifecycle"],
    summary="Download reconstructed file bytes (RESTful alias)",
)
async def download_restored_alias_endpoint(revision_id: str) -> Response:
    """RESTful alias for /download-restored/{revision_id}."""
    return await _handle_download_revision(revision_id)

