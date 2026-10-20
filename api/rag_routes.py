"""
api/rag_routes.py
-----------------
FastAPI router for DCLC-RAG endpoints (Phase 6E):
- POST /rag/ingest: Ingest file as a new document or a new version of an existing document.
- GET  /rag/documents: List all registered documents, versions, and chunk statistics.
- GET  /rag/status: Return system status, active LLM provider, and dedup metrics.
- POST /rag/query: Grounded question answering scoped to optional doc_id and version_id.
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from api.rag_schemas import (
    RAGDocumentsResponse,
    RAGIngestResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGStatusResponse,
)
from api.rag_service import RAGBackendService

logger = logging.getLogger("dclc_rag_api")

router = APIRouter(prefix="/rag", tags=["DCLC-RAG"])

# Singleton service instance for RAG operations
rag_service = RAGBackendService()

# Maximum upload limit (default 25 MB for text/markdown documents)
MAX_RAG_FILE_BYTES = 25 * 1024 * 1024


@router.post(
    "/ingest",
    response_model=RAGIngestResponse,
    summary="Ingest a document version into DCLC-RAG",
    description=(
        "Upload a .txt, .md, or .py document. If doc_id is omitted, registers a new document. "
        "If doc_id is provided, registers a new version revision for that document."
    ),
)
async def ingest_document(
    file: Annotated[UploadFile, File(description="Document file (.txt, .md, .py)")],
    doc_id: Annotated[Optional[str], Form(description="Optional existing doc_id to add a version to")] = None,
    min_chunk_size: Annotated[int, Form(ge=32, le=65536, description="FastCDC minimum chunk size")] = 64,
    avg_chunk_size: Annotated[int, Form(ge=64, le=131072, description="FastCDC average chunk size")] = 256,
    max_chunk_size: Annotated[int, Form(ge=128, le=262144, description="FastCDC maximum chunk size")] = 1024,
) -> RAGIngestResponse:
    """Ingest a document into the RAG registry and rebuild index structures."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a non-empty filename.",
        )

    if min_chunk_size > avg_chunk_size or avg_chunk_size > max_chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chunk size constraint violated: min_chunk_size <= avg_chunk_size <= max_chunk_size.",
        )

    content = await file.read()
    if len(content) > MAX_RAG_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_RAG_FILE_BYTES} bytes.",
        )

    return rag_service.ingest_document(
        file_bytes=content,
        filename=file.filename,
        doc_id=doc_id,
        min_size=min_chunk_size,
        avg_size=avg_chunk_size,
        max_size=max_chunk_size,
    )


@router.get(
    "/documents",
    response_model=RAGDocumentsResponse,
    summary="List registered documents and versions",
    description="Retrieve all documents, version identifiers, chunk counts, and deduplication statistics.",
)
async def list_documents() -> RAGDocumentsResponse:
    """List all registered documents and versions in the DCLC-RAG knowledge base."""
    return rag_service.list_documents()


@router.get(
    "/status",
    response_model=RAGStatusResponse,
    summary="Get DCLC-RAG system status and deduplication metrics",
    description="Check the active LLM provider, dense model, indexed chunks, and deduplication savings.",
)
async def get_rag_status() -> RAGStatusResponse:
    """Return operational status and deduplication statistics."""
    return rag_service.get_status()


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Ask a grounded question with optional version scoping",
    description=(
        "Retrieve relevant chunks via hybrid search, enforce strict version isolation, "
        "and generate a grounded answer with validated citations."
    ),
)
async def query_rag(request: RAGQueryRequest) -> RAGQueryResponse:
    """Execute grounded question answering with verified citations."""
    return rag_service.answer_query(request)
