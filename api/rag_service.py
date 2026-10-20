"""
api/rag_service.py
------------------
Service layer coordinating DCLC-RAG document ingestion, multi-version mapping,
deduplication-aware hybrid retrieval, and grounded LLM question answering (Phase 6E).
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException, status

from api.rag_schemas import (
    RAGChunkInfo,
    RAGDocumentSummary,
    RAGDocumentsResponse,
    RAGIngestResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGStatusResponse,
    RAGVerifiedCitation,
    RAGVersionSummary,
)
from src.rag.document_parser import UnsupportedFileError
from src.rag.generation.answer_engine import RAGAnswer, RAGAnswerEngine
from src.rag.generation.provider import LLMProvider, get_llm_provider
from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.version_mapper import DocumentVersion, VersionRegistry

logger = logging.getLogger(__name__)


class RAGBackendService:
    """
    Thread-safe backend service maintaining in-memory state for DCLC-RAG.

    Coordinates:
    - VersionRegistry (Phase 6B document parsing and version tracking)
    - ChunkVersionIndex (Phase 6C deduplication-aware reverse index)
    - BM25Index (sparse keyword retrieval)
    - EmbeddingStore (dense semantic retrieval with shared vector reuse)
    - HybridRetriever (convex combination BM25 + Dense)
    - RAGAnswerEngine (Phase 6D 10-stage grounded question answering)
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        precision: Optional[str] = None,
    ) -> None:
        self._lock = threading.Lock()
        self.registry = VersionRegistry()
        self.chunk_index = ChunkVersionIndex()
        self.bm25_index = BM25Index()
        self.embedding_store = EmbeddingStore(precision=precision)
        self.retriever: Optional[HybridRetriever] = None
        self.provider = provider or get_llm_provider()
        self.engine: Optional[RAGAnswerEngine] = None

    def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        doc_id: Optional[str] = None,
        min_size: int = 64,
        avg_size: int = 256,
        max_size: int = 1024,
    ) -> RAGIngestResponse:
        """
        Ingest a document file (.txt, .md, .py) into the DCLC-RAG knowledge base.

        If doc_id is provided, this creates a new version revision for that existing document.
        If doc_id is None, a new logical document is registered based on the filename.
        """
        # Validate file extension
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".md", ".py"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '{suffix}'. Supported extensions: .txt, .md, .py",
            )

        with self._lock:
            # Check existing document if doc_id was specified
            target_filename = Path(filename).name
            is_new_doc = True

            if doc_id is not None:
                if doc_id not in self.registry.list_doc_ids():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Document ID '{doc_id}' not found in registry.",
                    )
                existing_versions = self.registry.get_all_versions(doc_id)
                target_filename = existing_versions[0].filename
                is_new_doc = False

            # Ingest via temporary file so FastCDC and document parser can operate on exact bytes
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / target_filename
                tmp_path.write_bytes(file_bytes)

                try:
                    # Ingest document version
                    version: DocumentVersion = self.registry.ingest_document(
                        tmp_path,
                        min_size=min_size,
                        avg_size=avg_size,
                        max_size=max_size,
                    )
                except UnsupportedFileError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=str(exc),
                    )

            # Check if this exact content version was already ingested
            actual_doc_id = version.doc_id
            version_id = version.version_id
            versions_for_doc = self.registry.get_all_versions(actual_doc_id)
            is_duplicate_version = len(versions_for_doc) > 1 and versions_for_doc[-1].version_id != version_id

            # Rebuild reverse chunk index
            self.chunk_index.build(self.registry)

            # Rebuild retrieval indices
            self.bm25_index.build_dedup_aware(self.chunk_index)
            self.embedding_store.build(self.chunk_index)

            # Rebuild retriever and answering engine
            self.retriever = HybridRetriever(self.bm25_index, self.embedding_store, self.chunk_index)
            self.engine = RAGAnswerEngine(
                retriever=self.retriever,
                version_index=self.chunk_index,
                registry=self.registry,
                provider=self.provider,
            )

            shared_in_version = sum(1 for c in version.chunks if c.is_shared)

            msg = (
                f"Ingested {version.filename} version {version.version_id[:8]} "
                f"({len(version.chunks)} chunks, {shared_in_version} shared)."
            )

            return RAGIngestResponse(
                doc_id=version.doc_id,
                version_id=version.version_id,
                filename=version.filename,
                extension=version.extension,
                content_sha256=version.content_sha256,
                chunk_count=len(version.chunks),
                total_bytes=version.total_bytes,
                shared_chunks_in_version=shared_in_version,
                is_new_document=is_new_doc,
                is_duplicate_version=is_duplicate_version,
                message=msg,
            )

    def list_documents(self) -> RAGDocumentsResponse:
        """Return all registered documents, versions, and global chunk deduplication counts."""
        with self._lock:
            doc_summaries: List[RAGDocumentSummary] = []
            total_versions = 0

            for doc_id in self.registry.list_doc_ids():
                versions = self.registry.get_all_versions(doc_id)
                total_versions += len(versions)

                v_summaries = [
                    RAGVersionSummary(
                        version_id=v.version_id,
                        doc_id=v.doc_id,
                        filename=v.filename,
                        extension=v.extension,
                        content_sha256=v.content_sha256,
                        chunk_count=len(v.chunks),
                        total_bytes=v.total_bytes,
                    )
                    for v in versions
                ]

                first_v = versions[0]
                doc_summaries.append(
                    RAGDocumentSummary(
                        doc_id=doc_id,
                        filename=first_v.filename,
                        extension=first_v.extension,
                        version_count=len(versions),
                        versions=v_summaries,
                    )
                )

            total_unique_chunks = len(self.chunk_index.all_chunk_ids())
            shared_chunks = self.registry.shared_chunk_count()

            return RAGDocumentsResponse(
                documents=doc_summaries,
                total_documents=len(doc_summaries),
                total_versions=total_versions,
                total_unique_chunks=total_unique_chunks,
                shared_chunk_count=shared_chunks,
            )

    def get_status(self) -> RAGStatusResponse:
        """Return operational status, active LLM provider, and vector deduplication ratio."""
        with self._lock:
            total_docs = len(self.registry.list_doc_ids())
            total_versions = sum(
                len(self.registry.get_all_versions(did))
                for did in self.registry.list_doc_ids()
            )
            total_slots = sum(
                sum(len(v.chunks) for v in self.registry.get_all_versions(did))
                for did in self.registry.list_doc_ids()
            )
            unique_chunks = len(self.chunk_index.all_chunk_ids())
            shared_chunks = self.registry.shared_chunk_count()

            dedup_savings_pct = 0.0
            if total_slots > 0:
                dedup_savings_pct = max(0.0, (1.0 - (unique_chunks / total_slots)) * 100.0)

            model_name = getattr(self.provider, "model", "mock-rag-llm")

            return RAGStatusResponse(
                status="ready" if total_docs > 0 else "empty",
                provider_name=self.provider.provider_name,
                model_name=model_name,
                total_documents=total_docs,
                total_versions=total_versions,
                total_unique_chunks=unique_chunks,
                shared_chunk_count=shared_chunks,
                dense_embeddings_count=self.embedding_store.embedding_count,
                dense_dimension=self.embedding_store.embedding_dim,
                dedup_savings_pct=round(dedup_savings_pct, 2),
                embedding_precision=self.embedding_store.precision,
                storage_bytes=self.embedding_store.storage_bytes,
                compression_ratio_vs_fp32=round(self.embedding_store.compression_ratio_vs_fp32, 2),
            )

    def answer_query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """
        Execute grounded RAG question answering scoped to optional doc_id and version_id.
        """
        clean_query = request.query.strip()
        if not clean_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must not be empty.",
            )

        with self._lock:
            # Validate document ID scope if provided
            if request.doc_id is not None:
                if request.doc_id not in self.registry.list_doc_ids():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Requested doc_id '{request.doc_id}' not found.",
                    )

            # Validate version ID scope if provided
            if request.version_id is not None:
                found_version = False
                for did in self.registry.list_doc_ids():
                    try:
                        self.registry.get_version(did, request.version_id)
                        found_version = True
                        break
                    except KeyError:
                        continue
                if not found_version:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Requested version_id '{request.version_id}' not found.",
                    )

            # If no documents are ingested yet, abstain immediately
            if self.engine is None or len(self.registry.list_doc_ids()) == 0:
                return RAGQueryResponse(
                    status="ABSTAIN",
                    query=clean_query,
                    answer="INSUFFICIENT_EVIDENCE: No documents have been ingested into the DCLC-RAG knowledge base yet.",
                    target_doc_id=request.doc_id,
                    target_version_id=request.version_id,
                    citations=[],
                    retrieved_chunk_count=0,
                    used_chunk_count=0,
                    provider_name=self.provider.provider_name,
                    model_name="none",
                    latency_retrieval_ms=0.0,
                    latency_generation_ms=0.0,
                    total_latency_ms=0.0,
                    abstention_reason="Knowledge base is empty",
                )

            # Execute the 10-stage RAG answer engine
            rag_answer: RAGAnswer = self.engine.answer_question(
                query=clean_query,
                doc_id=request.doc_id,
                version_id=request.version_id,
                top_k=request.top_k,
                alpha=request.alpha,
                temperature=request.temperature,
            )

            # Map VerifiedCitation dataclasses to Pydantic responses
            api_citations = [
                RAGVerifiedCitation(
                    tag=c.tag,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    version_id=c.version_id,
                    filename=c.filename,
                    chunk_index=c.chunk_index,
                    byte_offset=c.byte_offset,
                    byte_length=c.byte_length,
                    snippet=c.snippet,
                    is_valid=c.is_valid,
                )
                for c in rag_answer.citations
            ]

            return RAGQueryResponse(
                status=rag_answer.status,
                query=rag_answer.query,
                answer=rag_answer.answer,
                target_doc_id=rag_answer.target_doc_id,
                target_version_id=rag_answer.target_version_id,
                citations=api_citations,
                retrieved_chunk_count=rag_answer.retrieved_chunk_count,
                used_chunk_count=rag_answer.used_chunk_count,
                provider_name=rag_answer.provider_name,
                model_name=rag_answer.model_name,
                latency_retrieval_ms=rag_answer.latency_retrieval_ms,
                latency_generation_ms=rag_answer.latency_generation_ms,
                total_latency_ms=rag_answer.total_latency_ms,
                abstention_reason=rag_answer.abstention_reason,
                invalid_citations=rag_answer.invalid_citations,
            )
