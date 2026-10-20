"""
src/rag/generation/answer_engine.py
------------------------------------
DCLC-RAG Phase 6D — Ten-Stage Grounded Question Answering Engine.

Orchestrates:
1. Query & version scope receipt.
2. Hybrid retrieval (BM25 + Dense Cosine).
3. Pre-generation version filtering (strict version isolation).
4. Text resolution via ChunkVersionIndex.
5. Fingerprint verification (SHA-256 integrity).
6. Anti-injection bounded context assembly.
7. Grounded LLM generation (temperature=0.0).
8. Source metadata attachment.
9. Citation validation and enrichment.
10. Abstention policy handling.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.generation.citation_validator import (
    CitationValidationReport,
    VerifiedCitation,
    validate_and_enrich_citations,
)
from src.rag.generation.prompt_template import (
    SYSTEM_PROMPT,
    FormattedEvidenceChunk,
    format_evidence_packet,
)
from src.rag.generation.provider import (
    LLMProvider,
    LLMResponse,
    get_llm_provider,
)
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from src.rag.version_mapper import VersionRegistry


@dataclass
class RAGAnswer:
    """Final output from the RAG Question Answering Engine."""
    status: str                         # "SUCCESS", "ABSTAIN", "ERROR"
    query: str
    answer: str
    target_doc_id: Optional[str]
    target_version_id: Optional[str]
    citations: List[VerifiedCitation]
    retrieved_chunk_count: int
    used_chunk_count: int
    provider_name: str
    model_name: str
    latency_retrieval_ms: float
    latency_generation_ms: float
    total_latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    fingerprints_verified: bool = True
    abstention_reason: Optional[str] = None
    invalid_citations: List[str] = field(default_factory=list)


class RAGAnswerEngine:
    """
    Orchestrates the 10-stage RAG answer generation pipeline.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        version_index: ChunkVersionIndex,
        registry: Optional[VersionRegistry] = None,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        self.retriever = retriever
        self.version_index = version_index
        self.registry = registry
        self.provider = provider or get_llm_provider()

    def answer_question(
        self,
        query: str,
        doc_id: Optional[str] = None,
        version_id: Optional[str] = None,
        top_k: int = 5,
        alpha: float = 0.5,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> RAGAnswer:
        """
        Execute the 10-stage question-answering pipeline.
        """
        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 1: Validate input
        # ------------------------------------------------------------------
        clean_query = query.strip()
        if not clean_query:
            return RAGAnswer(
                status="ABSTAIN",
                query=query,
                answer="No query provided.",
                target_doc_id=doc_id,
                target_version_id=version_id,
                citations=[],
                retrieved_chunk_count=0,
                used_chunk_count=0,
                provider_name=self.provider.provider_name,
                model_name="none",
                latency_retrieval_ms=0.0,
                latency_generation_ms=0.0,
                total_latency_ms=0.0,
                abstention_reason="Empty query",
            )

        # ------------------------------------------------------------------
        # Stage 2 & 3: Hybrid Retrieval with Version Filtering
        # ------------------------------------------------------------------
        t_ret_0 = time.perf_counter()
        retrieval_results: List[RetrievalResult] = self.retriever.retrieve(
            query=clean_query,
            top_k=top_k,
            alpha=alpha,
            doc_id=doc_id,
            version_id=version_id,
        )
        latency_retrieval_ms = (time.perf_counter() - t_ret_0) * 1000.0

        if not retrieval_results:
            total_lat = (time.perf_counter() - t_start) * 1000.0
            return RAGAnswer(
                status="ABSTAIN",
                query=clean_query,
                answer="INSUFFICIENT_EVIDENCE: No matching document chunks found for the requested version scope.",
                target_doc_id=doc_id,
                target_version_id=version_id,
                citations=[],
                retrieved_chunk_count=0,
                used_chunk_count=0,
                provider_name=self.provider.provider_name,
                model_name="none",
                latency_retrieval_ms=latency_retrieval_ms,
                latency_generation_ms=0.0,
                total_latency_ms=total_lat,
                abstention_reason="No chunks survived version filtering",
            )

        # ------------------------------------------------------------------
        # Stage 4 & 5: Text Resolution & Integrity Fingerprint Verification
        # ------------------------------------------------------------------
        formatted_chunks: List[FormattedEvidenceChunk] = []
        fingerprints_ok = True

        for i, res in enumerate(retrieval_results):
            tag = f"[chunk_{i+1}]"

            # Prefer the matching filtered ref
            ref = res.filtered_refs[0] if res.filtered_refs else res.version_refs[0]

            # Verify byte fingerprint against registry if available
            byte_offset = 0
            byte_len = len(res.text_content.encode("utf-8"))
            if self.registry is not None:
                try:
                    ver = self.registry.get_version(ref.doc_id, ref.version_id)
                    chunk_rec = ver.chunks[ref.chunk_index]
                    computed_sha = hashlib.sha256(chunk_rec.raw_bytes).hexdigest()
                    if computed_sha != res.chunk_id:
                        fingerprints_ok = False
                    byte_offset = chunk_rec.byte_offset
                    byte_len = chunk_rec.byte_length
                except (KeyError, IndexError):
                    pass

            formatted_chunks.append(
                FormattedEvidenceChunk(
                    tag=tag,
                    chunk_id=res.chunk_id,
                    doc_id=ref.doc_id,
                    version_id=ref.version_id,
                    filename=ref.filename,
                    chunk_index=ref.chunk_index,
                    byte_offset=byte_offset,
                    byte_length=byte_len,
                    text_content=res.text_content,
                )
            )

        # ------------------------------------------------------------------
        # Stage 6: Context Assembly with Anti-Injection Framing
        # ------------------------------------------------------------------
        prompt = format_evidence_packet(
            query=clean_query,
            chunks=formatted_chunks,
            target_version_id=version_id,
        )

        # ------------------------------------------------------------------
        # Stage 7: Grounded LLM Generation
        # ------------------------------------------------------------------
        t_gen_0 = time.perf_counter()
        try:
            llm_resp: LLMResponse = self.provider.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            total_lat = (time.perf_counter() - t_start) * 1000.0
            return RAGAnswer(
                status="ERROR",
                query=clean_query,
                answer=f"Provider Error: {exc}",
                target_doc_id=doc_id,
                target_version_id=version_id,
                citations=[],
                retrieved_chunk_count=len(retrieval_results),
                used_chunk_count=0,
                provider_name=self.provider.provider_name,
                model_name="error",
                latency_retrieval_ms=latency_retrieval_ms,
                latency_generation_ms=0.0,
                total_latency_ms=total_lat,
                abstention_reason=str(exc),
            )

        latency_generation_ms = (time.perf_counter() - t_gen_0) * 1000.0

        # ------------------------------------------------------------------
        # Stage 8 & 9: Citation Extraction, Validation, and Enrichment
        # ------------------------------------------------------------------
        val_report: CitationValidationReport = validate_and_enrich_citations(
            generated_text=llm_resp.content,
            evidence_chunks=formatted_chunks,
            target_version_id=version_id,
        )

        # ------------------------------------------------------------------
        # Stage 10: Abstention Policy Check
        # ------------------------------------------------------------------
        raw_answer = llm_resp.content.strip()
        is_abstention = (
            "insufficient_evidence" in raw_answer.lower()
            or not val_report.has_valid_citations
            and "insufficient" in raw_answer.lower()
        )

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        if is_abstention:
            return RAGAnswer(
                status="ABSTAIN",
                query=clean_query,
                answer=raw_answer,
                target_doc_id=doc_id,
                target_version_id=version_id,
                citations=val_report.valid_citations,
                retrieved_chunk_count=len(retrieval_results),
                used_chunk_count=len(val_report.valid_citations),
                provider_name=llm_resp.provider,
                model_name=llm_resp.model,
                latency_retrieval_ms=latency_retrieval_ms,
                latency_generation_ms=latency_generation_ms,
                total_latency_ms=total_latency_ms,
                prompt_tokens=llm_resp.prompt_tokens,
                completion_tokens=llm_resp.completion_tokens,
                fingerprints_verified=fingerprints_ok,
                abstention_reason="Model signaled insufficient or inconclusive evidence",
                invalid_citations=val_report.invalid_citations,
            )

        return RAGAnswer(
            status="SUCCESS",
            query=clean_query,
            answer=raw_answer,
            target_doc_id=doc_id,
            target_version_id=version_id,
            citations=val_report.valid_citations,
            retrieved_chunk_count=len(retrieval_results),
            used_chunk_count=len(val_report.valid_citations),
            provider_name=llm_resp.provider,
            model_name=llm_resp.model,
            latency_retrieval_ms=latency_retrieval_ms,
            latency_generation_ms=latency_generation_ms,
            total_latency_ms=total_latency_ms,
            prompt_tokens=llm_resp.prompt_tokens,
            completion_tokens=llm_resp.completion_tokens,
            fingerprints_verified=fingerprints_ok,
            invalid_citations=val_report.invalid_citations,
        )
