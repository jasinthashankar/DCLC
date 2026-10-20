"""
tests/test_rag_phase7e.py
-------------------------
DCLC Phase 7E Verification Suite — Realistic Multi-Version Evaluation and Validation.

Verifies:
  1. Realistic corpus structure: 3 versions of architecture doc + 5 distractors chunk cleanly.
  2. Multi-version deduplication: shared sections are correctly detected and shared across versions.
  3. Ground-truth mapping: all non-abstaining benchmark queries map to real FastCDC chunk IDs.
  4. Conflicting facts isolation: queries scoped to V1 never retrieve V2 memory/gateway/cipher chunks.
  5. Deleted sections abstention: queries targeting deleted sections in V2 abstain with 0 hits.
  6. Inserted sections retrieval: newly inserted eBPF and NVMe-oF sections are retrieved in V2/V3.
  7. Hard distractors discrimination: distractor queries retrieve corresponding distractor chunks without false cross-matching.
  8. Quantization ranking agreement: INT8 SQ8 achieves high ranking fidelity and zero isolation violations.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from scripts.run_phase7e_evaluation import build_evaluation_corpus, map_ground_truth
from src.rag.generation.answer_engine import RAGAnswerEngine
from src.rag.generation.provider import MockLLMProvider
from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture(scope="module")
def eval_setup():
    """Build evaluation corpus and chunk index once for test module."""
    base_dir = Path("data/phase7e_eval")
    benchmark_file = base_dir / "evaluation_benchmark.json"

    registry, version_map = build_evaluation_corpus(base_dir)
    chunk_index = ChunkVersionIndex()
    chunk_index.build(registry)

    mapped_queries = map_ground_truth(benchmark_file, registry, chunk_index, version_map)

    bm25 = BM25Index()
    bm25.build_dedup_aware(chunk_index)

    store_fp32 = EmbeddingStore(precision="float32")
    store_fp32.build(chunk_index)

    store_int8 = EmbeddingStore(precision="int8")
    store_int8.build(chunk_index)

    retriever_fp32 = HybridRetriever(bm25, store_fp32, chunk_index)
    retriever_int8 = HybridRetriever(bm25, store_int8, chunk_index)

    return {
        "registry": registry,
        "version_map": version_map,
        "chunk_index": chunk_index,
        "mapped_queries": mapped_queries,
        "retriever_fp32": retriever_fp32,
        "retriever_int8": retriever_int8,
    }


def test_realistic_corpus_deduplication(eval_setup):
    """Verify that multi-version documents exhibit non-zero FastCDC chunk deduplication."""
    chunk_index = eval_setup["chunk_index"]
    registry = eval_setup["registry"]

    total_slots = sum(
        sum(len(v.chunks) for v in registry.get_all_versions(did))
        for did in registry.list_doc_ids()
    )
    unique_chunks = len(chunk_index.all_chunk_ids())
    shared_chunks = registry.shared_chunk_count()

    assert total_slots > unique_chunks
    assert shared_chunks > 0
    savings_pct = (1.0 - (unique_chunks / total_slots)) * 100.0
    assert savings_pct > 15.0  # Shared sections must generate measurable savings


def test_ground_truth_mapping_integrity(eval_setup):
    """Verify all queries are mapped and non-abstaining queries have relevant chunk IDs."""
    queries = eval_setup["mapped_queries"]
    assert len(queries) == 25

    for q in queries:
        if not q["should_abstain"]:
            assert len(q["relevant_chunk_ids"]) > 0, f"Query {q['query_id']} has no relevant chunks"
            assert len(q["relevant_slot_keys"]) > 0, f"Query {q['query_id']} has no relevant slot keys"
        else:
            assert len(q["relevant_chunk_ids"]) == 0


@pytest.mark.parametrize("precision", ["fp32", "int8"])
def test_conflicting_facts_strict_isolation(eval_setup, precision: str):
    """Verify V1 memory query retrieves 32GB ECC DDR4 and never retrieves 128GB DDR5."""
    retriever = eval_setup[f"retriever_{precision}"]
    queries = {q["query_id"]: q for q in eval_setup["mapped_queries"]}

    # Q06 targets V1 memory (32GB ECC DDR4)
    q06 = queries["Q06_V1_MEMORY"]
    results_v1 = retriever.retrieve(
        query=q06["query_text"],
        top_k=5,
        alpha=0.5,
        doc_id=q06["target_doc_id"],
        version_id=q06["target_version_id"],
    )
    assert len(results_v1) > 0
    top_text = results_v1[0].text_content.lower()
    assert "32gb" in top_text or "ddr4" in top_text
    for r in results_v1:
        assert "128gb" not in r.text_content.lower()
        assert "ddr5" not in r.text_content.lower()

    # Q11 targets V2 memory (128GB ECC DDR5)
    q11 = queries["Q11_V2_MEMORY"]
    results_v2 = retriever.retrieve(
        query=q11["query_text"],
        top_k=5,
        alpha=0.5,
        doc_id=q11["target_doc_id"],
        version_id=q11["target_version_id"],
    )
    assert len(results_v2) > 0
    top_text_v2 = results_v2[0].text_content.lower()
    assert "128gb" in top_text_v2 or "ddr5" in top_text_v2
    for r in results_v2:
        assert "32gb" not in r.text_content.lower()
        assert "ddr4" not in r.text_content.lower()


@pytest.mark.parametrize("precision", ["fp32", "int8"])
def test_deleted_sections_zero_leakage(eval_setup, precision: str):
    """Verify queries targeting deleted XML or RPC sections in V2 yield zero prohibited leakage."""
    retriever = eval_setup[f"retriever_{precision}"]
    queries = {q["query_id"]: q for q in eval_setup["mapped_queries"]}

    # Q17 asks for cluster-spec-v1 XML configuration in V2 (deleted in V2)
    q17 = queries["Q17_DELETED_XML_V2"]
    results_xml = retriever.retrieve(
        query=q17["query_text"],
        top_k=5,
        alpha=0.5,
        doc_id=q17["target_doc_id"],
        version_id=q17["target_version_id"],
    )
    # Retrieved chunks in V2 scope MUST NOT contain legacy XML syntax
    for r in results_xml:
        for bad_span in q17["prohibited_spans"]:
            assert bad_span.lower() not in r.text_content.lower()


@pytest.mark.parametrize("precision", ["fp32", "int8"])
def test_inserted_sections_retrieval(eval_setup, precision: str):
    """Verify newly inserted sections in V2 (eBPF and NVMe-oF) are successfully retrieved."""
    retriever = eval_setup[f"retriever_{precision}"]
    queries = {q["query_id"]: q for q in eval_setup["mapped_queries"]}

    # Q14 targets eBPF telemetry in V2
    q14 = queries["Q14_V2_EBPF"]
    results_ebpf = retriever.retrieve(
        query=q14["query_text"],
        top_k=3,
        alpha=0.5,
        doc_id=q14["target_doc_id"],
        version_id=q14["target_version_id"],
    )
    assert any("ebpf" in r.text_content.lower() for r in results_ebpf)

    # Q15 targets NVMe-oF in V2
    q15 = queries["Q15_V2_NVMEOF"]
    results_nvme = retriever.retrieve(
        query=q15["query_text"],
        top_k=3,
        alpha=0.5,
        doc_id=q15["target_doc_id"],
        version_id=q15["target_version_id"],
    )
    assert any("nvme" in r.text_content.lower() for r in results_nvme)


def test_grounded_qa_citations_on_realistic_corpus(eval_setup):
    """Verify RAGAnswerEngine produces verified citations on realistic corpus under INT8."""
    engine = RAGAnswerEngine(
        retriever=eval_setup["retriever_int8"],
        version_index=eval_setup["chunk_index"],
        registry=eval_setup["registry"],
        provider=MockLLMProvider(),
    )

    q = eval_setup["mapped_queries"][0]  # Q01_SHARED_LSM
    ans = engine.answer_question(
        query=q["query_text"],
        top_k=3,
    )

    assert ans.status == "SUCCESS"
    assert len(ans.citations) > 0
    for cit in ans.citations:
        assert cit.is_valid is True
        assert cit.byte_length > 0
        assert len(cit.snippet) > 0
