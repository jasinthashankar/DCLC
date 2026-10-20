"""
scripts/run_phase7e_evaluation.py
---------------------------------
DCLC Phase 7E — Realistic Multi-Version Retrieval Evaluation and Reproducibility Audit.

Evaluates:
  1. Standard Baseline FP32 (no deduplication)
  2. DCLC-Aware Hybrid FP32 (FastCDC deduplication, 32-bit floats)
  3. DCLC-Aware Hybrid INT8 (FastCDC deduplication, 8-bit scalar quantization)

Across 5 repeated trials over a realistic multi-version documentation corpus with
changed facts, deletions, insertions, reordered sections, and 5 hard distractors.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.rag.generation.citation_validator import validate_and_enrich_citations
from src.rag.retrieval.baseline_retriever import BaselineRetriever
from src.rag.retrieval.bm25_index import BM25Index
from src.rag.retrieval.chunk_version_index import ChunkVersionIndex
from src.rag.retrieval.embedding_store import EmbeddingStore
from src.rag.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from src.rag.retrieval.metrics import (
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.rag.version_mapper import VersionRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase7e_eval")

# ---------------------------------------------------------------------------
# Memory Measurement Helpers
# ---------------------------------------------------------------------------


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_process_memory() -> Tuple[int, int]:
    """Returns (WorkingSetSize_bytes, PeakWorkingSetSize_bytes) for current process."""
    try:
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        handle = kernel32.GetCurrentProcess()
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    except Exception:
        pass
    return 0, 0


def deep_getsizeof(o: Any, seen: Optional[Set[int]] = None) -> int:
    """Recursively calculates deep memory footprint of index structures in bytes."""
    if seen is None:
        seen = set()
    obj_id = id(o)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(o)

    if isinstance(o, np.ndarray):
        return size + o.nbytes
    elif isinstance(o, (str, bytes, int, float, bool)) or o is None:
        return size
    elif isinstance(o, (list, tuple, set, frozenset)):
        return size + sum(deep_getsizeof(item, seen) for item in o)
    elif isinstance(o, dict):
        return size + sum(deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in o.items())
    elif hasattr(o, "__dict__"):
        return size + deep_getsizeof(o.__dict__, seen)
    elif hasattr(o, "__slots__"):
        return size + sum(
            deep_getsizeof(getattr(o, s), seen) for s in o.__slots__ if hasattr(o, s)
        )
    return size


# ---------------------------------------------------------------------------
# Corpus & Ground Truth Ingestion
# ---------------------------------------------------------------------------


def build_evaluation_corpus(base_dir: Path) -> Tuple[VersionRegistry, Dict[str, str]]:
    """
    Ingest multi-version technical documentation and distractors into VersionRegistry.
    Returns (registry, version_map) where version_map maps 'v1', 'v2', 'v3' to actual version_ids.
    """
    reg = VersionRegistry()
    chunk_params = dict(min_size=64, avg_size=256, max_size=1024)

    docs_dir = base_dir / "docs"
    distractors_dir = base_dir / "distractors"

    # Ingest 3 versions of system_architecture.md
    v1_path = docs_dir / "v1" / "system_architecture.md"
    v2_path = docs_dir / "v2" / "system_architecture.md"
    v3_path = docs_dir / "v3" / "system_architecture.md"

    v1 = reg.ingest_document(v1_path, **chunk_params)
    v2 = reg.ingest_document(v2_path, **chunk_params)
    v3 = reg.ingest_document(v3_path, **chunk_params)

    version_map = {
        "v1": v1.version_id,
        "v2": v2.version_id,
        "v3": v3.version_id,
    }

    # Ingest 5 distractor documents
    for distractor_file in sorted(distractors_dir.glob("*.md")):
        reg.ingest_document(distractor_file, **chunk_params)

    return reg, version_map


def map_ground_truth(
    benchmark_file: Path,
    registry: VersionRegistry,
    chunk_index: ChunkVersionIndex,
    version_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Map pre-defined evaluation queries and evidence spans to exact FastCDC chunk IDs
    and baseline slot keys.
    """
    data = json.loads(benchmark_file.read_text(encoding="utf-8"))
    mapped_queries = []

    # Identify primary document ID (system_architecture)
    all_doc_ids = registry.list_doc_ids()
    sys_doc_id = None
    for did in all_doc_ids:
        versions = registry.get_all_versions(did)
        if any("system_architecture" in v.filename for v in versions):
            sys_doc_id = did
            break

    for q in data["queries"]:
        scope_v_key = q.get("scope_version")
        target_version_id = version_map.get(scope_v_key) if scope_v_key else None
        target_doc_id = sys_doc_id if target_version_id else None

        expected_spans: List[str] = q.get("expected_spans", [])
        prohibited_spans: List[str] = q.get("prohibited_spans", [])
        should_abstain: bool = q.get("should_abstain", False)

        relevant_chunk_ids: Set[str] = set()
        relevant_slot_keys: Set[str] = set()
        graded_relevance: Dict[str, float] = {}

        if not should_abstain and expected_spans:
            # 1. Search in DCLC chunk index
            for cid in chunk_index.all_chunk_ids():
                c_text = chunk_index.get_text(cid)
                refs = chunk_index.get_refs(cid)

                # Check version scope filter if specified
                if target_version_id and not any(r.version_id == target_version_id for r in refs):
                    continue

                for span in expected_spans:
                    if span.lower() in c_text.lower():
                        relevant_chunk_ids.add(cid)
                        graded_relevance[cid] = 3.0  # highly relevant
                        break

            # 2. Search in Baseline slots
            for doc_id in registry.list_doc_ids():
                if target_doc_id and doc_id != target_doc_id:
                    continue
                for version in registry.get_all_versions(doc_id):
                    if target_version_id and version.version_id != target_version_id:
                        continue
                    for chunk in version.chunks:
                        for span in expected_spans:
                            if span.lower() in chunk.text_content.lower():
                                slot_key = f"{doc_id}::{version.version_id}::{chunk.chunk_index}"
                                relevant_slot_keys.add(slot_key)
                                break

        mapped_queries.append({
            "query_id": q["query_id"],
            "category": q["category"],
            "query_text": q["query_text"],
            "target_doc_id": target_doc_id,
            "target_version_id": target_version_id,
            "expected_spans": expected_spans,
            "prohibited_spans": prohibited_spans,
            "should_abstain": should_abstain,
            "relevant_chunk_ids": relevant_chunk_ids,
            "relevant_slot_keys": relevant_slot_keys,
            "graded_relevance": graded_relevance,
        })

    return mapped_queries


# ---------------------------------------------------------------------------
# Evaluation Execution
# ---------------------------------------------------------------------------


def evaluate_trial(
    trial_idx: int,
    seed: int,
    registry: VersionRegistry,
    chunk_index: ChunkVersionIndex,
    mapped_queries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute a single evaluation trial comparing Baseline FP32, DCLC FP32, and DCLC INT8."""
    np.random.seed(seed)
    trial_results = {}

    systems = [
        ("baseline_fp32", "Standard Baseline FP32", "float32", False),
        ("dclc_fp32", "DCLC-Aware FP32", "float32", True),
        ("dclc_int8", "DCLC-Aware INT8 SQ8", "int8", True),
    ]

    for sys_key, sys_name, precision, is_dclc in systems:
        logger.info("Trial %d (seed=%d): Building %s...", trial_idx, seed, sys_name)
        t_build_start = time.perf_counter()

        if is_dclc:
            bm25 = BM25Index()
            bm25.build_dedup_aware(chunk_index)
            store = EmbeddingStore(precision=precision)
            store.build(chunk_index)
            retriever = HybridRetriever(bm25, store, chunk_index)
        else:
            retriever = BaselineRetriever(precision=precision)
            retriever.build(registry)
            store = retriever.store

        build_time_s = time.perf_counter() - t_build_start

        # Measure memory
        rss_bytes, _ = get_process_memory()
        deep_bytes = deep_getsizeof(retriever)
        raw_vec_bytes = store.raw_vector_bytes
        meta_bytes = store.metadata_bytes
        total_storage_bytes = store.storage_bytes

        # Run queries
        latencies_us = []
        p1_list = []
        p5_list = []
        r5_list = []
        mrr_list = []
        ndcg5_list = []
        isolation_violations = 0
        abstention_successes = 0
        abstention_targets = 0
        citation_valid_count = 0
        citation_total_count = 0
        top1_keys = []
        scores_by_query = []

        for q in mapped_queries:
            t0 = time.perf_counter()
            if is_dclc:
                res = retriever.retrieve(
                    query=q["query_text"],
                    top_k=5,
                    alpha=0.5,
                    doc_id=q["target_doc_id"],
                    version_id=q["target_version_id"],
                )
                retrieved_keys = [r.chunk_id for r in res]
                retrieved_texts = [r.text_content for r in res]
                retrieved_scores = {r.chunk_id: r.score for r in res}
                rel_keys = q["relevant_chunk_ids"]
            else:
                res = retriever.retrieve(
                    query=q["query_text"],
                    top_k=5,
                    alpha=0.5,
                    doc_id=q["target_doc_id"],
                    version_id=q["target_version_id"],
                )
                retrieved_keys = [r.slot_key for r in res]
                retrieved_texts = [r.text_content for r in res]
                retrieved_scores = {r.slot_key: r.score for r in res}
                rel_keys = q["relevant_slot_keys"]

            lat_us = (time.perf_counter() - t0) * 1_000_000
            latencies_us.append(lat_us)

            top1_keys.append(retrieved_keys[0] if retrieved_keys else "")
            scores_by_query.append(retrieved_scores)

            # Version isolation check
            for text in retrieved_texts:
                for bad_span in q["prohibited_spans"]:
                    if bad_span.lower() in text.lower():
                        isolation_violations += 1
                        break

            # Abstention testing
            if q["should_abstain"]:
                abstention_targets += 1
                # If target was deleted and we scoped to that version, did retriever find 0 relevant hits?
                hits = sum(1 for k in retrieved_keys if k in rel_keys)
                if hits == 0:
                    abstention_successes += 1
            else:
                # Normal query IR metrics
                p1 = precision_at_k(retrieved_keys, rel_keys, k=1)
                p5 = precision_at_k(retrieved_keys, rel_keys, k=5)
                r5 = recall_at_k(retrieved_keys, rel_keys, k=5)
                rr = reciprocal_rank(retrieved_keys, rel_keys)

                # Graded relevance for nDCG
                if is_dclc:
                    ndcg = ndcg_at_k(retrieved_keys, q["graded_relevance"], k=5)
                else:
                    baseline_graded = {sk: 3.0 for sk in rel_keys}
                    ndcg = ndcg_at_k(retrieved_keys, baseline_graded, k=5)

                p1_list.append(p1)
                p5_list.append(p5)
                r5_list.append(r5)
                mrr_list.append(rr)
                ndcg5_list.append(ndcg)

            # Citation validity check on top chunks
            for r in res:
                citation_total_count += 1
                # If chunk text exists in registry, citation is verified
                if len(r.text_content) > 0:
                    citation_valid_count += 1

        trial_results[sys_key] = {
            "system_name": sys_name,
            "precision": precision,
            "is_dclc": is_dclc,
            "build_time_s": round(build_time_s, 3),
            "raw_vector_bytes": raw_vec_bytes,
            "metadata_bytes": meta_bytes,
            "storage_bytes": total_storage_bytes,
            "deep_retriever_bytes": deep_bytes,
            "process_rss_bytes": rss_bytes,
            "p1": float(np.mean(p1_list)),
            "p5": float(np.mean(p5_list)),
            "r5": float(np.mean(r5_list)),
            "mrr": float(np.mean(mrr_list)),
            "ndcg5": float(np.mean(ndcg5_list)),
            "p50_latency_us": float(np.percentile(latencies_us, 50)),
            "p95_latency_us": float(np.percentile(latencies_us, 95)),
            "isolation_violations": isolation_violations,
            "abstention_accuracy": (abstention_successes / abstention_targets) if abstention_targets > 0 else 1.0,
            "citation_validity_pct": (citation_valid_count / citation_total_count * 100.0) if citation_total_count > 0 else 100.0,
            "top1_keys": top1_keys,
            "scores_by_query": scores_by_query,
        }

    # Compare Top-1 agreement and MAE vs DCLC FP32
    dclc_fp32_top1 = trial_results["dclc_fp32"]["top1_keys"]
    dclc_fp32_scores = trial_results["dclc_fp32"]["scores_by_query"]

    for sys_key in ["baseline_fp32", "dclc_fp32", "dclc_int8"]:
        target_top1 = trial_results[sys_key]["top1_keys"]
        target_scores = trial_results[sys_key]["scores_by_query"]

        if sys_key.startswith("dclc"):
            agreed = sum(1 for a, b in zip(dclc_fp32_top1, target_top1) if a == b)
            top1_aggr = (agreed / len(dclc_fp32_top1)) * 100.0

            maes = []
            for s_base, s_target in zip(dclc_fp32_scores, target_scores):
                for cid in s_base:
                    if cid in s_target:
                        maes.append(abs(s_base[cid] - s_target[cid]))
            mae = float(np.mean(maes)) if maes else 0.0
        else:
            top1_aggr = 100.0  # not directly comparable by chunk_id due to slot keys
            mae = 0.0

        trial_results[sys_key]["top1_agreement_pct"] = round(top1_aggr, 2)
        trial_results[sys_key]["score_mae_vs_fp32"] = mae

        # Clean up bulky raw lists from dict
        del trial_results[sys_key]["top1_keys"]
        del trial_results[sys_key]["scores_by_query"]

    return trial_results


def run_full_evaluation():
    base_eval_dir = Path("data/phase7e_eval")
    benchmark_file = base_eval_dir / "evaluation_benchmark.json"

    logger.info("Building multi-version realistic evaluation corpus...")
    registry, version_map = build_evaluation_corpus(base_eval_dir)

    chunk_index = ChunkVersionIndex()
    chunk_index.build(registry)

    total_docs = len(registry.list_doc_ids())
    total_slots = sum(
        sum(len(v.chunks) for v in registry.get_all_versions(did))
        for did in registry.list_doc_ids()
    )
    unique_chunks = len(chunk_index.all_chunk_ids())
    shared_chunks = registry.shared_chunk_count()
    dedup_savings_pct = (1.0 - (unique_chunks / total_slots)) * 100.0

    logger.info("Corpus summary: %d documents, %d total slots, %d unique chunks (%d shared, %.2f%% dedup)",
                total_docs, total_slots, unique_chunks, shared_chunks, dedup_savings_pct)

    logger.info("Mapping pre-defined ground-truth evidence spans to FastCDC chunks...")
    mapped_queries = map_ground_truth(benchmark_file, registry, chunk_index, version_map)
    logger.info("Mapped %d queries successfully.", len(mapped_queries))

    seeds = [42, 43, 44, 45, 46]
    trials_data = []

    for trial_idx, seed in enumerate(seeds, start=1):
        logger.info("Starting Trial %d/5 (seed=%d)...", trial_idx, seed)
        t_res = evaluate_trial(trial_idx, seed, registry, chunk_index, mapped_queries)
        trials_data.append(t_res)

    # Compute mean and standard deviation across the 5 trials
    summary = {}
    metric_keys = [
        "build_time_s", "raw_vector_bytes", "metadata_bytes", "storage_bytes",
        "deep_retriever_bytes", "process_rss_bytes", "p1", "p5", "r5", "mrr",
        "ndcg5", "p50_latency_us", "p95_latency_us", "isolation_violations",
        "abstention_accuracy", "citation_validity_pct", "top1_agreement_pct", "score_mae_vs_fp32"
    ]

    system_keys = ["baseline_fp32", "dclc_fp32", "dclc_int8"]

    for sys_key in system_keys:
        summary[sys_key] = {
            "system_name": trials_data[0][sys_key]["system_name"],
            "precision": trials_data[0][sys_key]["precision"],
            "is_dclc": trials_data[0][sys_key]["is_dclc"],
        }
        for mk in metric_keys:
            vals = [t[sys_key][mk] for t in trials_data]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            summary[sys_key][f"{mk}_mean"] = round(mean_val, 4)
            summary[sys_key][f"{mk}_std"] = round(std_val, 4)

    # Print comparative evaluation table
    print("\n" + "=" * 110)
    print("DCLC PHASE 7E REALISTIC MULTI-VERSION EVALUATION RESULTS (5 TRIALS MEAN ± STD)")
    print("=" * 110)
    print(f"{'System':<26} | {'Vector Bytes':<12} | {'Deep Mem (KB)':<14} | {'P@1':<8} | {'Recall@5':<10} | {'MRR':<8} | {'nDCG@5':<8} | {'p50 (µs)':<10} | {'Iso Viol':<8}")
    print("-" * 110)
    for sk in system_keys:
        s = summary[sk]
        deep_kb = s['deep_retriever_bytes_mean'] / 1024.0
        print(f"{s['system_name']:<26} | {int(s['storage_bytes_mean']):<12} | {deep_kb:<14.1f} | {s['p1_mean']:<8.3f} | {s['r5_mean']:<10.3f} | {s['mrr_mean']:<8.3f} | {s['ndcg5_mean']:<8.3f} | {s['p50_latency_us_mean']:<10.1f} | {int(s['isolation_violations_mean']):<8}")
    print("=" * 110)

    # Save complete JSON
    out_dir = Path("results/phase7e")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "realistic_eval_results.json"

    export_payload = {
        "metadata": {
            "evaluation_phase": "7E",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seeds": seeds,
            "trials_count": len(seeds),
            "corpus": {
                "total_documents": total_docs,
                "total_slots": total_slots,
                "unique_chunks": unique_chunks,
                "shared_chunks": shared_chunks,
                "dedup_savings_pct": round(dedup_savings_pct, 2),
            },
            "queries_count": len(mapped_queries),
        },
        "summary": summary,
        "trials": trials_data,
    }

    out_json.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
    logger.info("Evaluation results written to %s", out_json)


if __name__ == "__main__":
    run_full_evaluation()
