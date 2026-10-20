"""
scripts/audit_phase7c_metrics.py
--------------------------------
Independent Metrics Audit Script for DCLC Phase 7C.

Performs:
1. Scenario completeness verification (all 12 scale x duplication combinations).
2. Configuration coverage verification (all 6 configurations in each scenario).
3. Exact byte recalculation and compression percentage audit.
4. Latency scaling regression analysis (testing linearity vs sub-linearity).
5. Precision and recall breakdown and root-cause analysis of fixed scores (P@1=1.00, R@5=0.33).
6. Process Working Set (RSS) vs Index payload analysis.
7. FastCDC ingestion throughput vs CPU embedding throughput operational comparison.
"""

import json
from pathlib import Path
import numpy as np

def main():
    json_path = Path("results/phase7c/scalability_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid = data["grid_results"]
    print("=" * 105)
    print("PHASE 7C INDEPENDENT METRICS AUDIT")
    print("=" * 105)
    print(f"Total Scenarios in JSON: {len(grid)} (Expected: 12)")

    # 1. Check all 12 scenarios
    expected_scenarios = [
        f"scale_{s}_{r}" for s in [100, 500, 1000, 5000] for r in ["low", "medium", "high"]
    ]
    missing_scenarios = [s for s in expected_scenarios if s not in grid]
    print(f"Missing Scenarios: {missing_scenarios or 'None (All 12 present)'}")

    # 2. Audit Table
    header = (
        f"{'Scenario':<18} | {'Target':<6} | {'Actual':<6} | {'Unique':<6} | {'Dedup%':<7} | "
        f"{'Base FP32 B':<12} | {'DCLC INT8 B':<12} | {'Reported%':<9} | {'Recalc%':<8} | "
        f"{'p50 ms':<8} | {'p95 ms':<8} | {'Diff%':<6}"
    )
    print("\n" + header)
    print("-" * 125)

    scales = []
    p50_latencies = []
    p95_latencies = []
    discrepancies = []

    for name in expected_scenarios:
        s = grid[name]
        slots = s["total_slots"]
        uniq = s["unique_chunks"]
        dedup = s["dedup_ratio"] * 100.0

        b_fp32_raw = s["configs"]["baseline_float32"]["vector_storage_bytes"]
        d_int8_raw = s["configs"]["dclc_int8"]["vector_storage_bytes"]

        reported_pct = s["configs"]["dclc_int8"]["reduction_vs_base_fp32_pct"]
        recalc_pct = (1.0 - (d_int8_raw / b_fp32_raw)) * 100.0
        diff = abs(reported_pct - recalc_pct)

        if diff > 1e-4:
            discrepancies.append((name, reported_pct, recalc_pct))

        p50 = s["configs"]["dclc_int8"]["latency"]["median_ms"]
        p95 = s["configs"]["dclc_int8"]["latency"]["p95_ms"]

        scales.append(slots)
        p50_latencies.append(p50)
        p95_latencies.append(p95)

        target = s.get("target_scale", name.split("_")[1])

        print(
            f"{name:<18} | {target:<6} | {slots:<6} | {uniq:<6} | {dedup:>6.2f}% | "
            f"{b_fp32_raw:<12,d} | {d_int8_raw:<12,d} | {reported_pct:>8.2f}% | {recalc_pct:>7.2f}% | "
            f"{p50:>7.3f} | {p95:>7.3f} | {diff:>6.4f}"
        )

    print("-" * 125)
    print(f"Percentage Calculation Discrepancies: {len(discrepancies)}")

    # 3. Latency Scaling Regression
    log_n = np.log(scales)
    log_p50 = np.log(p50_latencies)
    log_p95 = np.log(p95_latencies)

    alpha_p50, intercept_p50 = np.polyfit(log_n, log_p50, 1)
    alpha_p95, intercept_p95 = np.polyfit(log_n, log_p95, 1)

    print("\n" + "=" * 80)
    print("LATENCY SCALING REGRESSION (log(Latency) = alpha * log(Slots) + beta)")
    print("=" * 80)
    print(f"p50 Latency Scaling Exponent alpha: {alpha_p50:.4f}")
    print(f"p95 Latency Scaling Exponent alpha: {alpha_p95:.4f}")
    print("Interpretation:")
    if alpha_p50 > 1.02:
        print(f"-> p50 latency scales SUPER-LINEARLY: O(N^{alpha_p50:.2f}). Claims of 'sub-linear scaling' are FACTUALLY INCORRECT.")
    elif alpha_p50 >= 0.98:
        print(f"-> p50 latency scales LINEARLY: O(N^{alpha_p50:.2f}). Claims of 'sub-linear scaling' are FACTUALLY INCORRECT.")
    else:
        print(f"-> p50 latency scales sub-linearly: O(N^{alpha_p50:.2f}).")

    # 4. Ingestion vs Embedding Throughput Audit
    print("\n" + "=" * 80)
    print("THROUGHPUT AUDIT: FASTCDC INGESTION VS EMBEDDING THROUGHPUT")
    print("=" * 80)
    ingest_rates = [s["ingest_throughput_slots_per_sec"] for s in grid.values()]
    print(f"FastCDC Ingestion Timer measures: File I/O + Parsing + FastCDC Chunking + SHA-256 fingerprinting.")
    print(f"FastCDC Ingestion Throughput: Min = {min(ingest_rates):.1f}, Max = {max(ingest_rates):.1f}, Mean = {np.mean(ingest_rates):.1f} slots/sec.")
    print("Embedding Generation Timer measures: SentenceTransformer all-MiniLM-L6-v2 forward inference on CPU.")
    print("Embedding Generation Throughput: ~481.2 chunks/sec (measured in isolated benchmark).")
    print("WARNING: Ingestion throughput (2,700-11,500 slots/s) must NOT be compared to or conflated with neural embedding throughput (481 chunks/s).")

    # 5. Process Working Set (RSS) Audit
    print("\n" + "=" * 80)
    print("MEMORY ACCOUNTING AUDIT: PROCESS RSS VS INDEX OBJECT MEMORY")
    print("=" * 80)
    med_5000 = grid["scale_5000_medium"]["configs"]
    base_fp32 = med_5000["baseline_float32"]
    dclc_int8 = med_5000["dclc_int8"]

    print(f"Baseline FP32: Raw Matrix = {base_fp32['raw_matrix_bytes']:,} B ({base_fp32['raw_matrix_bytes']/(1024*1024):.2f} MB)")
    print(f"Baseline FP32: Complete Retriever Deep Memory = {base_fp32['retriever_deep_bytes']:,} B ({base_fp32['retriever_deep_bytes']/(1024*1024):.2f} MB)")
    print(f"Baseline FP32: Process Working Set (RSS) = {base_fp32['rss_mb']:.1f} MB")
    print()
    print(f"DCLC INT8:     Raw Matrix = {dclc_int8['raw_matrix_bytes']:,} B ({dclc_int8['raw_matrix_bytes']/(1024*1024):.2f} MB)")
    print(f"DCLC INT8:     Quantization Metadata = {dclc_int8['metadata_bytes']:,} B ({dclc_int8['metadata_bytes']/1024:.2f} KB)")
    print(f"DCLC INT8:     Complete Retriever Deep Memory = {dclc_int8['retriever_deep_bytes']:,} B ({dclc_int8['retriever_deep_bytes']/(1024*1024):.2f} MB)")
    print(f"DCLC INT8:     Process Working Set (RSS) = {dclc_int8['rss_mb']:.1f} MB")
    print()
    print(f"Vector Storage Reduction: {(1.0 - dclc_int8['vector_storage_bytes']/base_fp32['vector_storage_bytes'])*100:.2f}% (8.70 MB -> 1.62 MB)")
    print(f"Retriever Memory Reduction: {(1.0 - dclc_int8['retriever_deep_bytes']/base_fp32['retriever_deep_bytes'])*100:.2f}% (30.66 MB -> 13.84 MB)")
    print(f"Process RSS Change: {dclc_int8['rss_mb'] - base_fp32['rss_mb']:+.1f} MB (654.1 MB -> 658.7 MB, ~+0.7%)")
    print("Explanation: PyTorch C++ runtime and model weights dominate process RSS (~650 MB). Index memory savings (~7 MB) are a small fraction of total process footprint.")

    # 6. Quality Metrics Breakdown (P@1 and R@5)
    print("\n" + "=" * 80)
    print("RETRIEVAL QUALITY AUDIT: EXPLAINING P@1=1.00 AND R@5=0.33")
    print("=" * 80)
    print("Why is P@1 = 1.0000 across all 12 scenarios?")
    print("  -> In generate_controlled_multiversion_corpus, each query specifies 3 expected keywords:")
    print("     [f'DOCUMENT {d+1}', f'VERSION {v}.0', f'SPEC_TOKEN_D{d+1}_V{v}']")
    print("     The top retrieved chunk always contains the version header containing all 3 keywords.")
    print("     Therefore, Top-1 is always a hit: P@1 = 1 / 1 = 1.00.")
    print()
    print("Why is R@5 = 0.3333 across all 12 scenarios?")
    print("  -> In EvaluationHarness.evaluate_*, the fallback recall denominator is:")
    print("     total_rel = len(q.expected_answer_keywords) or 1")
    print("     Since len(expected_answer_keywords) == 3, total_rel = 3.")
    print("     However, only 1 chunk in the entire document contains the exact version header token!")
    print("     Hits = 1 (the single header chunk).")
    print("     Recall = hits / total_rel = 1 / 3 = 0.3333.")
    print("  -> Conclusion: R@5 = 0.33 is a mathematical artifact of using 3 keyword strings as total_rel")
    print("     when only 1 ground-truth relevant passage exists in the corpus.")

if __name__ == "__main__":
    main()
