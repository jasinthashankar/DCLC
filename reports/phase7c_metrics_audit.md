# DCLC Phase 7C: Independent Metrics Audit and Verification Report

**Repository:** `C:\Users\jasin\OneDrive\Desktop\DCLC`  
**Phase:** 7C — Scalability, Memory Profiling and Research Validation  
**Date:** 2026-10-18  
**Audit Purpose:** Rigorous verification and recalculation of all Phase 7C benchmark measurements, addressing latency scaling, retrieval quality metrics, throughput operational boundaries, and memory accounting.

---

## 1. Audit Scope & Methodology

This audit independently inspected:
- Raw benchmark dataset: [`results/phase7c/scalability_results.json`](file:///C:/Users/jasin/OneDrive/Desktop/DCLC/results/phase7c/scalability_results.json) (89 KB, all 12 scenarios and 6 configurations)
- Benchmark implementation: [`scripts/benchmark_phase7c_scalability.py`](file:///C:/Users/jasin/OneDrive/Desktop/DCLC/scripts/benchmark_phase7c_scalability.py)
- Evaluation harness: [`src/rag/retrieval/evaluation_harness.py`](file:///C:/Users/jasin/OneDrive/Desktop/DCLC/src/rag/retrieval/evaluation_harness.py)
- Independent verification script: [`scripts/audit_phase7c_metrics.py`](file:///C:/Users/jasin/OneDrive/Desktop/DCLC/scripts/audit_phase7c_metrics.py)
- Full regression suite: 170 unit and integration tests across Phase 6B–E, 7B, and 7C.

No benchmarks were rerun; all recalculations were performed directly on the primary raw JSON measurements.

---

## 2. Check 1: Latency Scaling Audit (Linear vs Sub-Linear)

### Finding & Correction
- **Original Report Claim:** *"Median (p50) query latency scaled sub-linearly from ~0.2 ms at 100 slots to ~13–15 ms at 5,000 slots on CPU."*
- **Audit Outcome:** **CLAIM REFUTED AND CORRECTED.** Latency does **NOT** scale sub-linearly; it scales **super-linearly ($O(N^{1.08})$)**.

### Mathematical Recalculation
We performed an ordinary least-squares log-log linear regression across all 12 benchmark scenarios:
$$\ln(\text{Latency}) = \alpha \cdot \ln(\text{Actual Slots}) + \beta$$

- **p50 Latency Scaling Exponent:** $\alpha = 1.0764$
- **p95 Latency Scaling Exponent:** $\alpha = 1.0423$

| Scale Target | Actual Slots | Mean p50 Latency | Latency Ratio vs 100-slot Base | Slot Ratio vs 100-slot Base |
| :---: | :---: | :---: | :---: | :---: |
| 100 | 125 | 0.237 ms | 1.00x | 1.00x |
| 500 | 588 | 1.098 ms | 4.63x | 4.70x |
| 1,000 | 1,149 | 2.449 ms | 10.33x | 9.19x |
| 5,000 | 5,685 | 14.125 ms | 59.60x | 45.48x |

### Architectural Root Cause
In both `BaselineRetriever` and `HybridRetriever`:
1. Dense retrieval performs an exact matrix-vector multiplication ($M \times D$) over all candidate vectors in NumPy.
2. BM25 performs a dictionary scan over all candidate slot token lists.
Because no approximate nearest neighbor (ANN) index (e.g., HNSW or ScaNN) is used, algorithmic time complexity is strictly $O(N)$. At 5,000+ slots, minor CPU cache-line evictions and Python loop overhead cause the empirical exponent to reach $\alpha \approx 1.08$.

---

## 3. Check 2: Retrieval Quality Audit (P@1=1.00 and R@5=0.33 Root Cause)

### Finding & Explanation
In the report summary, $P@1 = 1.00$ and $R@5 = 0.33$ appeared across all 12 scenarios.

### Why is P@1 = 1.0000?
In `generate_controlled_multiversion_corpus`, queries are synthesized to target specific version tokens:
```python
kw = [f"DOCUMENT {d+1}", f"VERSION {v}.0", ver_token]
```
The first chunk of each ingested document version contains the exact header line:
```text
=== DOCUMENT {d+1} VERSION {v}.0 ({ver_token}) ===
```
Because the query text includes the unique `ver_token`, BM25 and hybrid scoring rank this chunk at rank 1 with 100% precision.

### Why is R@5 = 0.3333?
In `EvaluationHarness.evaluate_dclc` and `evaluate_baseline` ([evaluation_harness.py#L122-L125](file:///C:/Users/jasin/OneDrive/Desktop/DCLC/src/rag/retrieval/evaluation_harness.py#L122-L125)):
```python
total_rel = len(q.relevant_chunks) if q.relevant_chunks else len(q.expected_answer_keywords) or 1
rec = min(1.0, hits / total_rel)
```
When `relevant_chunks` is not explicitly set, the harness uses `len(expected_answer_keywords)` as the denominator.
Because `expected_answer_keywords` contains exactly **3** keyword strings, `total_rel = 3`.
However, only **1** chunk in the document contains that header token.
Therefore, `hits = 1`, and the calculated recall is strictly:
$$\text{Recall} = \frac{\text{hits}}{\text{total\_rel}} = \frac{1}{3} = 0.3333$$

**Audit Conclusion:** $R@5 = 0.3333$ is a mathematical artifact of the evaluation harness fallback formula, not a deficiency in the retriever. In terms of true relevant passage retrieval, 1 out of 1 relevant passage was retrieved (**100% true recall**).

### Top-1 Agreement vs Semantic Discrimination
- **Top-1 Agreement with FP32 (100.0%):** Validates that INT8 SQ8 quantization does not alter the top candidate ranking compared to the FP32 baseline.
- **Evaluation Limitation:** Because queries contain unique version tokens, the test suite verifies version scoping and quantization stability under standard hybrid retrieval, but does **not** evaluate subtle semantic near-neighbor discrimination among thousands of ambiguous distractors.

---

## 4. Check 3: Throughput Audit (FastCDC Ingestion vs Neural Embedding)

### Finding & Boundary Disambiguation
The report previously presented ingestion throughput figures of 2,700–11,500 slots/sec alongside embedding measurements. These timers measure entirely different pipeline operations:

| Operation | Component | Measured Throughput | Measurement Scope |
| :--- | :--- | :---: | :--- |
| **FastCDC Document Ingestion** | `VersionRegistry.ingest_document()` | **2,732 – 11,527 slots/s** (mean 6,700 slots/s) | File I/O, UTF-8 text parsing, FastCDC boundary window chunking, and SHA-256 fingerprinting. |
| **Neural Embedding Generation** | `SentenceTransformer('all-MiniLM-L6-v2')` | **481.2 chunks/s** (batch size 32, CPU) | Deep Transformer forward-pass inference on CPU (384 dimensions). |

**Audit Conclusion:** FastCDC ingestion and neural embedding generation must remain strictly separated in all performance discussions.

---

## 5. Check 4: Memory Accounting Audit (Vector Storage vs Process RSS)

### Breakdown at Maximum Scale (Scale 5,000, Medium Duplication: 5,667 slots, 4,139 unique chunks)

| Memory Metric | Baseline FP32 | Baseline INT8 | DCLC FP32 | DCLC INT8 | Reduction vs Base FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Vector Matrix** | 8,704,512 B | 2,176,128 B | 6,357,504 B | 1,589,376 B | **81.74%** |
| **Quantization Metadata** | 0 B | 45,336 B | 0 B | 33,112 B | N/A (8 B/vec) |
| **Total Stored Vector Bytes** | 8,704,512 B | 2,221,464 B | 6,357,504 B | **1,622,488 B** | **81.36%** |
| **EmbeddingStore Deep Object Memory** | 17,907,097 B | 4,941,222 B | 13,183,852 B | **3,714,041 B** | **79.26%** |
| **Complete Retriever Deep Memory** | 30,660,375 B | 17,694,500 B | 23,305,107 B | **13,835,296 B** | **54.88%** |
| **Process Working Set Size (RSS)** | 654.1 MB | 657.3 MB | 658.3 MB | 658.7 MB | **+0.7% (Unchanged)** |

### Explanation of Process RSS (~655 MB)
- PyTorch C++ runtime (`torch_cpu.dll`, OpenMP, MKL), model weights, tokenizer vocabularies, and Python interpreter overhead consume ~550–650 MB of resident memory.
- The entire vector index at 5,000 slots represents ~1.6 MB (INT8) to ~8.7 MB (FP32), which is less than 1.5% of total process RSS.
- **Rule:** The 81.36% memory reduction applies specifically to **vector array payload and index memory structures**, NOT to total OS process RSS.

---

## 6. Check 5: Scenario Completeness & Data Consistency

### Recalculation of All 12 Scenarios
Every percentage reduction reported in the master table was recalculated from the raw byte values in `results/phase7c/scalability_results.json`:
$$\text{Storage Reduction \%} = \left(1 - \frac{\text{DCLC INT8 Bytes}}{\text{Baseline FP32 Bytes}}\right) \times 100\%$$

| Scenario Key | Target Slots | Actual Slots | Unique Chunks | Base FP32 (B) | DCLC INT8 (B) | Reported % | Recalculated % | Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `scale_100_low` | 100 | 125 | 119 | 192,000 | 46,648 | 75.70% | 75.7042% | 0.0000% |
| `scale_100_medium` | 100 | 123 | 95 | 188,928 | 37,240 | 80.29% | 80.2888% | 0.0000% |
| `scale_100_high` | 100 | 128 | 56 | 196,608 | 21,952 | 88.83% | 88.8346% | 0.0000% |
| `scale_500_low` | 500 | 581 | 535 | 892,416 | 209,720 | 76.50% | 76.5000% | 0.0000% |
| `scale_500_medium` | 500 | 581 | 432 | 892,416 | 169,344 | 81.02% | 81.0241% | 0.0000% |
| `scale_500_high` | 500 | 602 | 241 | 924,672 | 94,472 | 89.78% | 89.7832% | 0.0000% |
| `scale_1000_low` | 1,000 | 1,148 | 1,053 | 1,763,328 | 412,776 | 76.59% | 76.5911% | 0.0000% |
| `scale_1000_medium` | 1,000 | 1,161 | 867 | 1,783,296 | 339,864 | 80.94% | 80.9418% | 0.0000% |
| `scale_1000_high` | 1,000 | 1,137 | 440 | 1,746,432 | 172,480 | 90.12% | 90.1238% | 0.0000% |
| `scale_5000_low` | 5,000 | 5,755 | 5,143 | 8,839,680 | 2,016,056 | 77.19% | 77.1931% | 0.0000% |
| `scale_5000_medium` | 5,000 | 5,667 | 4,139 | 8,704,512 | 1,622,488 | 81.36% | 81.3604% | 0.0000% |
| `scale_5000_high` | 5,000 | 5,634 | 2,212 | 8,653,824 | 867,104 | 89.98% | 89.9801% | 0.0000% |

**Discrepancies Found:** Exactly **0**. All values match raw data to 4 decimal places.

---

## 7. Status of Claims & Readiness for Phase 7D

| Research Claim | Audit Status | Audit Assessment |
| :--- | :---: | :--- |
| **Vector Storage Compounding Savings (75%–90%)** | **CONFIRMED** | Fully supported by raw byte counts across all 12 scenarios. |
| **Strict Version Isolation (0 Violations)** | **CONFIRMED** | Fully supported across 72 evaluated configurations. |
| **Top-1 Agreement Parity (100.0%)** | **CONFIRMED** | INT8 SQ8 ranks the exact same candidate as FP32 in all tested queries. |
| **Sub-Linear Query Latency** | **REFUTED & CORRECTED** | Latency scales **linearly / super-linearly ($O(N^{1.08})$)** due to exact matrix and dictionary scans. |
| **R@5 = 0.33 Meaning** | **CLARIFIED** | Mathematical artifact of keyword count fallback ($1/3$), representing 100% of available ground-truth passages. |
| **Process RSS Reduction** | **CLARIFIED** | Process RSS is dominated by PyTorch (~650 MB); index memory savings (~7 MB) are specific to data structures. |

### Are Additional Experiments Required Before Phase 7D?
**No additional experiments are required.** The empirical data rigorously supports Phase 7C's core research findings regarding embedding memory optimization and cross-version deduplication. The report has been updated to remove the inaccurate sub-linear latency claim and to provide transparent explanations of quality metrics and memory boundaries.
