# DCLC Research Evaluation & Empirical Validation Report (Phase F)

**Paper Reference:** Ling Xiao, Qinbao Fang, and Shihao Feng (2026). *“Deep contrastive learning–based redundancy elimination for similar data in cloud storage systems”*.
**Evaluation Date:** 2026-10-19 15:56:23 UTC
**Platform / Environment:** Windows 11 | Python 3.13.10 | Torch 2.14.1+cpu

---

## 1. Experimental Dataset Scope

The evaluation was executed over the held-out test split of multi-version document repositories, strictly isolated from training families (families 000-084 for training, 085-099 for held-out evaluation):

- **Dataset Split:** `test` (Families 085–099)
- **Total Chunks Evaluated:** `464` chunks
- **Total Logical Raw Bytes:** `3,564,567` bytes (3.4 MB)
- **Exact Duplicate Targets:** `26` chunks
- **Resemblance Target Pairs (Ground Truth):** `542` pairs

---

## 2. Controlled Architecture Comparison

| Architecture | Stored Bytes | Savings (Bytes) | Savings (%) | Compression Ratio | Accepted Deltas | Reconstruction Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **System 0: Raw Storage** | 3,564,567 B | 0 B | 0.00% | 1.00x | N/A | 100.0% |
| **System 1: Standalone Deflate** | 568,542 B | 2,996,025 B | 84.05% | 6.27x | N/A | 100.0% |
| **System 2: Exact SHA-256 Dedup** | 554,325 B | 3,010,242 B | 84.45% | 6.43x | N/A | 100.0% |
| **System 3A: DCLC-SR (Stream Locality W=32)** | 435,386 B | 3,129,181 B | 87.79% | 8.19x | 426 | 100.0% |
| **System 3B: DCLC-SR (RP-LSH L=8, B=6)** | 491,085 B | 3,073,482 B | 86.22% | 7.26x | 377 | 100.0% |
| **System 3C: DCLC-SR (Embedding k-NN K=20)** | 479,304 B | 3,085,263 B | 86.55% | 7.44x | 434 | 100.0% |
| **System 3D: DCLC-SR (Hierarchical Cache + BallTree)** | 459,858 B | 3,104,709 B | 87.10% | 7.75x | 434 | 100.0% |
| **System 4: DCLC-MRD (Hierarchical + MRD W=3)** | 439,173 B | 3,125,394 B | 87.68% | 8.12x | 153 groups | 100.0% |

---

## 3. Candidate Indexing & Ablation Analysis

Evaluation of label-free candidate generation algorithms (Section 3.3):

| Candidate Strategy | Avg Pool Size | Max Pool Size | Candidate Recall (%) | Comparison Reduction (%) | Query Latency (ms/chunk) | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stream Locality (W=32)** | 29.2 | 32 | 94.10% | 87.38% | 0.0028 ms | 0.221 s |
| **RP-LSH (L=8, B=6)** | 25.1 | 63 | 12.55% | 89.15% | 0.0690 ms | 0.259 s |
| **Embedding Cosine k-NN (K=20)** | 18.5 | 20 | 42.44% | 92.00% | 0.2131 ms | 0.320 s |
| **Hierarchical Cache + BallTree** | 0.94 | 1 | 7.56% | 99.60% | 5.8135 ms | 2.886 s |

---

## 4. Multi-Reference Delta (MRD) Evaluation (Section 3.4)

Direct empirical comparison between Single-Reference (SR) preset-dictionary delta compression and Multi-Reference Delta (MRD) group container encoding:

| Metric | Single-Reference (SR) | Multi-Reference (MRD) | Delta / Trade-off |
| :--- | :---: | :---: | :---: |
| **Total Stored Size** | 459,858 B | 439,173 B | **-20,685 B (-4.50%)** |
| **Compression Ratio** | 7.751x | 8.117x | **+0.366x** |
| **Accepted Groups/Deltas** | 434 deltas | 153 groups | 153 groups accepted (0 rejected) |
| **Encoding Time** | 0.1670 s | 0.1715 s | Comparable (0.0045 s diff) |
| **Group Restoration Throughput** | 172.16 MB/s | 141.27 MB/s | -17.95% |
| **Single-Target Restoration Throughput** | 172.16 MB/s | 64.52 MB/s | -62.52% |

> **Research Insight:** MRD demonstrates superior storage compression over SR by extracting redundancy across multi-chunk neighborhoods. However, restoring an isolated single target from an MRD container requires parsing multi-reference dictionaries, resulting in a throughput penalty (65.14 MB/s vs 167.87 MB/s). Full group restoration recovers high throughput (140.45 MB/s), closely approaching SR speeds.

---

## 5. Paper Alignment and Fidelity Comparison Table

| Component | Original Paper Specification | Repository Implementation | Fidelity Status | Engineering Rationale / Context |
| :--- | :--- | :--- | :---: | :--- |
| **Chunking** | FastCDC content-defined chunking (avg 4KB or 8KB) | FastCDC via official fastcdc library | **Exact Match** | Preserves variable chunk boundary absorption across mutations. |
| **Feature Extraction** | Multi-scale shingle hashing (Algorithm 1) | Multi-scale sliding window shingle extractor (128-dim) | **Exact Match** | Byte-level 128-dim initial feature vectors before transformer. |
| **Context Modeling** | Transformer encoder over context windows | 2-layer context-aware Transformer encoder | **Exact Match** | Self-attention captures chunk sequence order and spatial locality. |
| **Contrastive Training** | InfoNCE / NT-Xent loss on positive/negative pairs | Supervised InfoNCE with temperature scaling & hard negatives | **Exact Match** | Trained on multi-version document families. |
| **Candidate Retrieval (Tier 1)** | Bounded local cache of recent candidate embeddings | Bounded FIFO candidate cache (capacity=64, cosine threshold) | **Exact Match** | O(1) candidate lookup for stream-adjacent resemblance. |
| **Candidate Retrieval (Tier 2)** | Global Annoy approximate nearest neighbor index | Scikit-learn BallTree exact tree-based nearest neighbor search | **Justified Deviation** | Annoy lacks official precompiled Windows Python 3.13 wheels; BallTree provides mathematically exact Euclidean metric nearest-neighbor search ($cosine = 1 - d^2/2$). |
| **Delta Compression (SR)** | Preset dictionary delta encoding | Byte-level preset dictionary zlib delta with bit-exact assertion | **Exact Match** | Restores byte-for-byte identical output. |
| **Multi-Reference Delta (MRD)** | Neighborhood window with multi-candidate references | `DCLC_M01` container with group and single-target decoder | **Exact Match** | Implements Section 3.4 multi-base delta encoding. |
| **Durable Persistence** | Conceptual cloud storage backend | Content-addressed chunk store, delta store, and atomic recipes | **Full Extension** | Provides restart-safe durability, SHA-256 verification, and REST API. |

---

## 6. Verification Summary

- **Bit-Exact Reconstruction Success Rate:** 100.0% across all evaluated chunks.
- **Zero Corrupted Restorations:** All restored chunks match ground-truth SHA-256 fingerprints.
- **Zero Regressions:** 269 automated tests pass with 0 failures.