# DCLC Final-Year CSE Project: HOD & Panel Demonstration Guide

**Project Title:** Deep Contrastive Learning–Based Redundancy Elimination for Similar Data in Cloud Storage Systems (DCLC)  
**Student Presenter:** Jasinth Ashankar (Final-Year B.Tech CSE)  
**Research Paper:** Ling Xiao, Qinbao Fang, and Shihao Feng (2026), *Cluster Computing*, DOI: [10.1007/s44443-026-01042-5](https://doi.org/10.1007/s44443-026-01042-5)  
**Demonstration Environment:** Windows 11 | Python 3.13.10 | FastAPI | React 19.2 | PyTorch 2.14  

---

## 1. Quick Pre-Demo Setup Checklist

Before calling the panel or starting your presentation:

1. **Terminal 1: Start FastAPI Backend**
   ```powershell
   cd C:\Users\jasin\OneDrive\Desktop\DCLC
   .\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *Verify:* Navigate to `http://127.0.0.1:8000/docs` to see Swagger UI with all 19 routes active.

2. **Terminal 2: Start React Frontend**
   ```powershell
   cd C:\Users\jasin\OneDrive\Desktop\DCLC\frontend
   npm run dev
   ```
   *Verify:* Navigate to `http://localhost:5173` to see the live DCLC dashboard with all tabs active.

3. **Terminal 3: Verification Sandbox / Terminal Commands**
   Keep a terminal ready in `C:\Users\jasin\OneDrive\Desktop\DCLC` for live CLI demonstrations and test execution.

---

## 2. Five Live Demonstration Scenarios

### Scenario 1: FastCDC Chunking & Boundary-Shift Absorption
**Concept:** Explain why fixed-size chunking fails and show how FastCDC dynamically identifies chunk boundaries.

**Interactive UI Steps:**
1. In the frontend sidebar, click on **"Chunk Analysis"**.
2. Select or upload any file (or test document).
3. Set Minimum Chunk Size (`2048`), Average (`4096`), Maximum (`8192`).
4. Click **"Analyze FastCDC Chunk Boundaries"**.
5. **Show Panel:** The chunk distribution histogram, the unique SHA-256 fingerprints, and the average chunk size hovering closely around 4 KB.

**CLI Alternative (cURL):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/chunk-preview" `
  -F "file=@tests/test_data/sample.txt;type=text/plain" `
  -F "min_chunk_size=2048" -F "avg_chunk_size=4096" -F "max_chunk_size=8192"
```

---

### Scenario 2: Two-Revision Comparison & Boundary Absorption
**Concept:** Demonstrate how DCLC absorbs localized insertions/deletions without breaking downstream duplicate detection.

**Interactive UI Steps:**
1. Click on **"Compare Files"** in the sidebar.
2. Select **Base File** (e.g. `data/multi_version/raw/family_085/v0.txt` or a sample document).
3. Select **Target File** (e.g. `v1.txt` with middle edits).
4. Select Candidate Strategy: **Hierarchical Cache + BallTree**.
5. Select Delta Mode: **Single-Reference (SR)**.
6. Click **"Execute Redundancy Elimination"**.
7. **Show Panel:**
   - **Exact Duplicates:** Highlight green chunks that matched exact SHA-256 (32-byte pointer overhead).
   - **Resemblance Hits:** Highlight blue chunks where neural cosine similarity $\ge 0.50$ enabled preset dictionary delta compression.
   - **Storage Savings %:** Point out the $\approx 85\text{--}90\%$ reduction in bytes stored.
   - **Bit-Exact Verification:** 100.0% reconstruction verified on all chunks.

---

### Scenario 3: Multi-Reference Delta (MRD) vs. Single-Reference (SR)
**Concept:** Demonstrate Section 3.4 of the paper where multiple candidate base chunks are pooled to compress consecutive target chunks.

**Interactive UI Steps:**
1. On the **"Compare Files"** screen:
2. Switch Delta Compression Mode to **"Multi-Reference Delta (MRD Sec 3.4)"**.
3. Set **MRD Window Size** to `3` (neighborhood targets) and **Max References** to `3`.
4. Click **"Execute Redundancy Elimination"**.
5. **Show Panel:**
   - The breakdown table shows `delta_mode: "multi"` and groups of 3 target chunks encoded together.
   - Point out the higher compression efficiency compared to SR mode.
   - Highlight the **Bit-Exact Reconstruction** badge: every target chunk is individually and group-restored and validated.

---

### Scenario 4: Durable Revision Persistence, Restart-Safety & Download (Phase E)
**Concept:** Demonstrate that DCLC is a complete storage system, not just an in-memory simulation.

**Interactive UI Steps:**
1. In the sidebar, click on the **"Durable Store"** tab.
2. Upload `Base Reference File` and `Target Revision File`.
3. Choose `Hierarchical Cache` and `Multi-Reference Delta (MRD)`.
4. Click **"Compress & Durably Persist Revision"**.
5. **Show Panel:**
   - The new revision appears in the **Durable Revisions Manifest** table with its `rev_YYYYMMDD_HHMMSS_<hash>` ID, original raw size, stored size, and savings %.
6. **Kill & Restart Verification (The "Acid Test"):**
   - Click **"Verify"** button: Backend restores the file from disk references (`chunks/` and `deltas/`) and confirms SHA-256 match.
   - Click **"Download"** button: The reconstructed binary file downloads directly in the browser!
   - Click **"Recipe"** button: Inspect the atomic JSON manifest showing exactly which chunks are `exact`, `delta_mrd`, or `standalone`.

**CLI Alternative (cURL):**
```powershell
# 1. Store revision
curl.exe -X POST "http://127.0.0.1:8000/store-revision" `
  -F "base_file=@tests/test_data/base.txt" `
  -F "target_file=@tests/test_data/target.txt" `
  -F "candidate_strategy=hierarchical_cache" `
  -F "delta_mode=multi"

# 2. List stored revisions
curl.exe "http://127.0.0.1:8000/revisions"

# 3. Restore and verify SHA-256
curl.exe "http://127.0.0.1:8000/restore-revision/<REVISION_ID>"
```

---

### Scenario 5: Full Test-Split Benchmark & Ablation Study (Phase F)
**Concept:** Present the rigorous research benchmark on the held-out test split (families 085–099).

**Interactive UI Steps:**
1. Click on **"Storage Benchmark"** in the sidebar.
2. Click **"Run Full Test-Split Benchmark"** (or view the pre-computed results).
3. **Show Panel:**
   - **System 0 (Raw):** 3.56 MB (1.00x)
   - **System 1 (Deflate):** 568 KB (6.27x)
   - **System 2 (Exact Dedup):** 554 KB (6.43x)
   - **System 3D (DCLC-SR with Hierarchical Cache):** 459 KB (7.75x)
   - **System 4 (DCLC-MRD):** 439 KB (8.12x, **87.68% savings**)
4. Open the generated report directly: `reports/research_evaluation_report.md`.

---

## 3. Defensive Q&A Cheat Sheet for Panel Examination

### Q1: Why did you use `BallTree` instead of `Annoy` mentioned in the paper?
**Answer:**  
> "The original paper used Spotify's Annoy library for Tier 2 approximate nearest neighbor search. However, Annoy does not provide official precompiled binary wheels for Python 3.13 on 64-bit Windows.  
> Rather than using an unverified C++ compile, we deployed `scikit-learn`'s `BallTree` over $L_2$-normalized unit vectors. Mathematically, for unit vectors $u$ and $v$, Euclidean distance $d$ and cosine similarity are linked by $\cos(u, v) = 1 - d^2/2$. Minimizing Euclidean distance is mathematically exact to maximizing cosine similarity. Furthermore, `BallTree` performs exact metric tree search rather than randomized tree approximation, which guarantees 100% precision in nearest neighbor retrieval for our evaluation workload."

### Q2: How does the Hierarchical Cache improve retrieval performance?
**Answer:**  
> "Cloud storage streams have high temporal and spatial locality. Tier 1 is a bounded FIFO cache of 64 recent chunk embeddings. When a new chunk arrives, it queries Tier 1 in $O(64)$ time. If similarity $\ge 0.70$, it hits Tier 1 immediately without querying the global index. In our test evaluation, this achieved a **99.60% reduction in pairwise comparisons**, inspecting on average only $0.94$ candidates per chunk while achieving 87.10% storage savings."

### Q3: What is the trade-off with Multi-Reference Delta (MRD)?
**Answer:**  
> "MRD extracts redundancy across multi-chunk neighborhoods by combining $W=3$ target chunks and up to $M=3$ reference base chunks into a unified container (`DCLC_M01`). This reduces stored bytes by an additional 20.6 KB over Single-Reference mode (reaching an 8.12x compression ratio).  
> The trade-off is in restoration throughput: reconstructing a single isolated chunk from an MRD container requires parsing the multi-reference dictionary, which runs at 64.52 MB/s versus 172.16 MB/s for SR. However, full group restoration operates at 141.27 MB/s, recovering near-SR speeds."

### Q4: How is data integrity guaranteed? What if a chunk gets corrupted?
**Answer:**  
> "Integrity is guaranteed at multiple layers:
> 1. FastCDC chunks and delta containers are stored content-addressed by their SHA-256 cryptographic hash.
> 2. Every write uses atomic temporary files (`.tmp`) followed by `os.replace` to prevent partial writes.
> 3. During restoration, every chunk is verified against its manifest SHA-256. If a chunk is tampered with or corrupted, the system detects the checksum mismatch and cleanly raises HTTP 422 `StorageCorruptionError` rather than returning corrupted data.
> 4. Across our entire test suite of 269 automated tests, every single restoration achieved 100.0% bit-exact reconstruction."

---

## 4. Test Suite Verification Command for Panel
To prove to the panel that the implementation is robust and regression-free:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
**Expected Output:**  
`269 passed, 4 warnings in ~45s` (0 failures, 100% passing).
