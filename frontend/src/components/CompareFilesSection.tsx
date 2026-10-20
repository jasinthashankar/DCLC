import { useState, type FC, type FormEvent, type DragEvent } from 'react';
import {
  Upload,
  ArrowRightLeft,
  Sliders,
  CheckCircle2,
  FileText,
  AlertCircle,
  Search,
  Check,
  ChevronDown,
  ChevronUp,
  HelpCircle,
} from 'lucide-react';
import type { RevisionCompareResponse } from '../types/api';
import { api } from '../services/api';
import { formatBytes, formatPct, formatRatio, truncateHash } from '../utils/formatters';

export const CompareFilesSection: FC = () => {
  const [baseFile, setBaseFile] = useState<File | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [candidateStrategy, setCandidateStrategy] = useState<string>('stream_locality');
  const [threshold, setThreshold] = useState<number>(0.70);

  // Delta compression mode parameters (Phase C)
  const [deltaMode, setDeltaMode] = useState<string>('single');
  const [mrdWindowSize, setMrdWindowSize] = useState<number>(3);
  const [mrdMaxRefs, setMrdMaxRefs] = useState<number>(3);

  // FastCDC chunk size parameters
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [minChunkSize, setMinChunkSize] = useState<number>(4096);
  const [avgChunkSize, setAvgChunkSize] = useState<number>(8192);
  const [maxChunkSize, setMaxChunkSize] = useState<number>(16384);

  // Execution state
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RevisionCompareResponse | null>(null);

  // Chunk table filtering
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [filterType, setFilterType] = useState<string>('all');

  const handleBaseDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files?.[0]) setBaseFile(e.dataTransfer.files[0]);
  };

  const handleTargetDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files?.[0]) setTargetFile(e.dataTransfer.files[0]);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!baseFile || !targetFile) {
      setError('Please choose both a base file and a target file to compare.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const resp = await api.compareRevisions(
        baseFile,
        targetFile,
        candidateStrategy,
        threshold,
        minChunkSize,
        avgChunkSize,
        maxChunkSize,
        deltaMode,
        mrdWindowSize,
        mrdMaxRefs
      );
      setResult(resp);
    } catch (err: any) {
      setError(err?.detail || err?.message || 'Failed to execute file comparison.');
    } finally {
      setLoading(false);
    }
  };

  // Filtered chunk comparisons
  const filteredChunks = result?.detailed_chunk_comparisons.filter((c) => {
    const matchesSearch =
      c.target_chunk_index.toString().includes(searchTerm) ||
      c.target_sha256.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (c.selected_reference_sha256?.toLowerCase().includes(searchTerm.toLowerCase()) ?? false);

    if (!matchesSearch) return false;

    if (filterType === 'exact') return c.is_exact_duplicate;
    if (filterType === 'delta') return c.delta_accepted;
    if (filterType === 'standalone') return !c.is_exact_duplicate && !c.delta_accepted;
    return true;
  }) || [];

  return (
    <div className="space-y-8 animate-fadeIn max-w-5xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
          <ArrowRightLeft className="w-6 h-6 text-indigo-400" />
          <span>Compare Files & Evaluate Redundancy Elimination</span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Select two file versions. DCLC partitions both with FastCDC, detects identical chunks, and encodes similar non-identical chunks using learned delta compression.
        </p>
      </div>

      {/* Step-by-Step Upload & Settings Form */}
      <form onSubmit={handleSubmit} className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6 shadow-sm">
        {/* Step Indicator */}
        <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-400 pb-2 border-b border-slate-800">
          <span className="w-5 h-5 rounded-full bg-indigo-600 text-white flex items-center justify-center text-[11px]">1</span>
          <span>Select File Pair</span>
        </div>

        {/* File Drop Zones */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Base File */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleBaseDrop}
            className={`border-2 border-dashed rounded-xl p-5 text-center transition-all ${
              baseFile
                ? 'border-indigo-500/50 bg-indigo-500/5'
                : 'border-slate-800 hover:border-slate-700 bg-slate-950/40'
            }`}
          >
            <input
              type="file"
              id="base-file-input"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && setBaseFile(e.target.files[0])}
            />
            <label htmlFor="base-file-input" className="cursor-pointer block space-y-2">
              <div className="w-9 h-9 mx-auto rounded-full bg-slate-800 flex items-center justify-center text-indigo-400">
                <FileText className="w-5 h-5" />
              </div>
              <div>
                <span className="text-sm font-semibold text-white block">
                  {baseFile ? baseFile.name : 'Choose Base File (v1)'}
                </span>
                <span className="text-xs text-slate-400 mt-0.5 block">
                  {baseFile ? `${formatBytes(baseFile.size)}` : 'Click to select or drag & drop'}
                </span>
              </div>
            </label>
            {baseFile && (
              <button
                type="button"
                onClick={() => setBaseFile(null)}
                className="mt-2 text-xs text-rose-400 hover:text-rose-300 underline"
              >
                Clear
              </button>
            )}
          </div>

          {/* Target File */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleTargetDrop}
            className={`border-2 border-dashed rounded-xl p-5 text-center transition-all ${
              targetFile
                ? 'border-cyan-500/50 bg-cyan-500/5'
                : 'border-slate-800 hover:border-slate-700 bg-slate-950/40'
            }`}
          >
            <input
              type="file"
              id="target-file-input"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && setTargetFile(e.target.files[0])}
            />
            <label htmlFor="target-file-input" className="cursor-pointer block space-y-2">
              <div className="w-9 h-9 mx-auto rounded-full bg-slate-800 flex items-center justify-center text-cyan-400">
                <Upload className="w-5 h-5" />
              </div>
              <div>
                <span className="text-sm font-semibold text-white block">
                  {targetFile ? targetFile.name : 'Choose Target File (v2)'}
                </span>
                <span className="text-xs text-slate-400 mt-0.5 block">
                  {targetFile ? `${formatBytes(targetFile.size)}` : 'Click to select or drag & drop (file to compress)'}
                </span>
              </div>
            </label>
            {targetFile && (
              <button
                type="button"
                onClick={() => setTargetFile(null)}
                className="mt-2 text-xs text-rose-400 hover:text-rose-300 underline"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Step 2: Settings */}
        <div className="pt-2 border-t border-slate-800 space-y-4">
          <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-400">
            <span className="w-5 h-5 rounded-full bg-indigo-600 text-white flex items-center justify-center text-[11px]">2</span>
            <span>Algorithm Settings</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Candidate Indexer
              </label>
              <select
                value={candidateStrategy}
                onChange={(e) => setCandidateStrategy(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="stream_locality">Stream Locality (Window W=32) [Recommended]</option>
                <option value="hierarchical_cache">Hierarchical Cache + BallTree NN (Section 3.3)</option>
                <option value="rp_lsh">Random-Projection LSH (8 tables × 6 bits)</option>
                <option value="embedding_knn">Embedding Cosine k-NN (Top-20)</option>
              </select>
              <p className="text-[11px] text-slate-500 mt-1">
                Filters candidate reference chunks without comparing against all chunks.
              </p>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-semibold text-slate-300">
                  Similarity Threshold
                </label>
                <span className="text-xs font-mono font-bold text-indigo-400">
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.50"
                max="0.95"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                <span>0.50 (Loose)</span>
                <span>0.70 (Standard default)</span>
                <span>0.95 (Strict)</span>
              </div>
            </div>
          </div>

          {/* Delta Compression Mode (Phase C) */}
          <div className="pt-2 border-t border-slate-800">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Delta Encoding Architecture
                </label>
                <select
                  value={deltaMode}
                  onChange={(e) => setDeltaMode(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="single">Single-Reference Delta (SR Baseline)</option>
                  <option value="multi">Multi-Reference Delta (MRD, Section 3.4)</option>
                </select>
                <p className="text-[11px] text-slate-500 mt-1">
                  {deltaMode === 'single'
                    ? 'Encodes each similar target chunk against a single best candidate chunk.'
                    : 'Encodes sliding windows of target chunks against multiple candidate references.'}
                </p>
              </div>

              {deltaMode === 'multi' && (
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      MRD Window Size
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="16"
                      value={mrdWindowSize}
                      onChange={(e) => setMrdWindowSize(Math.max(1, parseInt(e.target.value) || 1))}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                    />
                    <p className="text-[10px] text-slate-500 mt-0.5">Target chunks window N(B_i)</p>
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      Max References
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="8"
                      value={mrdMaxRefs}
                      onChange={(e) => setMrdMaxRefs(Math.max(1, parseInt(e.target.value) || 1))}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                    />
                    <p className="text-[10px] text-slate-500 mt-0.5">Candidate chunks phi_i</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Advanced FastCDC Parameters */}
          <div className="pt-2">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center space-x-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>Advanced Chunking Parameters (FastCDC)</span>
              {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>

            {showAdvanced && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3 p-3.5 bg-slate-950/60 rounded-lg border border-slate-800 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">Min Chunk Size (bytes)</label>
                  <input
                    type="number"
                    min="64"
                    step="512"
                    value={minChunkSize}
                    onChange={(e) => setMinChunkSize(parseInt(e.target.value) || 4096)}
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Target Avg Size (bytes)</label>
                  <input
                    type="number"
                    min="128"
                    step="512"
                    value={avgChunkSize}
                    onChange={(e) => setAvgChunkSize(parseInt(e.target.value) || 8192)}
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Max Chunk Size (bytes)</label>
                  <input
                    type="number"
                    min="256"
                    step="1024"
                    value={maxChunkSize}
                    onChange={(e) => setMaxChunkSize(parseInt(e.target.value) || 16384)}
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-slate-200"
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg flex items-center space-x-2 text-rose-400 text-xs">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Step 3: Action */}
        <div className="flex justify-end pt-2 border-t border-slate-800">
          <button
            type="submit"
            disabled={loading || !baseFile || !targetFile}
            className="px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold shadow-md transition-all flex items-center space-x-2"
          >
            {loading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                <span>Running Chunking & Delta Pipeline...</span>
              </>
            ) : (
              <>
                <span>Compare Files</span>
                <ArrowRightLeft className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </form>

      {/* Results Display */}
      {result && (
        <div className="space-y-6 animate-fadeIn">
          {/* Summary Banner */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-white">Comparison Results</h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Base: <span className="text-slate-200 font-mono">{result.base_filename}</span> ({formatBytes(result.base_file_size_bytes)}, {result.base_chunk_count} chunks) → 
                  Target: <span className="text-slate-200 font-mono"> {result.target_filename}</span> ({formatBytes(result.target_file_size_bytes)}, {result.target_chunk_count} chunks)
                </p>
              </div>

              {/* Invertibility Badge */}
              <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-xs font-semibold">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>100% Reconstruction Verified</span>
              </div>
            </div>

            {/* Storage Progress / Comparison Metric Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800/80">
                <span className="text-slate-400">Target Raw Size</span>
                <div className="text-base font-bold text-white mt-1">{formatBytes(result.target_raw_bytes)}</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Uncompressed baseline</div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800/80">
                <span className="text-slate-400">Standalone Deflate</span>
                <div className="text-base font-bold text-slate-200 mt-1">{formatBytes(result.target_standalone_compressed_bytes)}</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Without deduplication</div>
              </div>

              <div className="p-3.5 rounded-lg bg-indigo-950/40 border border-indigo-500/40">
                <span className="text-indigo-300 font-semibold">DCLC Final Stored</span>
                <div className="text-base font-bold text-emerald-400 mt-1">{formatBytes(result.target_dclc_stored_bytes)}</div>
                <div className="text-[10px] text-emerald-400 font-semibold mt-0.5">
                  Savings: {formatPct(result.target_storage_savings_pct)}
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800/80">
                <span className="text-slate-400">Compression Ratio</span>
                <div className="text-base font-bold text-emerald-400 mt-1">{formatRatio(result.target_compression_ratio)}</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Original : Stored</div>
              </div>
            </div>

            {/* Deduplication & Delta Breakdown Stats */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs pt-1">
              <div className="p-2.5 rounded bg-slate-950 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Exact Duplicates:</span>
                <span className="font-bold text-emerald-400">{result.exact_duplicates_count} chunks</span>
              </div>
              <div className="p-2.5 rounded bg-slate-950 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Similar Candidates:</span>
                <span className="font-bold text-cyan-400">{result.resemblance_pairs_count} pairs</span>
              </div>
              <div className="p-2.5 rounded bg-slate-950 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Deltas Accepted:</span>
                <span className="font-bold text-indigo-400">{result.deltas_accepted_count} chunks</span>
              </div>
              <div className="p-2.5 rounded bg-slate-950 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Deltas Rejected:</span>
                <span className="font-bold text-amber-400">{result.deltas_rejected_count} chunks</span>
              </div>
            </div>

            {/* Phase C Architecture Telemetry (MRD & Hierarchical Indexer) */}
            {(result.delta_mode === 'multi' || result.indexer_stats) && (
              <div className="p-3 rounded-lg bg-indigo-950/30 border border-indigo-500/20 text-xs flex flex-wrap items-center justify-between gap-2">
                {result.delta_mode === 'multi' && (
                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-semibold text-[11px]">Multi-Reference Delta (MRD)</span>
                    <span className="text-slate-300">
                      Groups: <strong className="text-emerald-400">{result.mrd_groups_accepted_count}</strong> accepted / <strong>{result.mrd_groups_count}</strong> total
                      (Window={result.mrd_window_size}, MaxRefs={result.mrd_max_refs})
                    </span>
                  </div>
                )}
                {result.indexer_stats && (
                  <div className="text-slate-400 text-[11px]">
                    Candidate Index: <span className="text-slate-200">{result.indexer_stats.cache_hits} cache hits</span>,{' '}
                    <span className="text-slate-200">{result.indexer_stats.cache_misses} misses</span>,{' '}
                    <span className="text-slate-200">{result.indexer_stats.ann_queries} BallTree NN queries</span> ({result.indexer_stats.total_retrieval_time_s}s)
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Granular Chunk Decision Breakdown */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h4 className="text-sm font-bold text-white">Target Chunk Decisions ({result.detailed_chunk_comparisons.length})</h4>
                <p className="text-xs text-slate-400">Individual decision for each target chunk relative to the base file.</p>
              </div>

              <div className="flex items-center space-x-2">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
                  <input
                    type="text"
                    placeholder="Search chunk or SHA-256..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-3 py-1 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                  />
                </div>

                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="all">All Chunks</option>
                  <option value="exact">Exact Duplicates</option>
                  <option value="delta">Deltas Accepted</option>
                  <option value="standalone">Standalone</option>
                </select>
              </div>
            </div>

            {/* Chunk Decision Table */}
            <div className="overflow-x-auto border border-slate-800 rounded-lg">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="p-3">Chunk</th>
                    <th className="p-3">Target Size</th>
                    <th className="p-3">Decision</th>
                    <th className="p-3">Reference Chunk</th>
                    <th className="p-3">Cosine Sim</th>
                    <th className="p-3">Physical Stored</th>
                    <th className="p-3">Reconstruction</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                  {filteredChunks.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="p-4 text-center text-slate-500 font-sans">
                        No chunks match your current search/filter.
                      </td>
                    </tr>
                  ) : (
                    filteredChunks.map((c) => {
                      let decisionBadge = (
                        <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 text-[11px] font-sans">
                          Standalone Deflate
                        </span>
                      );
                      if (c.is_exact_duplicate) {
                        decisionBadge = (
                          <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-[11px] font-sans font-medium">
                            Exact Duplicate
                          </span>
                        );
                      } else if (c.delta_accepted) {
                        decisionBadge = (
                          <span className="px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 text-[11px] font-sans font-medium">
                            Delta Accepted
                          </span>
                        );
                      } else if (c.resemblance_detected && !c.delta_accepted) {
                        decisionBadge = (
                          <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30 text-[11px] font-sans font-medium">
                            Delta Unhelpful
                          </span>
                        );
                      }

                      return (
                        <tr key={c.target_chunk_index} className="hover:bg-slate-800/30 transition-colors">
                          <td className="p-3 text-slate-400 font-bold">#{c.target_chunk_index}</td>
                          <td className="p-3 font-sans text-slate-200">{formatBytes(c.target_size_bytes)}</td>
                          <td className="p-3">{decisionBadge}</td>
                          <td className="p-3 text-slate-300">
                            {c.selected_reference_chunk_index !== null ? (
                              <span title={c.selected_reference_sha256 || ''}>
                                #{c.selected_reference_chunk_index} ({truncateHash(c.selected_reference_sha256 || '', 4, 4)})
                              </span>
                            ) : (
                              <span className="text-slate-600">—</span>
                            )}
                          </td>
                          <td className="p-3">
                            {c.neural_cosine_similarity !== null ? (
                              <span className={c.neural_cosine_similarity >= c.similarity_threshold ? 'text-indigo-400 font-bold' : 'text-slate-400'}>
                                {c.neural_cosine_similarity.toFixed(4)}
                              </span>
                            ) : (
                              <span className="text-slate-600">—</span>
                            )}
                          </td>
                          <td className="p-3 font-sans text-slate-200">
                            {c.is_exact_duplicate
                              ? '32 B (Pointer)'
                              : c.delta_size_bytes
                              ? `${formatBytes(c.delta_size_bytes)} (Delta)`
                              : `${formatBytes(c.standalone_compressed_size_bytes)} (Deflate)`}
                          </td>
                          <td className="p-3">
                            {c.reconstruction_verified ? (
                              <span className="inline-flex items-center space-x-1 text-emerald-400 font-sans text-[11px]">
                                <Check className="w-3.5 h-3.5" />
                                <span>100% Invertible</span>
                              </span>
                            ) : (
                              <span className="text-rose-400 font-sans text-[11px]">Failed</span>
                            )}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Plain-Language Explanation of Terms */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center space-x-2 text-white font-semibold text-xs uppercase tracking-wider">
              <HelpCircle className="w-4 h-4 text-indigo-400" />
              <span>Terminology Reference</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs text-slate-300">
              <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800/80 space-y-1">
                <span className="font-bold text-emerald-400">Exact Duplicate:</span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Chunk is 100% identical to a base chunk. It is eliminated and replaced with a 32-byte cryptographic hash pointer.
                </p>
              </div>

              <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800/80 space-y-1">
                <span className="font-bold text-cyan-400">Similarity / Resemblance:</span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Non-identical chunks whose Transformer encoder embeddings achieve cosine similarity $\ge$ threshold (e.g. 0.70).
                </p>
              </div>

              <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800/80 space-y-1">
                <span className="font-bold text-indigo-400">Delta Encoding:</span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Compressing difference bytes against the reference chunk using RFC 1950 preset dictionaries in a 76-byte container.
                </p>
              </div>

              <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800/80 space-y-1">
                <span className="font-bold text-purple-400">Reconstruction Verified:</span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  The system immediately decompresses the delta in memory and asserts that the output is 100% bit-exact to the target file.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
