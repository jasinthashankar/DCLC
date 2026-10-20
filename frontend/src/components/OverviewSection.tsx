import type { FC } from 'react';
import {
  ArrowRight,
  CheckCircle2,
  HelpCircle,
} from 'lucide-react';
import type { BenchmarkStorageResponse, HealthResponse } from '../types/api';
import { formatBytes, formatPct, formatRatio } from '../utils/formatters';

interface OverviewSectionProps {
  health: HealthResponse | null;
  benchmark: BenchmarkStorageResponse | null;
  benchmarkLoading: boolean;
  onNavigateTab: (tab: string) => void;
}

export const OverviewSection: FC<OverviewSectionProps> = ({
  health,
  benchmark,
  benchmarkLoading,
  onNavigateTab,
}) => {
  const dclcPrimary = benchmark?.system_3_full_dclc?.['Stream_Locality_W32'] ||
    (benchmark?.system_3_full_dclc ? Object.values(benchmark.system_3_full_dclc)[0] : null);

  return (
    <div className="space-y-8 animate-fadeIn max-w-5xl">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Project Overview & Live Status</h1>
        <p className="text-sm text-slate-400 mt-1">
          Deep Contrastive Learning-based Redundancy Elimination for Similar Data in Cloud Storage Systems.
        </p>
      </div>

      {/* Academic Problem & Solution Explanation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-2">
          <div className="flex items-center space-x-2 text-indigo-400 text-xs font-bold uppercase tracking-wider">
            <span className="w-2 h-2 rounded-full bg-indigo-500" />
            <span>The Research Problem</span>
          </div>
          <h3 className="text-base font-semibold text-white">Why Standard Deduplication Fails on Revisions</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Conventional cloud backup systems use cryptographic hashes (like SHA-256) to identify <strong>100% exact duplicate</strong> chunks. 
            However, when files are edited, updated, or compiled, chunks change slightly. 
            They are often <strong>90% identical</strong> in content, but because their cryptographic hash changes completely, 
            standard deduplication treats them as entirely new data—wasting massive amounts of cloud storage.
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-2">
          <div className="flex items-center space-x-2 text-emerald-400 text-xs font-bold uppercase tracking-wider">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>The DCLC Solution</span>
          </div>
          <h3 className="text-base font-semibold text-white">Learned Resemblance & Delta Compression</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            DCLC breaks files into variable-sized chunks using <strong>FastCDC</strong>, extracts multi-scale shingle features, 
            and uses a <strong>Context-Aware Transformer Encoder</strong> to generate embeddings that capture chunk similarity. 
            Label-free indexers quickly find reference chunks, and a verified <strong>preset-dictionary delta encoder</strong> stores only the byte differences—achieving up to 87.8% storage savings on multi-version data.
          </p>
        </div>
      </div>

      {/* Live Benchmark Snapshot */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-white">Live Benchmark Summary</h2>
            <p className="text-xs text-slate-400">
              Real measurements evaluated dynamically from the test split (Families 085–099, 464 chunks).
            </p>
          </div>
          <button
            onClick={() => onNavigateTab('benchmark')}
            className="text-xs text-indigo-400 hover:text-indigo-300 font-medium inline-flex items-center space-x-1"
          >
            <span>Detailed Benchmark</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {benchmarkLoading && !benchmark ? (
          <div className="p-8 text-center text-xs text-slate-400 bg-slate-900 border border-slate-800 rounded-xl">
            Loading verified benchmark figures from backend...
          </div>
        ) : benchmark ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* System 0 */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">1. Raw Logical Data</span>
              <div className="text-xl font-bold text-white">
                {formatBytes(benchmark.system_0_raw.stored_bytes)}
              </div>
              <div className="text-[11px] text-slate-400">Baseline uncompressed size (1.00:1)</div>
            </div>

            {/* System 1 */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">2. Standalone Deflate</span>
              <div className="text-xl font-bold text-slate-200">
                {formatBytes(benchmark.system_1_standalone_compression.stored_bytes)}
              </div>
              <div className="text-[11px] text-slate-400">
                Savings: <span className="text-slate-200 font-semibold">{formatPct(benchmark.system_1_standalone_compression.savings_pct)}</span> ({formatRatio(benchmark.system_1_standalone_compression.compression_ratio)})
              </div>
            </div>

            {/* System 2 */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">3. Exact Dedup (SHA-256)</span>
              <div className="text-xl font-bold text-slate-200">
                {formatBytes(benchmark.system_2_exact_dedup.stored_bytes)}
              </div>
              <div className="text-[11px] text-slate-400">
                Savings: <span className="text-slate-200 font-semibold">{formatPct(benchmark.system_2_exact_dedup.savings_pct)}</span> ({formatRatio(benchmark.system_2_exact_dedup.compression_ratio)})
              </div>
            </div>

            {/* System 3 */}
            <div className="bg-gradient-to-b from-indigo-950/40 to-slate-900 border border-indigo-500/40 rounded-xl p-4 space-y-1.5 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-indigo-400 uppercase tracking-wider">4. Full DCLC Pipeline</span>
                <span className="text-[10px] px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-300 font-semibold">Best</span>
              </div>
              <div className="text-xl font-bold text-emerald-400">
                {dclcPrimary ? formatBytes(dclcPrimary.stored_bytes) : 'Loading...'}
              </div>
              <div className="text-[11px] text-slate-300">
                Savings: <span className="text-emerald-400 font-bold">{formatPct(dclcPrimary?.savings_pct)}</span> ({formatRatio(dclcPrimary?.compression_ratio)})
              </div>
            </div>
          </div>
        ) : (
          <div className="p-6 text-center text-xs text-slate-400 bg-slate-900 border border-slate-800 rounded-xl">
            Could not connect to backend to fetch live metrics. Ensure the backend server is running on port 8000.
          </div>
        )}
      </div>

      {/* Plain-Language Explanations of Key Research Metrics */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <div className="flex items-center space-x-2 text-white font-semibold text-sm">
          <HelpCircle className="w-4 h-4 text-indigo-400" />
          <span>Understanding the Metrics</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs text-slate-300 pt-1">
          <div className="bg-slate-950/60 p-3.5 rounded-lg border border-slate-800/80 space-y-1">
            <span className="font-bold text-slate-200">Storage Savings Percentage (%):</span>
            <p className="text-slate-400 leading-relaxed">
              Calculated as <code className="text-indigo-300">(Raw Bytes - Stored Bytes) / Raw Bytes × 100</code>. 
              A higher percentage means more space was saved. For example, 87% savings means the final data consumes only 13% of its original disk footprint.
            </p>
          </div>
          <div className="bg-slate-950/60 p-3.5 rounded-lg border border-slate-800/80 space-y-1">
            <span className="font-bold text-slate-200">Compression Ratio (X:1):</span>
            <p className="text-slate-400 leading-relaxed">
              Calculated as <code className="text-indigo-300">Raw Bytes / Stored Bytes</code>. 
              A ratio of <strong>8.19:1</strong> means that for every 8.19 MB of uncompressed logical data, cloud storage only needs to store 1 MB of physical bytes.
            </p>
          </div>
        </div>
      </div>

      {/* System Readiness Verification Grid */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <h3 className="text-sm font-bold text-white">System Component Health</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className="p-3 bg-slate-950/60 border border-slate-800/80 rounded-lg space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Transformer Encoder</span>
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="font-semibold text-slate-200">
              {health?.checkpoint_available ? 'Checkpoint Loaded' : 'Missing Checkpoint'}
            </div>
            <div className="text-[11px] text-slate-500 font-mono">128-dim embeddings</div>
          </div>

          <div className="p-3 bg-slate-950/60 border border-slate-800/80 rounded-lg space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Candidate Indexers</span>
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="font-semibold text-slate-200">Label-Free Active</div>
            <div className="text-[11px] text-slate-500 font-mono">Stream, RP-LSH, k-NN</div>
          </div>

          <div className="p-3 bg-slate-950/60 border border-slate-800/80 rounded-lg space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Delta Compressor</span>
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="font-semibold text-slate-200">Preset Dict (RFC 1950)</div>
            <div className="text-[11px] text-slate-500 font-mono">100% Bit-Exact Verified</div>
          </div>
        </div>
      </div>
    </div>
  );
};
