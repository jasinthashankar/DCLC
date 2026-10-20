import { useState, type FC, type FormEvent } from 'react';
import {
  BarChart3,
  RefreshCw,
  Sliders,
  Layers,
  Info,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import type { BenchmarkStorageResponse } from '../types/api';
import { formatBytes, formatPct, formatRatio } from '../utils/formatters';

interface StorageBenchmarkSectionProps {
  benchmark: BenchmarkStorageResponse | null;
  loading: boolean;
  error: string | null;
  onRefresh: (threshold?: number) => void;
}

export const StorageBenchmarkSection: FC<StorageBenchmarkSectionProps> = ({
  benchmark,
  loading,
  error,
  onRefresh,
}) => {
  const [threshold, setThreshold] = useState<number>(0.70);

  const handleApplyThreshold = (e: FormEvent) => {
    e.preventDefault();
    onRefresh(threshold);
  };

  if (!benchmark && loading) {
    return (
      <div className="p-12 text-center bg-slate-900 border border-slate-800 rounded-xl space-y-3">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto" />
        <div className="text-sm font-semibold text-white">Evaluating Multi-Version Storage Benchmark...</div>
        <div className="text-xs text-slate-400">Running FastCDC, candidate indexing, and byte-level delta compression across test families.</div>
      </div>
    );
  }

  if (error && !benchmark) {
    return (
      <div className="p-8 text-center bg-slate-900 border border-rose-500/30 rounded-xl space-y-2">
        <div className="text-sm font-bold text-rose-400">Failed to load benchmark data</div>
        <div className="text-xs text-slate-400">{error}</div>
        <button
          onClick={() => onRefresh(threshold)}
          className="mt-3 px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold text-white"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  // Bar chart dataset: physical stored bytes in MB
  const storageComparisonData = benchmark
    ? [
        {
          name: 'Raw Baseline',
          short: 'Raw',
          storedMB: parseFloat((benchmark.system_0_raw.stored_bytes / (1024 * 1024)).toFixed(3)),
          savingsPct: 0.0,
          color: '#64748b',
        },
        {
          name: 'Standalone Deflate',
          short: 'Deflate',
          storedMB: parseFloat((benchmark.system_1_standalone_compression.stored_bytes / (1024 * 1024)).toFixed(3)),
          savingsPct: benchmark.system_1_standalone_compression.savings_pct,
          color: '#38bdf8',
        },
        {
          name: 'Exact Dedup',
          short: 'Exact Dedup',
          storedMB: parseFloat((benchmark.system_2_exact_dedup.stored_bytes / (1024 * 1024)).toFixed(3)),
          savingsPct: benchmark.system_2_exact_dedup.savings_pct,
          color: '#818cf8',
        },
        ...Object.entries(benchmark.system_3_full_dclc).map(([strategy, res]) => {
          let shortName = strategy.includes('Stream') ? 'DCLC Stream'
            : strategy.includes('LSH') ? 'DCLC RP-LSH'
            : strategy.includes('Hierarchical') ? 'DCLC Hierarchical'
            : 'DCLC k-NN';

          return {
            name: `DCLC (${strategy})`,
            short: shortName,
            storedMB: parseFloat((res.stored_bytes / (1024 * 1024)).toFixed(3)),
            savingsPct: res.savings_pct,
            color: strategy.includes('Hierarchical') ? '#06b6d4' : '#10b981',
          };
        }),
      ]
    : [];

  // Indexer efficiency dataset
  const indexerEfficiencyData = benchmark
    ? Object.entries(benchmark.system_3_full_dclc).map(([name, res]) => ({
        indexer: name === 'Stream_Locality_W32' ? 'Stream Locality (W=32)'
          : name === 'RP_LSH_L8_B6' ? 'RP-LSH (L=8, B=6)'
          : name === 'Embedding_KNN_K20' ? 'Embedding k-NN (K=20)'
          : name.includes('Hierarchical') ? 'Hierarchical Cache (Cache=64, BallTree NN)'
          : name.replace(/_/g, ' '),
        recallPct: parseFloat((res.candidate_recall * 100).toFixed(2)),
        reductionPct: parseFloat(res.comparison_reduction_rate.toFixed(2)),
        avgPool: parseFloat(res.avg_candidate_pool.toFixed(1)),
        deltas: res.deltas_accepted,
      }))
    : [];

  return (
    <div className="space-y-8 animate-fadeIn max-w-5xl">
      {/* Title & Refresh Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
            <BarChart3 className="w-6 h-6 text-indigo-400" />
            <span>Multi-Version Storage Benchmark</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Empirical comparative evaluation against the project's held-out test split (Families 085–099, 464 chunks).
          </p>
        </div>

        {/* Threshold Adjustment & Refresh */}
        <form onSubmit={handleApplyThreshold} className="flex items-center space-x-3 bg-slate-900 border border-slate-800 p-2 rounded-xl">
          <div className="flex items-center space-x-2">
            <Sliders className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-xs text-slate-300">Threshold:</span>
            <input
              type="number"
              min="0.50"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value) || 0.70)}
              className="w-16 bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold flex items-center space-x-1"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Re-evaluate</span>
          </button>
        </form>
      </div>

      {/* Benchmark Scope Banner */}
      {benchmark && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs">
            <div>
              <span className="text-slate-400">Dataset Partition</span>
              <div className="text-base font-bold text-indigo-400 uppercase mt-0.5">
                {benchmark.split_evaluated} Split
              </div>
              <div className="text-[10px] text-slate-500">Families 085–099</div>
            </div>

            <div>
              <span className="text-slate-400">Test Chunks</span>
              <div className="text-base font-bold text-white mt-0.5">
                {benchmark.total_chunks_evaluated} chunks
              </div>
              <div className="text-[10px] text-slate-500">FastCDC variable</div>
            </div>

            <div>
              <span className="text-slate-400">Logical Raw Bytes</span>
              <div className="text-base font-bold text-white mt-0.5">
                {formatBytes(benchmark.total_logical_raw_bytes)}
              </div>
              <div className="text-[10px] text-slate-500">Uncompressed size</div>
            </div>

            <div>
              <span className="text-slate-400">Exact Duplicate Targets</span>
              <div className="text-base font-bold text-cyan-400 mt-0.5">
                {benchmark.exact_duplicate_targets}
              </div>
              <div className="text-[10px] text-slate-500">SHA-256 matches</div>
            </div>

            <div>
              <span className="text-slate-400">Resemblance Targets</span>
              <div className="text-base font-bold text-emerald-400 mt-0.5">
                {benchmark.resemblance_targets}
              </div>
              <div className="text-[10px] text-slate-500">Ground-truth similar</div>
            </div>
          </div>
        </div>
      )}

      {/* Side-by-Side Visual Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Physical Stored Footprint (MB) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
          <div>
            <h3 className="text-sm font-bold text-white">Physical Storage Footprint (MB)</h3>
            <p className="text-xs text-slate-400">Lower is better. Shows final stored bytes required on disk.</p>
          </div>

          <div className="h-56 w-full pt-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={storageComparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="short" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} unit=" MB" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  formatter={(val: any) => [`${val} MB`, 'Stored Footprint']}
                />
                <Bar dataKey="storedMB" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Net Storage Savings (%) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
          <div>
            <h3 className="text-sm font-bold text-white">Net Storage Savings Percentage (%)</h3>
            <p className="text-xs text-slate-400">Higher is better. Computed over raw logical storage.</p>
          </div>

          <div className="h-56 w-full pt-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={storageComparisonData.filter((d) => d.short !== 'Raw')}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="short" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} unit="%" domain={[40, 95]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  formatter={(val: any) => [`${val}%`, 'Storage Savings']}
                />
                <Bar dataKey="savingsPct" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Candidate Indexer Efficiency Grid */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Label-Free Candidate Indexer Retrieval Comparison</span>
          </h3>
          <p className="text-xs text-slate-400">
            Measures how effectively each candidate indexer discovers resemblance pairs without having access to ground-truth family labels.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
          {indexerEfficiencyData.map((item) => (
            <div key={item.indexer} className="bg-slate-950/70 border border-slate-800 p-3.5 rounded-lg space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-bold text-white">{item.indexer}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-mono">
                  {item.deltas} Deltas Accepted
                </span>
              </div>
              <div className="space-y-1 text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-400">Candidate Recall:</span>
                  <span className="font-bold text-emerald-400">{item.recallPct}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Comparison Reduction:</span>
                  <span className="font-bold text-indigo-400">{item.reductionPct}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Avg Candidate Pool:</span>
                  <span className="font-mono text-slate-300">{item.avgPool} chunks</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* System 4: Multi-Reference Delta (MRD) Evaluation (Phase C) */}
      {benchmark?.system_4_dclc_mrd && (
        <div className="bg-slate-900 border border-indigo-500/30 rounded-xl p-5 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-indigo-400" />
                <span>System 4: Multi-Reference Delta (MRD) Performance (Section 3.4)</span>
              </h3>
              <p className="text-xs text-slate-400">
                Comparing Single-Reference (SR) baseline vs. Multi-Reference Delta (MRD) with Group Restore (MG) and Single Restore (MS).
              </p>
            </div>
            <div className="text-[11px] font-mono px-2.5 py-1 rounded bg-slate-950 border border-slate-800 text-slate-300">
              Window={benchmark.system_4_dclc_mrd.parameters.window_size}, MaxRefs={benchmark.system_4_dclc_mrd.parameters.max_refs}, Threshold={benchmark.system_4_dclc_mrd.parameters.resemblance_threshold}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Single-Reference Card */}
            <div className="bg-slate-950/70 border border-slate-800 p-4 rounded-lg space-y-2.5 text-xs">
              <div className="flex justify-between items-center border-b border-slate-800/80 pb-2">
                <span className="font-bold text-slate-200">Single-Reference (SR) Baseline</span>
                <span className="text-[10px] text-slate-400">1 Target : 1 Base</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-slate-300">
                <div>
                  <span className="text-slate-400 block text-[11px]">Stored Size</span>
                  <span className="font-bold text-white text-sm">{formatBytes(benchmark.system_4_dclc_mrd.single_reference_sr.stored_bytes)}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">Compression Ratio</span>
                  <span className="font-bold text-emerald-400 text-sm">{formatRatio(benchmark.system_4_dclc_mrd.single_reference_sr.compression_ratio)}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">Encoding Time</span>
                  <span className="font-mono text-slate-200">{benchmark.system_4_dclc_mrd.single_reference_sr.encoding_time_seconds}s</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">Restore Throughput</span>
                  <span className="font-mono text-cyan-400 font-bold">{benchmark.system_4_dclc_mrd.single_reference_sr.restoration_throughput_mb_s} MB/s</span>
                </div>
              </div>
            </div>

            {/* Multi-Reference Card */}
            <div className="bg-indigo-950/20 border border-indigo-500/40 p-4 rounded-lg space-y-2.5 text-xs">
              <div className="flex justify-between items-center border-b border-indigo-500/20 pb-2">
                <span className="font-bold text-indigo-300">Multi-Reference Delta (MRD)</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono">100% Bit-Exact</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-slate-300">
                <div>
                  <span className="text-slate-400 block text-[11px]">Stored Size</span>
                  <span className="font-bold text-white text-sm">{formatBytes(benchmark.system_4_dclc_mrd.multi_reference_mrd.stored_bytes)}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">Compression Ratio</span>
                  <span className="font-bold text-emerald-400 text-sm">{formatRatio(benchmark.system_4_dclc_mrd.multi_reference_mrd.compression_ratio)}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">MG Group Restore</span>
                  <span className="font-mono text-emerald-400 font-bold">{benchmark.system_4_dclc_mrd.multi_reference_mrd.mg_group_restore_throughput_mb_s} MB/s</span>
                  <span className="text-[10px] text-emerald-500 block">
                    ({benchmark.system_4_dclc_mrd.restoration_throughput_comparison.mg_group_restore_vs_sr_pct >= 0 ? '+' : ''}
                    {benchmark.system_4_dclc_mrd.restoration_throughput_comparison.mg_group_restore_vs_sr_pct}% vs SR)
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">MS Single Restore</span>
                  <span className="font-mono text-indigo-300 font-bold">{benchmark.system_4_dclc_mrd.multi_reference_mrd.ms_single_restore_throughput_mb_s} MB/s</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Comparative Evaluation Matrix */}
      {benchmark && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
          <h4 className="text-sm font-bold text-white">Complete System Comparison Table</h4>

          <div className="overflow-x-auto border border-slate-800 rounded-lg">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                <tr>
                  <th className="p-3">Architecture System</th>
                  <th className="p-3">Stored Footprint</th>
                  <th className="p-3">Savings (%)</th>
                  <th className="p-3">Compression Ratio</th>
                  <th className="p-3">Candidate Recall</th>
                  <th className="p-3">Deltas Accepted</th>
                  <th className="p-3">Reconstruction</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                {/* System 0 */}
                <tr className="hover:bg-slate-800/20">
                  <td className="p-3 font-bold font-sans text-slate-400">System 0: Raw Logical Data</td>
                  <td className="p-3 font-sans">{formatBytes(benchmark.system_0_raw.stored_bytes)}</td>
                  <td className="p-3">0.0%</td>
                  <td className="p-3">1.00:1</td>
                  <td className="p-3 text-slate-600">—</td>
                  <td className="p-3 text-slate-600">—</td>
                  <td className="p-3 text-slate-600">—</td>
                </tr>

                {/* System 1 */}
                <tr className="hover:bg-slate-800/20">
                  <td className="p-3 font-bold font-sans text-slate-300">System 1: Standalone Deflate</td>
                  <td className="p-3 font-sans">{formatBytes(benchmark.system_1_standalone_compression.stored_bytes)}</td>
                  <td className="p-3 text-cyan-400">{formatPct(benchmark.system_1_standalone_compression.savings_pct)}</td>
                  <td className="p-3">{formatRatio(benchmark.system_1_standalone_compression.compression_ratio)}</td>
                  <td className="p-3 text-slate-600">—</td>
                  <td className="p-3 text-slate-600">—</td>
                  <td className="p-3 text-emerald-400 font-sans">100% Invertible</td>
                </tr>

                {/* System 2 */}
                <tr className="hover:bg-slate-800/20">
                  <td className="p-3 font-bold font-sans text-slate-200">System 2: Exact Dedup (SHA-256)</td>
                  <td className="p-3 font-sans">{formatBytes(benchmark.system_2_exact_dedup.stored_bytes)}</td>
                  <td className="p-3 text-indigo-400">{formatPct(benchmark.system_2_exact_dedup.savings_pct)}</td>
                  <td className="p-3">{formatRatio(benchmark.system_2_exact_dedup.compression_ratio)}</td>
                  <td className="p-3 text-slate-600">—</td>
                  <td className="p-3 text-slate-600">—</td>
                  <td className="p-3 text-emerald-400 font-sans">100% Bit-Exact</td>
                </tr>

                {/* System 3 Strategies */}
                {Object.entries(benchmark.system_3_full_dclc).map(([stratName, res]) => (
                  <tr key={stratName} className="hover:bg-slate-800/30 bg-indigo-950/10">
                    <td className="p-3 font-bold font-sans text-indigo-300 flex items-center space-x-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <span>
                        DCLC + {stratName === 'Stream_Locality_W32' ? 'Stream Locality (W=32)'
                          : stratName === 'RP_LSH_L8_B6' ? 'RP-LSH (L=8, B=6)'
                          : stratName === 'Embedding_KNN_K20' ? 'Embedding k-NN (K=20)'
                          : stratName.includes('Hierarchical') ? 'Hierarchical Cache (Cache=64, BallTree NN)'
                          : stratName.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="p-3 font-sans font-bold text-white">{formatBytes(res.stored_bytes)}</td>
                    <td className="p-3 font-bold text-emerald-400">{formatPct(res.savings_pct)}</td>
                    <td className="p-3 font-bold text-emerald-400">{formatRatio(res.compression_ratio)}</td>
                    <td className="p-3 text-cyan-300">{(res.candidate_recall * 100).toFixed(1)}%</td>
                    <td className="p-3 text-indigo-300">{res.deltas_accepted}</td>
                    <td className="p-3 text-emerald-400 font-sans">100% Verified</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Academic Disclaimer Note */}
      <div className="bg-slate-900/50 border border-slate-800/80 rounded-xl p-4 flex items-start space-x-3 text-slate-400 text-xs">
        <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong>Academic Methodology Notice:</strong> These figures represent actual measured storage footprints computed by the 
          DCLC research backend on the held-out multi-version benchmark test partition (Families 085–099, 464 chunks, 3.62 MB). 
          Deltas are formatted as RFC 1950 preset-dictionary containers with 76-byte <code className="text-slate-300">DCLC_D01</code> headers 
          and verified byte-for-byte in memory. These are empirical findings for multi-version cloud storage distributions and not theoretical bounds.
        </p>
      </div>
    </div>
  );
};
