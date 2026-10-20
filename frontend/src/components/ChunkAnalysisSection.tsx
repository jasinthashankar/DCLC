import { useState, type FC, type DragEvent, type FormEvent } from 'react';
import {
  Scissors,
  Upload,
  AlertCircle,
  Copy,
  Check,
  BarChart2,
  Sliders,
  ChevronDown,
  ChevronUp,
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
import type { ChunkPreviewResponse } from '../types/api';
import { api } from '../services/api';
import { formatBytes, truncateHash } from '../utils/formatters';

export const ChunkAnalysisSection: FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [minChunkSize, setMinChunkSize] = useState<number>(4096);
  const [avgChunkSize, setAvgChunkSize] = useState<number>(8192);
  const [maxChunkSize, setMaxChunkSize] = useState<number>(16384);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [previewData, setPreviewData] = useState<ChunkPreviewResponse | null>(null);
  const [copiedHash, setCopiedHash] = useState<string | null>(null);
  const [searchHash, setSearchHash] = useState<string>('');

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
  };

  const handleChunkFile = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file to chunk.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const resp = await api.previewChunks(file, minChunkSize, avgChunkSize, maxChunkSize);
      setPreviewData(resp);
    } catch (err: any) {
      setError(err?.detail || err?.message || 'Failed to process FastCDC chunk analysis.');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedHash(text);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  // Prepare chart data (sampled if > 60 chunks)
  const chartData = previewData?.chunks.map((c) => ({
    chunk: `#${c.chunk_index}`,
    bytes: c.byte_length,
    kb: parseFloat((c.byte_length / 1024).toFixed(2)),
  })) || [];

  const filteredChunks = previewData?.chunks.filter(
    (c) =>
      c.chunk_index.toString().includes(searchHash) ||
      c.sha256.toLowerCase().includes(searchHash.toLowerCase())
  ) || [];

  return (
    <div className="space-y-8 animate-fadeIn max-w-5xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
          <Scissors className="w-6 h-6 text-indigo-400" />
          <span>Content-Defined Chunk Analysis (FastCDC)</span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Inspect how FastCDC partitions arbitrary binary or text files into variable-sized chunks based on content boundaries.
        </p>
      </div>

      {/* Educational Note */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-start space-x-3 text-xs text-slate-300">
        <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed text-slate-400">
          <strong className="text-slate-200">Why Content-Defined Chunking?</strong> Fixed-size chunking (e.g. cutting every 8 KB) suffers from the 
          <em> "boundary shift problem"</em>: inserting a single byte at the beginning of a file shifts every subsequent chunk boundary, causing 100% of chunks to miss deduplication. 
          <strong> FastCDC</strong> solves this by using a fast gear-based rolling hash that detects cut boundaries from the byte contents themselves.
        </p>
      </div>

      {/* Upload and Configuration Form */}
      <form onSubmit={handleChunkFile} className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5 shadow-sm">
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-6 text-center transition-all ${
            file
              ? 'border-indigo-500/50 bg-indigo-500/5'
              : 'border-slate-800 hover:border-slate-700 bg-slate-950/40'
          }`}
        >
          <input
            type="file"
            id="analysis-file-input"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
          />
          <label htmlFor="analysis-file-input" className="cursor-pointer block space-y-2">
            <div className="w-10 h-10 mx-auto rounded-full bg-slate-800 flex items-center justify-center text-indigo-400">
              <Upload className="w-5 h-5" />
            </div>
            <div>
              <span className="text-sm font-semibold text-white block">
                {file ? file.name : 'Select or Drop File to Chunk'}
              </span>
              <span className="text-xs text-slate-400 mt-0.5 block">
                {file ? `${formatBytes(file.size)}` : 'Supports binaries, documents, archives, source files'}
              </span>
            </div>
          </label>
          {file && (
            <button
              type="button"
              onClick={() => setFile(null)}
              className="mt-2 text-xs text-rose-400 hover:text-rose-300 underline"
            >
              Clear
            </button>
          )}
        </div>

        {/* Parameters Accordion */}
        <div className="pt-1">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>FastCDC Size Parameters (Default: 4KB / 8KB / 16KB)</span>
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

        {error && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg flex items-center space-x-2 text-rose-400 text-xs">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="flex justify-end pt-1 border-t border-slate-800">
          <button
            type="submit"
            disabled={loading || !file}
            className="px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold shadow-md transition-all flex items-center space-x-2"
          >
            {loading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                <span>Chunking with FastCDC...</span>
              </>
            ) : (
              <>
                <span>Analyze Chunks</span>
                <Scissors className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </form>

      {/* Chunking Results */}
      {previewData && (
        <div className="space-y-6 animate-fadeIn">
          {/* Summary Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
              <span className="text-slate-400">Total File Size</span>
              <div className="text-base font-bold text-white mt-1">{formatBytes(previewData.total_bytes)}</div>
              <div className="text-[10px] text-slate-500 truncate mt-0.5">{previewData.filename}</div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
              <span className="text-slate-400">Total Chunks</span>
              <div className="text-base font-bold text-indigo-400 mt-1">{previewData.chunk_count}</div>
              <div className="text-[10px] text-slate-500 mt-0.5">Content-defined</div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
              <span className="text-slate-400">Min Chunk Size</span>
              <div className="text-base font-bold text-slate-200 mt-1">{formatBytes(previewData.min_chunk_bytes)}</div>
              <div className="text-[10px] text-slate-500 mt-0.5">Floor limit</div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
              <span className="text-slate-400">Average Chunk Size</span>
              <div className="text-base font-bold text-slate-200 mt-1">{formatBytes(previewData.avg_chunk_bytes)}</div>
              <div className="text-[10px] text-slate-500 mt-0.5">Target: {formatBytes(avgChunkSize)}</div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
              <span className="text-slate-400">Max Chunk Size</span>
              <div className="text-base font-bold text-slate-200 mt-1">{formatBytes(previewData.max_chunk_bytes)}</div>
              <div className="text-[10px] text-slate-500 mt-0.5">Ceiling cut-off</div>
            </div>
          </div>

          {/* Size Distribution Chart */}
          {chartData.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                    <BarChart2 className="w-4 h-4 text-cyan-400" />
                    <span>Chunk Size Distribution Across File (KB)</span>
                  </h3>
                  <p className="text-xs text-slate-400">Sequential FastCDC chunk lengths</p>
                </div>
              </div>

              <div className="h-44 w-full pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData.slice(0, 60)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="chunk" stroke="#64748b" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 10 }} unit=" KB" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                      formatter={(val: any) => [`${val} KB`, 'Chunk Length']}
                    />
                    <Bar dataKey="kb" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Granular Chunk Metadata Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <h4 className="text-sm font-bold text-white">Extracted Chunks ({previewData.chunks.length})</h4>
              <input
                type="text"
                placeholder="Search index or SHA-256..."
                value={searchHash}
                onChange={(e) => setSearchHash(e.target.value)}
                className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div className="overflow-x-auto border border-slate-800 rounded-lg">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="p-3">Index</th>
                    <th className="p-3">Byte Offset</th>
                    <th className="p-3">Length</th>
                    <th className="p-3">SHA-256 Fingerprint</th>
                    <th className="p-3 text-right">Copy</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                  {filteredChunks.map((chunk) => (
                    <tr key={chunk.chunk_index} className="hover:bg-slate-800/30 transition-colors">
                      <td className="p-3 text-slate-400 font-bold">#{chunk.chunk_index}</td>
                      <td className="p-3">{chunk.byte_offset.toLocaleString()} B</td>
                      <td className="p-3 font-sans text-slate-200">{formatBytes(chunk.byte_length)}</td>
                      <td className="p-3 text-slate-400">
                        <span title={chunk.sha256}>{truncateHash(chunk.sha256, 12, 12)}</span>
                      </td>
                      <td className="p-3 text-right font-sans">
                        <button
                          onClick={() => copyToClipboard(chunk.sha256)}
                          className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 inline-flex items-center space-x-1"
                          title="Copy SHA-256"
                        >
                          {copiedHash === chunk.sha256 ? (
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
