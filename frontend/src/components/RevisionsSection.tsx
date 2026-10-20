import { useState, useEffect, type FC, type ChangeEvent } from 'react';
import {
  Archive,
  Download,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  HardDrive,
  FileCheck,
  FileText,
  Layers,
  Sparkles,
} from 'lucide-react';
import { api } from '../services/api';
import type { StoredRevisionSummary, StoredRevisionDetail, RestoreRevisionResponse } from '../types/api';
import { formatBytes, formatPct } from '../utils/formatters';

export const RevisionsSection: FC = () => {
  const [revisions, setRevisions] = useState<StoredRevisionSummary[]>([]);
  const [loadingList, setLoadingList] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  // New revision form state
  const [baseFile, setBaseFile] = useState<File | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [candidateStrategy, setCandidateStrategy] = useState<string>('hierarchical_cache');
  const [deltaMode, setDeltaMode] = useState<'single' | 'multi'>('single');
  const [mrdWindowSize, setMrdWindowSize] = useState<number>(3);
  const [mrdMaxRefs, setMrdMaxRefs] = useState<number>(3);
  const [resemblanceThreshold, setResemblanceThreshold] = useState<number>(0.50);
  const [storing, setStoring] = useState(false);
  const [storeSuccess, setStoreSuccess] = useState<StoredRevisionDetail | null>(null);
  const [storeError, setStoreError] = useState<string | null>(null);

  // Verification & manifest inspection state
  const [restoringId, setRestoringId] = useState<string | null>(null);
  const [restoreResults, setRestoreResults] = useState<Record<string, RestoreRevisionResponse>>({});
  const [activeManifest, setActiveManifest] = useState<StoredRevisionDetail | null>(null);
  const [loadingManifest, setLoadingManifest] = useState(false);

  const fetchRevisions = async () => {
    setLoadingList(true);
    setListError(null);
    try {
      const data = await api.listRevisions();
      setRevisions(data);
    } catch (err: unknown) {
      setListError(err instanceof Error ? err.message : 'Failed to fetch stored revisions');
    } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => {
    fetchRevisions();
  }, []);

  const handleBaseChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setBaseFile(e.target.files[0]);
    }
  };

  const handleTargetChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setTargetFile(e.target.files[0]);
    }
  };

  const handleStore = async () => {
    if (!baseFile || !targetFile) {
      setStoreError('Please provide both base reference and target revision files.');
      return;
    }
    setStoring(true);
    setStoreError(null);
    setStoreSuccess(null);
    try {
      const recipe = await api.storeRevision(
        baseFile,
        targetFile,
        candidateStrategy,
        deltaMode,
        mrdWindowSize,
        mrdMaxRefs,
        resemblanceThreshold,
      );
      setStoreSuccess(recipe);
      await fetchRevisions();
    } catch (err: unknown) {
      setStoreError(err instanceof Error ? err.message : 'Error storing revision');
    } finally {
      setStoring(false);
    }
  };

  const handleRestore = async (revId: string) => {
    setRestoringId(revId);
    try {
      const res = await api.restoreRevision(revId);
      setRestoreResults((prev) => ({ ...prev, [revId]: res }));
    } catch (err: unknown) {
      alert(`Restoration failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRestoringId(null);
    }
  };

  const handleInspectManifest = async (revId: string) => {
    setLoadingManifest(true);
    try {
      const detail = await api.getRevision(revId);
      setActiveManifest(detail);
    } catch (err: unknown) {
      alert(`Failed to load recipe: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoadingManifest(false);
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn max-w-6xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
          <Archive className="h-7 w-7 text-indigo-400" />
          Durable Revision Storage & Lifecycle
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Content-addressed chunk/delta store with atomic recipes and restart-safe byte-exact reconstruction (Phase E).
        </p>
      </div>

      {/* Form: Persist New Revision */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <HardDrive className="h-5 w-5 text-indigo-400" />
          Persist New Revision Against Base Anchor
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">
              Base Reference File
            </label>
            <input
              type="file"
              onChange={handleBaseChange}
              className="w-full text-sm text-slate-300 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700 cursor-pointer bg-slate-950 p-2 rounded-lg border border-slate-800"
            />
            {baseFile && (
              <p className="text-xs text-slate-400 mt-1">
                Selected: <span className="text-indigo-300 font-mono">{baseFile.name}</span> ({formatBytes(baseFile.size)})
              </p>
            )}
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">
              Target Revision File
            </label>
            <input
              type="file"
              onChange={handleTargetChange}
              className="w-full text-sm text-slate-300 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700 cursor-pointer bg-slate-950 p-2 rounded-lg border border-slate-800"
            />
            {targetFile && (
              <p className="text-xs text-slate-400 mt-1">
                Selected: <span className="text-indigo-300 font-mono">{targetFile.name}</span> ({formatBytes(targetFile.size)})
              </p>
            )}
          </div>
        </div>

        {/* Strategy and Delta Mode Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4 pt-2 border-t border-slate-800">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Candidate Indexer</label>
            <select
              value={candidateStrategy}
              onChange={(e) => setCandidateStrategy(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 text-xs text-slate-200 rounded-lg p-2 focus:ring-1 focus:ring-indigo-500"
            >
              <option value="hierarchical_cache">Hierarchical Cache + BallTree (Sec 3.3)</option>
              <option value="stream_locality">Stream Locality Window (W=32)</option>
              <option value="rp_lsh">Random Projection LSH</option>
              <option value="cosine_knn">Embedding Cosine k-NN</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Delta Compression Mode</label>
            <select
              value={deltaMode}
              onChange={(e) => setDeltaMode(e.target.value as 'single' | 'multi')}
              className="w-full bg-slate-950 border border-slate-800 text-xs text-slate-200 rounded-lg p-2 focus:ring-1 focus:ring-indigo-500"
            >
              <option value="single">Single-Reference Delta (SR)</option>
              <option value="multi">Multi-Reference Delta (MRD Sec 3.4)</option>
            </select>
          </div>

          {deltaMode === 'multi' && (
            <>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">MRD Window (Targets): {mrdWindowSize}</label>
                <input
                  type="range"
                  min={1}
                  max={8}
                  value={mrdWindowSize}
                  onChange={(e) => setMrdWindowSize(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 mt-2"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Max References: {mrdMaxRefs}</label>
                <input
                  type="range"
                  min={1}
                  max={6}
                  value={mrdMaxRefs}
                  onChange={(e) => setMrdMaxRefs(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 mt-2"
                />
              </div>
            </>
          )}

          {deltaMode === 'single' && (
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Resemblance Threshold: {resemblanceThreshold.toFixed(2)}</label>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.05}
                value={resemblanceThreshold}
                onChange={(e) => setResemblanceThreshold(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 mt-2"
              />
            </div>
          )}
        </div>

        {storeError && (
          <div className="mb-4 p-3 bg-red-950/40 border border-red-800/60 rounded-lg text-xs text-red-300 flex items-center gap-2">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{storeError}</span>
          </div>
        )}

        {storeSuccess && (
          <div className="mb-4 p-3 bg-emerald-950/40 border border-emerald-800/60 rounded-lg text-xs text-emerald-300 flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 shrink-0" />
            <span>
              Revision <span className="font-mono font-bold">{storeSuccess.revision_id}</span> persisted successfully! Stored: {formatBytes(storeSuccess.stored_bytes)} ({formatPct(storeSuccess.storage_savings_pct)} savings).
            </span>
          </div>
        )}

        <button
          onClick={handleStore}
          disabled={storing || !baseFile || !targetFile}
          className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg flex items-center gap-2 transition-all shadow"
        >
          {storing ? (
            <>
              <RefreshCw className="h-4 w-4 animate-spin" />
              Compressing & Writing Atomic Recipe...
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              Compress & Durably Persist Revision
            </>
          )}
        </button>
      </div>

      {/* Persisted Revisions Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Layers className="h-5 w-5 text-indigo-400" />
            Durable Revisions Manifest ({revisions.length})
          </h2>
          <button
            onClick={fetchRevisions}
            disabled={loadingList}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded-lg flex items-center gap-1.5 transition"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loadingList ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {listError && (
          <div className="mb-4 p-3 bg-red-950/40 border border-red-800/60 rounded-lg text-xs text-red-300">
            {listError}
          </div>
        )}

        {revisions.length === 0 ? (
          <div className="text-center py-10 border border-dashed border-slate-800 rounded-lg text-slate-400 text-xs">
            No revisions persisted yet. Compress and store a revision above to see it listed here.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950 text-slate-400 uppercase font-semibold text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-3 px-3">Revision ID / Name</th>
                  <th className="py-3 px-3">Base Anchor</th>
                  <th className="py-3 px-3">Mode</th>
                  <th className="py-3 px-3">Raw Size</th>
                  <th className="py-3 px-3">Stored Size</th>
                  <th className="py-3 px-3">Savings</th>
                  <th className="py-3 px-3">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {revisions.map((rev) => {
                  const restoreRes = restoreResults[rev.revision_id];
                  const isRestoring = restoringId === rev.revision_id;

                  return (
                    <tr key={rev.revision_id} className="hover:bg-slate-800/30 transition">
                      <td className="py-3 px-3">
                        <div className="font-mono text-indigo-300 font-semibold">{rev.revision_id}</div>
                        <div className="text-[11px] text-slate-400 truncate max-w-xs">{rev.target_name}</div>
                      </td>
                      <td className="py-3 px-3 font-mono text-slate-400 text-[11px] truncate max-w-[140px]">
                        {rev.base_name}
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                            rev.delta_mode === 'multi'
                              ? 'bg-purple-900/60 text-purple-300 border border-purple-700/50'
                              : 'bg-blue-900/60 text-blue-300 border border-blue-700/50'
                          }`}
                        >
                          {rev.delta_mode === 'multi' ? 'MRD' : 'SR'}
                        </span>
                      </td>
                      <td className="py-3 px-3 font-mono">{formatBytes(rev.raw_bytes)}</td>
                      <td className="py-3 px-3 font-mono text-emerald-400 font-semibold">{formatBytes(rev.stored_bytes)}</td>
                      <td className="py-3 px-3">
                        <span className="font-semibold text-emerald-400">
                          {formatPct(rev.storage_savings_pct)}
                        </span>
                      </td>
                      <td className="py-3 px-3 space-x-1.5 whitespace-nowrap">
                        <button
                          onClick={() => handleRestore(rev.revision_id)}
                          disabled={isRestoring}
                          className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-indigo-300 text-[11px] rounded font-medium inline-flex items-center gap-1 transition"
                          title="Restore and verify cryptographic SHA-256 match"
                        >
                          {isRestoring ? (
                            <RefreshCw className="h-3 w-3 animate-spin" />
                          ) : restoreRes?.verified ? (
                            <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                          ) : (
                            <FileCheck className="h-3 w-3" />
                          )}
                          {restoreRes?.verified ? 'Verified' : 'Verify'}
                        </button>

                        <a
                          href={api.getDownloadRevisionUrl(rev.revision_id)}
                          download={rev.target_name}
                          className="px-2 py-1 bg-indigo-900/50 hover:bg-indigo-900 text-indigo-200 text-[11px] rounded font-medium inline-flex items-center gap-1 transition border border-indigo-700/40"
                          title="Download reconstructed binary file"
                        >
                          <Download className="h-3 w-3" />
                          Download
                        </a>

                        <button
                          onClick={() => handleInspectManifest(rev.revision_id)}
                          disabled={loadingManifest}
                          className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] rounded font-medium inline-flex items-center gap-1 transition disabled:opacity-50"
                          title="Inspect chunk recipe breakdown"
                        >
                          <FileText className="h-3 w-3" />
                          Recipe
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Manifest Inspection Modal / Panel */}
      {activeManifest && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
          <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <FileText className="h-4 w-4 text-indigo-400" />
              Manifest Recipe: <span className="font-mono text-indigo-300">{activeManifest.revision_id}</span>
            </h3>
            <button
              onClick={() => setActiveManifest(null)}
              className="text-xs text-slate-400 hover:text-white"
            >
              Close
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs mb-4">
            <div className="bg-slate-950 p-2 rounded border border-slate-800">
              <span className="text-slate-400 block text-[10px]">TOTAL CHUNKS</span>
              <span className="font-semibold text-white">{activeManifest.chunk_count}</span>
            </div>
            <div className="bg-slate-950 p-2 rounded border border-slate-800">
              <span className="text-slate-400 block text-[10px]">EXACT DUPLICATES</span>
              <span className="font-semibold text-emerald-400">{activeManifest.exact_duplicates_count}</span>
            </div>
            <div className="bg-slate-950 p-2 rounded border border-slate-800">
              <span className="text-slate-400 block text-[10px]">DELTA ENCODED</span>
              <span className="font-semibold text-indigo-400">{activeManifest.delta_chunks_count}</span>
            </div>
            <div className="bg-slate-950 p-2 rounded border border-slate-800">
              <span className="text-slate-400 block text-[10px]">STANDALONE COMPRESSED</span>
              <span className="font-semibold text-amber-400">{activeManifest.standalone_chunks_count}</span>
            </div>
          </div>

          <div className="max-h-60 overflow-y-auto border border-slate-800 rounded-lg">
            <table className="w-full text-left text-[11px] text-slate-300">
              <thead className="bg-slate-950 text-slate-400 uppercase text-[9px] sticky top-0 border-b border-slate-800">
                <tr>
                  <th className="py-2 px-2">#</th>
                  <th className="py-2 px-2">Type</th>
                  <th className="py-2 px-2">Chunk SHA-256</th>
                  <th className="py-2 px-2">Raw Size</th>
                  <th className="py-2 px-2">Stored Size</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {activeManifest.chunks.map((c, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/30">
                    <td className="py-1.5 px-2 font-mono text-slate-400">{c.chunk_index}</td>
                    <td className="py-1.5 px-2">
                      <span
                        className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-bold ${
                          c.type === 'exact'
                            ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/50'
                            : c.type.startsWith('delta')
                            ? 'bg-indigo-950 text-indigo-300 border border-indigo-800/50'
                            : 'bg-amber-950 text-amber-300 border border-amber-800/50'
                        }`}
                      >
                        {c.type}
                      </span>
                    </td>
                    <td className="py-1.5 px-2 font-mono text-[10px] text-slate-400">{c.chunk_sha256.slice(0, 16)}...</td>
                    <td className="py-1.5 px-2 font-mono">{c.chunk_size} B</td>
                    <td className="py-1.5 px-2 font-mono text-emerald-400">{c.stored_size} B</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
