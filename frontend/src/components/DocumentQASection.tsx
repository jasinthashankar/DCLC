import { useState, useEffect, useCallback, type FC, type ChangeEvent } from 'react';
import {
  BookOpen,
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Search,
  Sliders,
  ShieldCheck,
  Clock,
  Database,
  RefreshCw,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { api } from '../services/api';
import type {
  RAGDocumentsResponse,
  RAGDocumentSummary,
  RAGIngestResponse,
  RAGQueryResponse,
  RAGStatusResponse,
} from '../types/api';

export const DocumentQASection: FC = () => {
  // RAG State
  const [status, setStatus] = useState<RAGStatusResponse | null>(null);
  const [documentsData, setDocumentsData] = useState<RAGDocumentsResponse | null>(null);
  const [loadingInitial, setLoadingInitial] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Ingestion State
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [ingestMode, setIngestMode] = useState<'new' | 'version'>('new');
  const [selectedTargetDocId, setSelectedTargetDocId] = useState<string>('');
  const [minChunkSize, setMinChunkSize] = useState<number>(64);
  const [avgChunkSize, setAvgChunkSize] = useState<number>(256);
  const [maxChunkSize, setMaxChunkSize] = useState<number>(1024);
  const [showChunkSettings, setShowChunkSettings] = useState<boolean>(false);
  const [ingesting, setIngesting] = useState<boolean>(false);
  const [ingestResult, setIngestResult] = useState<RAGIngestResponse | null>(null);

  // Question Answering Query State
  const [queryText, setQueryText] = useState<string>('');
  const [scopeDocId, setScopeDocId] = useState<string>('all');
  const [scopeVersionId, setScopeVersionId] = useState<string>('all');
  const [topK, setTopK] = useState<number>(5);
  const [alpha, setAlpha] = useState<number>(0.5);
  const [showAdvancedParams, setShowAdvancedParams] = useState<boolean>(false);
  const [querying, setQuerying] = useState<boolean>(false);
  const [queryResponse, setQueryResponse] = useState<RAGQueryResponse | null>(null);

  // Refresh status and document list
  const loadData = useCallback(async () => {
    setErrorMsg(null);
    try {
      const [statusRes, docsRes] = await Promise.all([
        api.getRAGStatus(),
        api.getRAGDocuments(),
      ]);
      setStatus(statusRes);
      setDocumentsData(docsRes);
      if (docsRes.documents.length > 0 && !selectedTargetDocId) {
        setSelectedTargetDocId(docsRes.documents[0].doc_id);
      }
    } catch (err: any) {
      setErrorMsg(err?.detail || err?.message || 'Failed to load RAG system state.');
    } finally {
      setLoadingInitial(false);
      setRefreshing(false);
    }
  }, [selectedTargetDocId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  // Handle File Ingest
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setUploadFile(e.target.files[0]);
      setIngestResult(null);
    }
  };

  const handleIngest = async () => {
    if (!uploadFile) return;
    setIngesting(true);
    setErrorMsg(null);
    setIngestResult(null);

    const docIdParam = ingestMode === 'version' && selectedTargetDocId ? selectedTargetDocId : null;

    try {
      const result = await api.ingestRAGDocument(
        uploadFile,
        docIdParam,
        minChunkSize,
        avgChunkSize,
        maxChunkSize
      );
      setIngestResult(result);
      setUploadFile(null);
      // Reset input element
      const fileInput = document.getElementById('rag-file-input') as HTMLInputElement;
      if (fileInput) fileInput.value = '';
      await loadData();
    } catch (err: any) {
      setErrorMsg(err?.detail || err?.message || 'Failed to ingest document.');
    } finally {
      setIngesting(false);
    }
  };

  // Handle Query
  const handleAskQuestion = async () => {
    const trimmed = queryText.trim();
    if (!trimmed) return;

    setQuerying(true);
    setErrorMsg(null);
    setQueryResponse(null);

    const docId = scopeDocId !== 'all' ? scopeDocId : null;
    const versionId = scopeVersionId !== 'all' ? scopeVersionId : null;

    try {
      const resp = await api.queryRAG({
        query: trimmed,
        doc_id: docId,
        version_id: versionId,
        top_k: topK,
        alpha: alpha,
      });
      setQueryResponse(resp);
    } catch (err: any) {
      setErrorMsg(err?.detail || err?.message || 'Failed to query RAG engine.');
    } finally {
      setQuerying(false);
    }
  };

  // Compute available versions for currently selected doc scope
  const activeDocSummary: RAGDocumentSummary | undefined =
    scopeDocId !== 'all'
      ? documentsData?.documents.find((d) => d.doc_id === scopeDocId)
      : undefined;

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start space-x-3.5">
            <div className="w-10 h-10 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <h1 className="text-xl font-bold text-white tracking-tight">
                  Document Q&amp;A (DCLC-RAG)
                </h1>
                <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Version-Aware
                </span>
                {status?.embedding_precision && (
                  <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 uppercase">
                    Precision: {status.embedding_precision}
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
                Grounded question answering with deduplication-informed hybrid retrieval (BM25 + Dense Cosine),
                strict version isolation, and application-validated source citations.
              </p>
            </div>
          </div>

          <button
            onClick={handleRefresh}
            disabled={refreshing || loadingInitial}
            className="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700/60 flex items-center space-x-1.5 self-start md:self-auto transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            <span>Refresh State</span>
          </button>
        </div>

        {/* Global Pipeline Metrics */}
        {status && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mt-6 pt-5 border-t border-slate-800/80">
            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
              <div className="text-[10px] uppercase font-bold text-slate-500">Pipeline Status</div>
              <div className="text-sm font-semibold text-slate-200 capitalize mt-0.5 flex items-center space-x-1.5">
                <span className={`w-2 h-2 rounded-full ${status.status === 'ready' ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                <span>{status.status}</span>
              </div>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
              <div className="text-[10px] uppercase font-bold text-slate-500">LLM Provider</div>
              <div className="text-sm font-semibold text-indigo-300 truncate mt-0.5" title={status.model_name}>
                {status.provider_name}
              </div>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
              <div className="text-[10px] uppercase font-bold text-slate-500">Documents / Vers.</div>
              <div className="text-sm font-semibold text-slate-200 mt-0.5">
                {status.total_documents} <span className="text-slate-500 font-normal">/</span> {status.total_versions}
              </div>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
              <div className="text-[10px] uppercase font-bold text-slate-500">Unique Chunks</div>
              <div className="text-sm font-semibold text-slate-200 mt-0.5">
                {status.total_unique_chunks}
              </div>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
              <div className="text-[10px] uppercase font-bold text-slate-500">Dense Embeddings</div>
              <div className="text-sm font-semibold text-slate-200 mt-0.5">
                {status.dense_embeddings_count}{' '}
                <span className="text-slate-500 text-[10px]">
                  ({status.dense_dimension}d{status.embedding_precision ? ` • ${status.embedding_precision.toUpperCase()}` : ''})
                </span>
              </div>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
              <div className="text-[10px] uppercase font-bold text-slate-500">Vector Dedup Savings</div>
              <div className="text-sm font-bold text-emerald-400 mt-0.5">
                {status.dedup_savings_pct.toFixed(1)}%
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error Alert */}
      {errorMsg && (
        <div className="bg-rose-950/70 border border-rose-500/40 rounded-xl p-4 flex items-start space-x-3 text-rose-300 text-xs">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold">Notice:</span> {errorMsg}
          </div>
        </div>
      )}

      {/* Main 2-Column Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Document Ingestion & Registry (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Upload Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
              <UploadCloud className="w-4 h-4 text-indigo-400" />
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                Ingest Document
              </h2>
            </div>

            {/* Ingestion Mode Toggle */}
            <div className="flex items-center space-x-2 bg-slate-950/60 p-1 rounded-lg border border-slate-800 text-xs">
              <button
                type="button"
                onClick={() => setIngestMode('new')}
                className={`flex-1 py-1.5 px-2.5 rounded-md font-medium transition-all ${
                  ingestMode === 'new'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                New Document
              </button>
              <button
                type="button"
                onClick={() => setIngestMode('version')}
                disabled={!documentsData || documentsData.documents.length === 0}
                className={`flex-1 py-1.5 px-2.5 rounded-md font-medium transition-all ${
                  ingestMode === 'version'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 disabled:opacity-40'
                }`}
              >
                New Version
              </button>
            </div>

            {/* Target Document Selector for Version Mode */}
            {ingestMode === 'version' && documentsData && documentsData.documents.length > 0 && (
              <div className="space-y-1.5">
                <label className="text-[11px] font-medium text-slate-300">
                  Target Logical Document
                </label>
                <select
                  value={selectedTargetDocId}
                  onChange={(e) => setSelectedTargetDocId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  {documentsData.documents.map((doc) => (
                    <option key={doc.doc_id} value={doc.doc_id}>
                      {doc.filename} ({doc.version_count} version{doc.version_count > 1 ? 's' : ''})
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* File Upload Drop Area */}
            <div className="space-y-2">
              <label className="text-[11px] font-medium text-slate-300">
                Choose Document File (.txt, .md, .py)
              </label>
              <input
                id="rag-file-input"
                type="file"
                accept=".txt,.md,.py"
                onChange={handleFileChange}
                className="w-full text-xs text-slate-400 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 file:cursor-pointer cursor-pointer bg-slate-950 p-2 rounded-lg border border-slate-800"
              />
            </div>

            {/* Collapsible Chunk Settings */}
            <div>
              <button
                type="button"
                onClick={() => setShowChunkSettings(!showChunkSettings)}
                className="flex items-center space-x-1.5 text-[11px] text-slate-400 hover:text-slate-200 transition-colors"
              >
                <Sliders className="w-3 h-3" />
                <span>FastCDC Granularity Settings</span>
                {showChunkSettings ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>

              {showChunkSettings && (
                <div className="grid grid-cols-3 gap-2 mt-3 p-3 bg-slate-950 rounded-lg border border-slate-800/80 text-[11px]">
                  <div>
                    <label className="text-slate-500 block">Min (B)</label>
                    <input
                      type="number"
                      value={minChunkSize}
                      onChange={(e) => setMinChunkSize(Number(e.target.value))}
                      className="w-full bg-slate-900 border border-slate-700/60 rounded px-2 py-1 text-slate-200 mt-1"
                    />
                  </div>
                  <div>
                    <label className="text-slate-500 block">Avg (B)</label>
                    <input
                      type="number"
                      value={avgChunkSize}
                      onChange={(e) => setAvgChunkSize(Number(e.target.value))}
                      className="w-full bg-slate-900 border border-slate-700/60 rounded px-2 py-1 text-slate-200 mt-1"
                    />
                  </div>
                  <div>
                    <label className="text-slate-500 block">Max (B)</label>
                    <input
                      type="number"
                      value={maxChunkSize}
                      onChange={(e) => setMaxChunkSize(Number(e.target.value))}
                      className="w-full bg-slate-900 border border-slate-700/60 rounded px-2 py-1 text-slate-200 mt-1"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Ingest Action Button */}
            <button
              onClick={handleIngest}
              disabled={!uploadFile || ingesting}
              className="w-full py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md shadow-indigo-600/20 flex items-center justify-center space-x-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {ingesting ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Processing FastCDC &amp; Indexing...</span>
                </>
              ) : (
                <>
                  <UploadCloud className="w-3.5 h-3.5" />
                  <span>
                    {ingestMode === 'new' ? 'Ingest as New Document' : 'Register New Revision'}
                  </span>
                </>
              )}
            </button>

            {/* Ingestion Success Feedback */}
            {ingestResult && (
              <div className="bg-emerald-950/60 border border-emerald-500/40 rounded-lg p-3 text-xs text-emerald-300 space-y-1">
                <div className="flex items-center space-x-1.5 font-semibold">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>{ingestResult.message}</span>
                </div>
                <div className="text-[11px] text-emerald-400/80 font-mono pl-5">
                  Version SHA: {ingestResult.version_id} ({ingestResult.total_bytes} bytes, {ingestResult.chunk_count} chunks)
                </div>
              </div>
            )}
          </div>

          {/* Registered Documents List Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <Database className="w-4 h-4 text-indigo-400" />
                <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                  Knowledge Base ({documentsData?.total_documents || 0})
                </h2>
              </div>
              <span className="text-[11px] text-slate-500">
                {documentsData?.total_versions || 0} total revisions
              </span>
            </div>

            {documentsData && documentsData.documents.length === 0 ? (
              <div className="text-center py-8 text-xs text-slate-500">
                No documents ingested yet. Upload a .txt, .md, or .py file to begin.
              </div>
            ) : (
              <div className="space-y-3 max-h-[380px] overflow-y-auto pr-1">
                {documentsData?.documents.map((doc) => (
                  <div
                    key={doc.doc_id}
                    className="p-3 bg-slate-950/80 border border-slate-800/80 rounded-lg space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <FileText className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                        <span className="text-xs font-semibold text-slate-200 truncate">
                          {doc.filename}
                        </span>
                      </div>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">
                        {doc.version_count} vers.
                      </span>
                    </div>

                    <div className="space-y-1 pl-5 border-l border-slate-800/80">
                      {doc.versions.map((ver, idx) => (
                        <div
                          key={ver.version_id}
                          className="flex items-center justify-between text-[11px] text-slate-400 py-0.5"
                        >
                          <div className="flex items-center space-x-1.5">
                            <span className="text-slate-500 font-mono">v{idx + 1}:</span>
                            <span className="font-mono text-slate-300">
                              {ver.version_id.substring(0, 8)}
                            </span>
                          </div>
                          <span className="text-slate-500 text-[10px]">
                            {ver.chunk_count} chunks ({ver.total_bytes} B)
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Question Answering Panel (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Query Formulation Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
              <Search className="w-4 h-4 text-indigo-400" />
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                Grounded Question Answering
              </h2>
            </div>

            {/* Scope Selection Row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* Document Scope */}
              <div className="space-y-1">
                <label className="text-[11px] font-medium text-slate-300 flex items-center space-x-1">
                  <span>Document Scope</span>
                </label>
                <select
                  value={scopeDocId}
                  onChange={(e) => {
                    setScopeDocId(e.target.value);
                    setScopeVersionId('all'); // Reset version when doc changes
                  }}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="all">All Registered Documents</option>
                  {documentsData?.documents.map((doc) => (
                    <option key={doc.doc_id} value={doc.doc_id}>
                      {doc.filename}
                    </option>
                  ))}
                </select>
              </div>

              {/* Version Scope (Strict Version Isolation) */}
              <div className="space-y-1">
                <label className="text-[11px] font-medium text-slate-300 flex items-center space-x-1">
                  <span>Version Scope</span>
                  <span className="text-[10px] text-indigo-400">(Isolation)</span>
                </label>
                <select
                  value={scopeVersionId}
                  onChange={(e) => setScopeVersionId(e.target.value)}
                  disabled={scopeDocId === 'all'}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 disabled:opacity-40"
                >
                  <option value="all">All Versions (Cross-Version)</option>
                  {activeDocSummary?.versions.map((ver, idx) => (
                    <option key={ver.version_id} value={ver.version_id}>
                      v{idx + 1} ({ver.version_id.substring(0, 8)}...) - {ver.chunk_count} chunks
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Query Input */}
            <div className="space-y-2">
              <label className="text-[11px] font-medium text-slate-300">
                Question to Answer from Retrieved Chunks
              </label>
              <textarea
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                placeholder="e.g. What is the configured memory allocation and storage algorithm?"
                rows={3}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 resize-none font-sans"
              />
            </div>

            {/* Advanced Retrieval Sliders */}
            <div>
              <button
                type="button"
                onClick={() => setShowAdvancedParams(!showAdvancedParams)}
                className="flex items-center space-x-1.5 text-[11px] text-slate-400 hover:text-slate-200 transition-colors"
              >
                <Sliders className="w-3 h-3" />
                <span>Hybrid Balance &amp; Retrieval Parameters</span>
                {showAdvancedParams ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>

              {showAdvancedParams && (
                <div className="p-3 bg-slate-950 rounded-lg border border-slate-800/80 space-y-3 mt-3 text-xs">
                  <div>
                    <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                      <span>Hybrid Balance (alpha = {alpha.toFixed(2)})</span>
                      <span>{alpha === 1.0 ? 'Pure BM25' : alpha === 0.0 ? 'Pure Dense' : 'Convex Mix'}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.05"
                      value={alpha}
                      onChange={(e) => setAlpha(parseFloat(e.target.value))}
                      className="w-full accent-indigo-500 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-600">
                      <span>Dense Cosine (0.0)</span>
                      <span>BM25 Okapi (1.0)</span>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                      <span>Top-K Evidence Chunks: {topK}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      step="1"
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                      className="w-full accent-indigo-500 cursor-pointer"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Submit Query Button */}
            <button
              onClick={handleAskQuestion}
              disabled={!queryText.trim() || querying}
              className="w-full py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md shadow-indigo-600/20 flex items-center justify-center space-x-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {querying ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Retrieving Chunks &amp; Generating Answer...</span>
                </>
              ) : (
                <>
                  <Search className="w-3.5 h-3.5" />
                  <span>Ask Grounded Question</span>
                </>
              )}
            </button>
          </div>

          {/* Answer Display Card */}
          {queryResponse && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-sm space-y-5">
              {/* Status Header */}
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    Grounded Response
                  </h3>
                </div>

                <div className="flex items-center space-x-2">
                  {queryResponse.status === 'SUCCESS' && (
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      Grounded Answer
                    </span>
                  )}
                  {queryResponse.status === 'ABSTAIN' && (
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-amber-500/10 text-amber-400 border border-amber-500/20">
                      Abstained
                    </span>
                  )}
                  {queryResponse.status === 'ERROR' && (
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-rose-500/10 text-rose-400 border border-rose-500/20">
                      Error
                    </span>
                  )}
                </div>
              </div>

              {/* Abstention Warning Banner if applicable */}
              {queryResponse.status === 'ABSTAIN' && (
                <div className="bg-amber-950/50 border border-amber-500/30 rounded-lg p-3 text-xs text-amber-300 space-y-1">
                  <div className="flex items-center space-x-1.5 font-semibold">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                    <span>Abstention Enforced</span>
                  </div>
                  <p className="text-[11px] text-amber-300/90 pl-5">
                    {queryResponse.abstention_reason ||
                      'The model or evidence validator abstained because evidence was insufficient or out-of-scope.'}
                  </p>
                </div>
              )}

              {/* Generated Answer Body */}
              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 text-xs text-slate-100 leading-relaxed font-sans whitespace-pre-wrap">
                {queryResponse.answer}
              </div>

              {/* Verified Citations List */}
              {queryResponse.citations && queryResponse.citations.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-slate-300 uppercase tracking-wider text-[11px]">
                      Verified Citations ({queryResponse.citations.length})
                    </span>
                    <span className="text-[10px] text-emerald-400 flex items-center space-x-1">
                      <CheckCircle2 className="w-3 h-3" />
                      <span>Application Provenance Verified</span>
                    </span>
                  </div>

                  <div className="space-y-2.5">
                    {queryResponse.citations.map((cit, i) => (
                      <div
                        key={i}
                        className="p-3 bg-slate-950/70 border border-slate-800 rounded-lg text-xs space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <span className="px-2 py-0.5 rounded font-mono font-bold text-[10px] bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                              {cit.tag}
                            </span>
                            <span className="text-slate-300 font-semibold text-[11px]">
                              {cit.filename}
                            </span>
                          </div>
                          <span className="text-[10px] font-mono text-slate-500">
                            Version: {cit.version_id.substring(0, 8)}
                          </span>
                        </div>

                        {/* Snippet */}
                        <div className="text-[11px] text-slate-400 bg-black/40 p-2 rounded border border-slate-800/60 font-mono">
                          "{cit.snippet}"
                        </div>

                        {/* Coordinates & Fingerprint */}
                        <div className="flex flex-wrap items-center gap-3 text-[10px] text-slate-500 font-mono pt-1">
                          <span>Chunk #{cit.chunk_index}</span>
                          <span>
                            Byte Range: {cit.byte_offset} &ndash; {cit.byte_offset + cit.byte_length} ({cit.byte_length} B)
                          </span>
                          <span className="truncate max-w-[200px]" title={cit.chunk_id}>
                            SHA: {cit.chunk_id.substring(0, 16)}...
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Latency & Audit Metadata */}
              <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-[11px] text-slate-500">
                <div className="flex items-center space-x-3">
                  <span className="flex items-center space-x-1">
                    <Clock className="w-3 h-3 text-slate-400" />
                    <span>Retrieval: {queryResponse.latency_retrieval_ms.toFixed(1)}ms</span>
                  </span>
                  <span>&bull;</span>
                  <span>Gen: {queryResponse.latency_generation_ms.toFixed(1)}ms</span>
                  <span>&bull;</span>
                  <span>Total: {queryResponse.total_latency_ms.toFixed(1)}ms</span>
                </div>

                <div className="flex items-center space-x-2 font-mono text-[10px]">
                  <span>Provider: {queryResponse.provider_name}</span>
                  <span>({queryResponse.model_name})</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
