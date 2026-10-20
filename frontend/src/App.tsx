import { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { OverviewSection } from './components/OverviewSection';
import { CompareFilesSection } from './components/CompareFilesSection';
import { StorageBenchmarkSection } from './components/StorageBenchmarkSection';
import { ChunkAnalysisSection } from './components/ChunkAnalysisSection';
import { DocumentQASection } from './components/DocumentQASection';
import { RevisionsSection } from './components/RevisionsSection';
import { api } from './services/api';
import type { BenchmarkStorageResponse, HealthResponse } from './types/api';
import { AlertCircle, RefreshCw } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('overview');

  // Backend Health State
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Storage Benchmark State (Cached across Overview and Benchmark tabs)
  const [benchmark, setBenchmark] = useState<BenchmarkStorageResponse | null>(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState<boolean>(false);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);

  // Fetch Health
  const checkHealth = useCallback(async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const data = await api.getHealth();
      setHealth(data);
    } catch (err: any) {
      setHealth(null);
      setHealthError(err?.detail || err?.message || 'Failed to reach backend.');
    } finally {
      setHealthLoading(false);
    }
  }, []);

  // Fetch Benchmark
  const fetchBenchmark = useCallback(async (threshold = 0.70, enableMrd = true) => {
    setBenchmarkLoading(true);
    setBenchmarkError(null);
    try {
      const data = await api.getStorageBenchmark(threshold, enableMrd);
      setBenchmark(data);
    } catch (err: any) {
      setBenchmarkError(err?.detail || err?.message || 'Failed to load storage benchmark.');
    } finally {
      setBenchmarkLoading(false);
    }
  }, []);

  // Initialize on mount
  useEffect(() => {
    checkHealth();
    fetchBenchmark(0.70, true);
  }, [checkHealth, fetchBenchmark]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col md:flex-row font-sans">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        healthLoading={healthLoading}
        healthError={healthError}
        onRefreshHealth={checkHealth}
      />

      {/* Main Body */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Backend Offline Notification Banner */}
        {healthError && (
          <div className="bg-rose-950/80 border-b border-rose-500/30 px-5 py-3">
            <div className="flex items-center justify-between text-xs text-rose-300">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>
                  <strong>Backend Offline:</strong> Could not connect to <code>{api.getBaseUrl()}</code>.
                  Start the FastAPI server:
                  <code className="ml-1.5 px-2 py-0.5 rounded bg-black/40 text-rose-200">
                    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
                  </code>
                </span>
              </div>
              <button
                onClick={checkHealth}
                disabled={healthLoading}
                className="ml-4 px-2.5 py-1 rounded bg-rose-900/60 hover:bg-rose-800 text-rose-200 border border-rose-700/50 flex items-center space-x-1 shrink-0"
              >
                <RefreshCw className={`w-3 h-3 ${healthLoading ? 'animate-spin' : ''}`} />
                <span>Retry</span>
              </button>
            </div>
          </div>
        )}

        {/* Content View */}
        <main className="flex-1 p-6 sm:p-8 lg:p-10 overflow-y-auto">
          {activeTab === 'overview' && (
            <OverviewSection
              health={health}
              benchmark={benchmark}
              benchmarkLoading={benchmarkLoading}
              onNavigateTab={setActiveTab}
            />
          )}

          {activeTab === 'compare' && <CompareFilesSection />}
          {activeTab === 'revisions' && <RevisionsSection />}

          {activeTab === 'benchmark' && (
            <StorageBenchmarkSection
              benchmark={benchmark}
              loading={benchmarkLoading}
              error={benchmarkError}
              onRefresh={fetchBenchmark}
            />
          )}

          {activeTab === 'chunks' && <ChunkAnalysisSection />}
          {activeTab === 'qa' && <DocumentQASection />}
        </main>
      </div>
    </div>
  );
}

export default App;
