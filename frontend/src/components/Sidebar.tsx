import type { FC } from 'react';
import {
  Database,
  ArrowRightLeft,
  BarChart3,
  Scissors,
  Layers,
  BookOpen,
  Archive,
  ExternalLink,
  RefreshCw,
} from 'lucide-react';
import type { HealthResponse } from '../types/api';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  onRefreshHealth: () => void;
}

export const Sidebar: FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  health,
  healthLoading,
  healthError,
  onRefreshHealth,
}) => {
  const isHealthy = health?.status === 'healthy';

  const navItems = [
    {
      id: 'overview',
      label: 'Overview',
      desc: 'System status & key metrics',
      icon: Layers,
    },
    {
      id: 'compare',
      label: 'Compare Files',
      desc: 'Deduplication & delta encoding',
      icon: ArrowRightLeft,
    },
    {
      id: 'revisions',
      label: 'Durable Store',
      desc: 'Persist, restore & download',
      icon: Archive,
    },
    {
      id: 'benchmark',
      label: 'Storage Benchmark',
      desc: 'Test dataset comparative results',
      icon: BarChart3,
    },
    {
      id: 'chunks',
      label: 'Chunk Analysis',
      desc: 'FastCDC boundary inspection',
      icon: Scissors,
    },
    {
      id: 'qa',
      label: 'Document Q&A',
      desc: 'Version-aware grounded RAG',
      icon: BookOpen,
    },
  ];

  return (
    <aside className="w-full md:w-64 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800/80">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-md shadow-indigo-600/30">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-base text-white tracking-tight">DCLC</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-semibold">
                CSE 2026
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5 leading-tight">
              Cloud Redundancy Elimination
            </p>
          </div>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-3 space-y-1.5 overflow-y-auto">
        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">
          Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-start space-x-3 px-3 py-2.5 rounded-lg text-left transition-all ${
                isActive
                  ? 'bg-indigo-600 text-white shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Icon className={`w-4 h-4 mt-0.5 shrink-0 ${isActive ? 'text-white' : 'text-slate-400'}`} />
              <div>
                <div className="text-xs">{item.label}</div>
                <div className={`text-[10px] ${isActive ? 'text-indigo-100' : 'text-slate-500'}`}>
                  {item.desc}
                </div>
              </div>
            </button>
          );
        })}
      </nav>

      {/* Bottom Status & Links */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/40 space-y-3">
        {/* Backend Status Indicator */}
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center space-x-2">
            {healthError ? (
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
            ) : isHealthy ? (
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            ) : (
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
            )}
            <span className="text-slate-300 font-medium text-[11px]">
              {healthError ? 'Backend Offline' : isHealthy ? 'Backend Online' : 'Connecting...'}
            </span>
          </div>

          <button
            onClick={onRefreshHealth}
            disabled={healthLoading}
            title="Refresh backend status"
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${healthLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {health && (
          <div className="text-[10px] text-slate-400 space-y-0.5 bg-slate-900/80 p-2 rounded border border-slate-800/60 font-mono">
            <div>Device: <span className="text-slate-200">{health.device}</span></div>
            <div className="truncate" title={health.checkpoint_path}>
              Model: <span className="text-slate-200">{health.checkpoint_path.split('\\').pop()}</span>
            </div>
          </div>
        )}

        <div className="pt-1 flex items-center justify-between text-[11px] text-slate-400">
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-1 hover:text-indigo-400 transition-colors"
          >
            <span>Swagger API Docs</span>
            <ExternalLink className="w-3 h-3" />
          </a>
          <span className="text-slate-600">v1.0</span>
        </div>
      </div>
    </aside>
  );
};
