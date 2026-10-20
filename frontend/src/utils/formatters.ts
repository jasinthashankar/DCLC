/**
 * Utility formatters for research metrics, bytes, and numbers.
 */

export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 B';
  if (!Number.isFinite(bytes) || bytes < 0) return 'N/A';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const clampedIdx = Math.min(i, sizes.length - 1);

  return `${parseFloat((bytes / Math.pow(k, clampedIdx)).toFixed(dm))} ${sizes[clampedIdx]}`;
}

export function formatPct(value: number | undefined | null, decimals = 2): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return 'N/A';
  return `${value.toFixed(decimals)}%`;
}

export function formatRatio(value: number | undefined | null, decimals = 2): string {
  if (value === undefined || value === null || !Number.isFinite(value) || value <= 0) return 'N/A';
  return `${value.toFixed(decimals)}:1`;
}

export function truncateHash(hash: string, lead = 8, trail = 8): string {
  if (!hash || hash.length <= lead + trail) return hash;
  return `${hash.slice(0, lead)}...${hash.slice(-trail)}`;
}
