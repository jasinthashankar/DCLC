import { describe, it, expect } from 'vitest';
import { formatBytes, formatPct, formatRatio, truncateHash } from '../utils/formatters';

describe('Formatters Utility', () => {
  it('formats byte numbers into human-readable strings', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1024)).toBe('1 KB');
    expect(formatBytes(1048576)).toBe('1 MB');
    expect(formatBytes(3792942)).toBe('3.62 MB');
  });

  it('formats percentages correctly', () => {
    expect(formatPct(54.49123)).toBe('54.49%');
    expect(formatPct(0)).toBe('0.00%');
    expect(formatPct(null)).toBe('N/A');
    expect(formatPct(undefined)).toBe('N/A');
  });

  it('formats compression ratios', () => {
    expect(formatRatio(2.198)).toBe('2.20:1');
    expect(formatRatio(1.0)).toBe('1.00:1');
    expect(formatRatio(0)).toBe('N/A');
  });

  it('truncates hash strings properly', () => {
    const hash = '4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a';
    expect(truncateHash(hash, 4, 4)).toBe('4b22...bf8a');
    expect(truncateHash('short')).toBe('short');
  });
});
