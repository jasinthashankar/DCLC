import { describe, it, expect } from 'vitest';
import { api } from '../services/api';

describe('Frontend API Client Live Integration', () => {
  it('fetches health status from backend', async () => {
    const health = await api.getHealth();
    expect(health).toBeDefined();
    expect(health.status).toBe('healthy');
    expect(health.checkpoint_available).toBe(true);
    expect(health.active_features.length).toBeGreaterThan(0);
  });

  it('runs chunk preview on a synthetic text file', async () => {
    const file = new File([new Uint8Array(20000)], 'test_sample.txt', { type: 'text/plain' });
    const preview = await api.previewChunks(file, 2048, 4096, 8192);

    expect(preview.filename).toBe('test_sample.txt');
    expect(preview.total_bytes).toBe(20000);
    expect(preview.chunk_count).toBeGreaterThan(0);
    expect(preview.chunks.length).toBe(preview.chunk_count);
    expect(preview.chunks[0].sha256).toHaveLength(64);
  });

  it('runs revision comparison on base and target synthetic files', async () => {
    const content = 'DCLC Cloud Deduplication Framework test content sequence.\n'.repeat(300);
    const baseFile = new File([content], 'base.txt', { type: 'text/plain' });
    const targetFile = new File([content], 'target.txt', { type: 'text/plain' });

    const result = await api.compareRevisions(baseFile, targetFile, 'stream_locality', 0.70, 2048, 4096, 8192);

    expect(result.base_filename).toBe('base.txt');
    expect(result.target_filename).toBe('target.txt');
    expect(result.exact_duplicates_count).toBe(result.target_chunk_count);
    expect(result.reconstruction_success_rate).toBe(100.0);
    expect(result.target_storage_savings_pct).toBeGreaterThan(99.0);
  });

  it('fetches multi-version storage benchmark results', async () => {
    const benchmark = await api.getStorageBenchmark(0.70);

    expect(benchmark.split_evaluated).toBe('test');
    expect(benchmark.total_chunks_evaluated).toBe(464);
    expect(benchmark.system_0_raw.stored_bytes).toBeGreaterThan(0);
    expect(benchmark.system_2_exact_dedup.stored_bytes).toBeLessThan(benchmark.system_0_raw.stored_bytes);

    const primaryStrategy = benchmark.system_3_full_dclc['Stream_Locality_W32'];
    expect(primaryStrategy).toBeDefined();
    expect(primaryStrategy.stored_bytes).toBeLessThan(benchmark.system_2_exact_dedup.stored_bytes);
    expect(primaryStrategy.reconstruction_success_rate).toBe(100.0);
  }, 30000);
});
