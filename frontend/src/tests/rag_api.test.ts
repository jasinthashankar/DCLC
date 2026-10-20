import { describe, it, expect } from 'vitest';
import { api } from '../services/api';

describe('Frontend RAG API Client Live Integration', () => {
  it('fetches RAG system status from backend', async () => {
    const status = await api.getRAGStatus();
    expect(status).toBeDefined();
    expect(['ready', 'empty']).toContain(status.status);
    expect(status.provider_name).toBeDefined();
    expect(status.dedup_savings_pct).toBeGreaterThanOrEqual(0.0);
  });

  it('fetches RAG document registry list', async () => {
    const docs = await api.getRAGDocuments();
    expect(docs).toBeDefined();
    expect(Array.isArray(docs.documents)).toBe(true);
    expect(docs.total_documents).toBeGreaterThanOrEqual(0);
  });

  it('ingests a document and queries with citations', async () => {
    const content = (
      'DCLC Cloud Architecture Report:\n' +
      'Authentication requires mutual TLS certificates.\n' +
      'Worker node count is set to 48 compute instances.\n'
    ).repeat(10);
    const file = new File([content], 'cluster_spec.txt', { type: 'text/plain' });

    const ingestRes = await api.ingestRAGDocument(file, null, 64, 256, 1024);
    expect(ingestRes.filename).toBe('cluster_spec.txt');
    expect(ingestRes.chunk_count).toBeGreaterThan(0);
    expect(ingestRes.doc_id).toBeDefined();

    const queryRes = await api.queryRAG({
      query: 'What is the worker node count?',
      doc_id: ingestRes.doc_id,
      top_k: 3,
      alpha: 0.5,
    });
    expect(queryRes).toBeDefined();
    expect(['SUCCESS', 'ABSTAIN']).toContain(queryRes.status);
    expect(queryRes.retrieved_chunk_count).toBeGreaterThan(0);
  }, 30000);
});
