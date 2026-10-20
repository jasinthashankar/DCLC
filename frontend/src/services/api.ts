import type {
  BenchmarkStorageResponse,
  ChunkPreviewResponse,
  HealthResponse,
  RestoreRevisionResponse,
  RevisionCompareResponse,
  StoredRevisionDetail,
  StoredRevisionSummary,
} from '../types/api';

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '');

class ApiClientError extends Error {
  status: number;
  detail: string;

  constructor(status: number, message: string, detail?: string) {
    super(message);
    this.name = 'ApiClientError';
    this.status = status;
    this.detail = detail || message;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Server returned status ${res.status} (${res.statusText})`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        detail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // JSON parse error or empty response
    }
    throw new ApiClientError(res.status, detail, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getBaseUrl(): string {
    return API_BASE_URL;
  },

  async getHealth(): Promise<HealthResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      return await handleResponse<HealthResponse>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Cannot connect to DCLC backend at ${API_BASE_URL}. Ensure the backend server is running on port 8000.`
      );
    }
  },

  async previewChunks(
    file: File,
    minChunkSize = 4096,
    avgChunkSize = 8192,
    maxChunkSize = 16384
  ): Promise<ChunkPreviewResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const query = new URLSearchParams({
      min_chunk_size: minChunkSize.toString(),
      avg_chunk_size: avgChunkSize.toString(),
      max_chunk_size: maxChunkSize.toString(),
    });

    try {
      const res = await fetch(`${API_BASE_URL}/chunk-preview?${query.toString()}`, {
        method: 'POST',
        body: formData,
      });
      return await handleResponse<ChunkPreviewResponse>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error uploading '${file.name}' to ${API_BASE_URL}/chunk-preview.`
      );
    }
  },

  async compareRevisions(
    baseFile: File,
    targetFile: File,
    candidateStrategy = 'stream_locality',
    resemblanceThreshold = 0.70,
    minChunkSize = 4096,
    avgChunkSize = 8192,
    maxChunkSize = 16384,
    deltaMode = 'single',
    mrdWindowSize = 3,
    mrdMaxRefs = 3
  ): Promise<RevisionCompareResponse> {
    const formData = new FormData();
    formData.append('base_file', baseFile);
    formData.append('target_file', targetFile);
    formData.append('candidate_strategy', candidateStrategy);
    formData.append('resemblance_threshold', resemblanceThreshold.toString());
    formData.append('min_chunk_size', minChunkSize.toString());
    formData.append('avg_chunk_size', avgChunkSize.toString());
    formData.append('max_chunk_size', maxChunkSize.toString());
    formData.append('delta_mode', deltaMode);
    formData.append('mrd_window_size', mrdWindowSize.toString());
    formData.append('mrd_max_refs', mrdMaxRefs.toString());

    try {
      const res = await fetch(`${API_BASE_URL}/compare-revisions`, {
        method: 'POST',
        body: formData,
      });
      return await handleResponse<RevisionCompareResponse>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error comparing revisions with ${API_BASE_URL}/compare-revisions.`
      );
    }
  },

  async getStorageBenchmark(resemblanceThreshold = 0.70, enableMrd = false): Promise<BenchmarkStorageResponse> {
    const query = new URLSearchParams({
      resemblance_threshold: resemblanceThreshold.toString(),
      enable_mrd: enableMrd.toString(),
    });

    try {
      const res = await fetch(`${API_BASE_URL}/benchmark-storage?${query.toString()}`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      return await handleResponse<BenchmarkStorageResponse>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error fetching benchmark from ${API_BASE_URL}/benchmark-storage.`
      );
    }
  },

  // ---------------------------------------------------------------------------
  // DCLC-RAG Endpoints (Phase 6E)
  // ---------------------------------------------------------------------------

  async getRAGStatus(): Promise<import('../types/api').RAGStatusResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/rag/status`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      return await handleResponse<import('../types/api').RAGStatusResponse>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error fetching RAG status from ${API_BASE_URL}/rag/status.`
      );
    }
  },

  async getRAGDocuments(): Promise<import('../types/api').RAGDocumentsResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/rag/documents`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      return await handleResponse<import('../types/api').RAGDocumentsResponse>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error fetching RAG documents from ${API_BASE_URL}/rag/documents.`
      );
    }
  },

  async ingestRAGDocument(
    file: File,
    docId?: string | null,
    minChunkSize = 64,
    avgChunkSize = 256,
    maxChunkSize = 1024
  ): Promise<import('../types/api').RAGIngestResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (docId) {
      formData.append('doc_id', docId);
    }
    formData.append('min_chunk_size', minChunkSize.toString());
    formData.append('avg_chunk_size', avgChunkSize.toString());
    formData.append('max_chunk_size', maxChunkSize.toString());

    try {
      const res = await fetch(`${API_BASE_URL}/rag/ingest`, {
        method: 'POST',
        body: formData,
      });
      return await handleResponse<import('../types/api').RAGIngestResponse>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error ingesting '${file.name}' at ${API_BASE_URL}/rag/ingest.`
      );
    }
  },

  async queryRAG(
    request: import('../types/api').RAGQueryRequest
  ): Promise<import('../types/api').RAGQueryResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/rag/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(request),
      });
      return await handleResponse<import('../types/api').RAGQueryResponse>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error querying RAG engine at ${API_BASE_URL}/rag/query.`
      );
    }
  },

  async storeRevision(
    baseFile: File,
    targetFile: File,
    candidateStrategy = 'hierarchical_cache',
    deltaMode: 'single' | 'multi' = 'single',
    mrdWindowSize = 3,
    mrdMaxRefs = 3,
    resemblanceThreshold = 0.50,
  ): Promise<StoredRevisionDetail> {
    const formData = new FormData();
    formData.append('base_file', baseFile);
    formData.append('target_file', targetFile);
    formData.append('candidate_strategy', candidateStrategy);
    formData.append('delta_mode', deltaMode);
    formData.append('mrd_window_size', mrdWindowSize.toString());
    formData.append('mrd_max_refs', mrdMaxRefs.toString());
    formData.append('resemblance_threshold', resemblanceThreshold.toString());

    try {
      const res = await fetch(`${API_BASE_URL}/store-revision`, {
        method: 'POST',
        body: formData,
      });
      return await handleResponse<StoredRevisionDetail>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error persisting revision to ${API_BASE_URL}/store-revision.`
      );
    }
  },

  async listRevisions(): Promise<StoredRevisionSummary[]> {
    try {
      const res = await fetch(`${API_BASE_URL}/revisions`);
      return await handleResponse<StoredRevisionSummary[]>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error fetching revisions from ${API_BASE_URL}/revisions.`
      );
    }
  },

  async getRevision(revisionId: string): Promise<StoredRevisionDetail> {
    try {
      const res = await fetch(`${API_BASE_URL}/revisions/${encodeURIComponent(revisionId)}`);
      return await handleResponse<StoredRevisionDetail>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error fetching revision '${revisionId}' from ${API_BASE_URL}/revisions.`
      );
    }
  },

  async restoreRevision(revisionId: string): Promise<RestoreRevisionResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/restore-revision/${encodeURIComponent(revisionId)}`);
      return await handleResponse<RestoreRevisionResponse>(res);
    } catch (err) {
      if (err instanceof ApiClientError) throw err;
      throw new ApiClientError(
        0,
        `Network error restoring revision '${revisionId}' from ${API_BASE_URL}/restore-revision.`
      );
    }
  },

  getDownloadRevisionUrl(revisionId: string): string {
    return `${API_BASE_URL}/download-restored/${encodeURIComponent(revisionId)}`;
  },
};

export const apiService = api;


