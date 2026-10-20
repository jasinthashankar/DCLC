/**
 * Exact TypeScript representations matching DCLC backend Pydantic schemas (api/schemas.py).
 */

export interface HealthResponse {
  status: string;
  checkpoint_available: boolean;
  checkpoint_path: string;
  device: string;
  active_features: string[];
}

export interface ChunkDetail {
  chunk_index: number;
  byte_offset: number;
  byte_length: number;
  sha256: string;
}

export interface ChunkPreviewResponse {
  filename: string;
  total_bytes: number;
  chunk_count: number;
  min_chunk_bytes: number;
  avg_chunk_bytes: number;
  max_chunk_bytes: number;
  chunks: ChunkDetail[];
}

export interface ChunkPairComparison {
  target_chunk_index: number;
  target_sha256: string;
  target_size_bytes: number;
  is_exact_duplicate: boolean;
  candidate_strategy_used: string;
  selected_reference_chunk_index: number | null;
  selected_reference_sha256: string | null;
  neural_cosine_similarity: number | null;
  similarity_threshold: number;
  resemblance_detected: boolean;
  delta_accepted: boolean;
  delta_size_bytes: number | null;
  standalone_compressed_size_bytes: number;
  reconstruction_verified: boolean;
  rejection_reason: string | null;
  // Phase C extensions
  delta_mode?: string;
  mrd_group_index?: number | null;
  selected_reference_chunk_indices?: number[] | null;
  selected_reference_sha256s?: string[] | null;
}

export interface RevisionCompareResponse {
  base_filename: string;
  target_filename: string;
  base_file_size_bytes: number;
  target_file_size_bytes: number;
  base_chunk_count: number;
  target_chunk_count: number;
  candidate_strategy: string;
  similarity_threshold: number;
  exact_duplicates_count: number;
  resemblance_pairs_count: number;
  deltas_accepted_count: number;
  deltas_rejected_count: number;
  reconstruction_success_rate: number;
  target_raw_bytes: number;
  target_standalone_compressed_bytes: number;
  target_dclc_stored_bytes: number;
  target_storage_savings_pct: number;
  target_compression_ratio: number;
  detailed_chunk_comparisons: ChunkPairComparison[];
  // Phase C extensions
  delta_mode?: string;
  mrd_window_size?: number | null;
  mrd_max_refs?: number | null;
  mrd_groups_count?: number;
  mrd_groups_accepted_count?: number;
  mrd_groups_rejected_count?: number;
  indexer_stats?: {
    cache_hits: number;
    cache_misses: number;
    ann_queries: number;
    total_retrieval_time_s: number;
  } | null;
}

export interface IndexerBenchmarkResult {
  stored_bytes: number;
  savings_bytes: number;
  savings_pct: number;
  compression_ratio: number;
  candidate_recall: number;
  recalled_pairs_count: number;
  total_resem_targets: number;
  avg_candidate_pool: number;
  max_candidate_pool: number;
  comparison_reduction_rate: number;
  exact_duplicates_count: number;
  deltas_accepted: number;
  deltas_rejected: number;
  standalone_stored: number;
  reconstruction_verified_count: number;
  reconstruction_success_rate: number;
  timing_benchmarks?: {
    total_pipeline_time_seconds: number;
    candidate_query_time_seconds: number;
    scoring_time_seconds: number;
    delta_time_seconds: number;
    query_latency_ms_per_chunk: number;
    scoring_latency_ms_per_chunk: number;
    delta_latency_ms_per_chunk: number;
  };
}

export interface MRDStorageBenchmark {
  parameters: {
    window_size: number;
    max_refs: number;
    resemblance_threshold: number;
  };
  single_reference_sr: {
    stored_bytes: number;
    compression_ratio: number;
    deltas_accepted: number;
    deltas_rejected: number;
    encoding_time_seconds: number;
    restoration_throughput_mb_s: number;
  };
  multi_reference_mrd: {
    stored_bytes: number;
    compression_ratio: number;
    mrd_groups_accepted: number;
    mrd_groups_rejected: number;
    encoding_time_seconds: number;
    mg_group_restore_throughput_mb_s: number;
    ms_single_restore_throughput_mb_s: number;
  };
  restoration_throughput_comparison: {
    mg_group_restore_vs_sr_pct: number;
    ms_single_restore_vs_sr_pct: number;
  };
}

export interface BenchmarkStorageResponse {
  split_evaluated: string;
  total_chunks_evaluated: number;
  total_logical_raw_bytes: number;
  exact_duplicate_targets: number;
  resemblance_targets: number;
  system_0_raw: {
    stored_bytes: number;
    savings_pct: number;
    compression_ratio: number;
  };
  system_1_standalone_compression: {
    stored_bytes: number;
    savings_bytes: number;
    savings_pct: number;
    compression_ratio: number;
  };
  system_2_exact_dedup: {
    stored_bytes: number;
    savings_bytes: number;
    savings_pct: number;
    compression_ratio: number;
    exact_duplicates_eliminated: number;
  };
  system_3_full_dclc: Record<string, IndexerBenchmarkResult>;
  // Phase C extension
  system_4_dclc_mrd?: MRDStorageBenchmark | null;
}

export interface ApiError {
  error: string;
  detail: string;
}

// ---------------------------------------------------------------------------
// DCLC-RAG Types (Phase 6E)
// ---------------------------------------------------------------------------

export interface RAGVersionSummary {
  version_id: string;
  doc_id: string;
  filename: string;
  extension: string;
  content_sha256: string;
  chunk_count: number;
  total_bytes: number;
}

export interface RAGDocumentSummary {
  doc_id: string;
  filename: string;
  extension: string;
  version_count: number;
  versions: RAGVersionSummary[];
}

export interface RAGDocumentsResponse {
  documents: RAGDocumentSummary[];
  total_documents: number;
  total_versions: number;
  total_unique_chunks: number;
  shared_chunk_count: number;
}

export interface RAGIngestResponse {
  doc_id: string;
  version_id: string;
  filename: string;
  extension: string;
  content_sha256: string;
  chunk_count: number;
  total_bytes: number;
  shared_chunks_in_version: number;
  is_new_document: boolean;
  is_duplicate_version: boolean;
  message: string;
}

export interface RAGVerifiedCitation {
  tag: string;
  chunk_id: string;
  doc_id: string;
  version_id: string;
  filename: string;
  chunk_index: number;
  byte_offset: number;
  byte_length: number;
  snippet: string;
  is_valid: boolean;
}

export interface RAGQueryRequest {
  query: string;
  doc_id?: string | null;
  version_id?: string | null;
  top_k?: number;
  alpha?: number;
  temperature?: number;
}

export interface RAGQueryResponse {
  status: 'SUCCESS' | 'ABSTAIN' | 'ERROR';
  query: string;
  answer: string;
  target_doc_id: string | null;
  target_version_id: string | null;
  citations: RAGVerifiedCitation[];
  retrieved_chunk_count: number;
  used_chunk_count: number;
  provider_name: string;
  model_name: string;
  latency_retrieval_ms: number;
  latency_generation_ms: number;
  total_latency_ms: number;
  abstention_reason: string | null;
  invalid_citations: string[];
}

export interface RAGStatusResponse {
  status: string;
  provider_name: string;
  model_name: string;
  total_documents: number;
  total_versions: number;
  total_unique_chunks: number;
  shared_chunk_count: number;
  dense_embeddings_count: number;
  dense_dimension: number;
  dedup_savings_pct: number;
  embedding_precision?: string;
  storage_bytes?: number;
  compression_ratio_vs_fp32?: number;
}

export interface StoredRevisionSummary {
  revision_id: string;
  target_name: string;
  base_name: string;
  timestamp: string;
  raw_bytes: number;
  stored_bytes: number;
  storage_savings_pct: number;
  compression_ratio: number;
  sha256: string;
  delta_mode: string;
  chunk_count: number;
}

export interface StoredRevisionDetail {
  revision_id: string;
  target_name: string;
  base_name: string;
  timestamp: string;
  raw_bytes: number;
  stored_bytes: number;
  storage_savings_pct: number;
  compression_ratio: number;
  sha256: string;
  base_sha256: string;
  candidate_strategy: string;
  delta_mode: string;
  chunk_count: number;
  exact_duplicates_count: number;
  delta_chunks_count: number;
  standalone_chunks_count: number;
  chunks: Array<{
    chunk_index: number;
    type: string;
    chunk_sha256: string;
    chunk_size: number;
    stored_size: number;
    base_chunk_sha256?: string;
    compressed_chunk_sha256?: string;
    delta_container_sha256?: string;
    subchunk_index?: number;
    ref_chunk_sha256_list?: string[];
  }>;
}

export interface RestoreRevisionResponse {
  revision_id: string;
  target_name: string;
  sha256: string;
  raw_bytes: number;
  chunk_count: number;
  verified: boolean;
  reconstructed_bytes_match: boolean;
  message: string;
}

