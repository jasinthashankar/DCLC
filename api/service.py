"""Service Layer integrating the DCLC Research Pipeline into FastAPI (Phase 5A).

Reuses existing core research modules without duplicating or simplifying algorithms:
- src.fastcdc_chunker: Content-defined variable-sized chunking.
- src.initial_feature_extractor: Algorithm 1 multi-scale shingle extraction.
- src.context_builder: Context sequence window assembling.
- src.model: Context-aware Transformer inference.
- src.candidate_indexer: Label-free candidate indexers (RP-LSH, k-NN, Stream Locality).
- src.delta_encoder: Byte-level preset dictionary delta compression and verification.
- src.pipeline_evaluator: End-to-end comparative storage benchmark.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from api.schemas import (
    BenchmarkStorageResponse,
    ChunkDetail,
    ChunkPairComparison,
    ChunkPreviewResponse,
    HealthResponse,
    RevisionCompareResponse,
)
from src.candidate_indexer import (
    BaseCandidateIndexer,
    EmbeddingCosineKNNIndexer,
    HierarchicalCacheANNIndexer,
    RandomProjectionLSHIndexer,
    StreamLocalityIndexer,
)
from src.context_builder import build_all_context_sequences
from src.delta_encoder import (
    compress_standalone,
    decode_delta,
    decode_mrd_group,
    decode_mrd_single,
    evaluate_and_encode_delta,
    evaluate_and_encode_mrd,
)
from src.evaluate import load_model_and_checkpoint
from src.fastcdc_chunker import fastcdc_chunks
from src.initial_feature_extractor import extract_initial_feature
from src.model import DCLCTransformerContextEncoder
from src.pipeline_evaluator import run_pipeline_evaluation
from src.storage_manager import (
    MissingReferenceError,
    RevisionNotFoundError,
    StorageCorruptionError,
    StorageManager,
)

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path("checkpoints/dclc_encoder_version_aware.pt")
DEFAULT_STORAGE_DIR = Path("data/dclc_store")


class DCLCBackendService:
    """Manages model lifecycle and research pipeline execution for REST API."""

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
        storage_dir: Path | str = DEFAULT_STORAGE_DIR,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._model: Optional[DCLCTransformerContextEncoder] = None
        self.storage_manager = StorageManager(base_dir=storage_dir)

    def get_model(self) -> DCLCTransformerContextEncoder:
        """Lazy load or return cached Transformer encoder in eval mode."""
        if self._model is not None:
            return self._model

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.checkpoint_path}. "
                "Ensure version-aware training was completed."
            )

        logger.info(f"Loading version-aware model from {self.checkpoint_path}...")
        self._model = load_model_and_checkpoint(self.checkpoint_path, device=self.device)
        self._model.eval()
        return self._model

    def check_health(self) -> HealthResponse:
        """Inspect service health and checkpoint availability."""
        ckpt_exists = self.checkpoint_path.exists()
        active_features = [
            "FastCDC Variable Chunking",
            "Algorithm 1 Multi-Scale Shingle Extraction",
            "Radius-4 Context Window Assembler",
            "Verified Preset-Dictionary Delta Compression",
            "Label-Free Candidate Indexing (RP-LSH, k-NN, Stream Locality)",
            "Hierarchical Cache + BallTree Nearest-Neighbor Indexing (Section 3.3)",
            "Multi-Reference Delta Encoding (MRD, Section 3.4)",
        ]
        if ckpt_exists:
            active_features.append("Trained Context-Aware Transformer Inference")

        return HealthResponse(
            status="healthy" if ckpt_exists else "degraded_missing_checkpoint",
            checkpoint_available=ckpt_exists,
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
            active_features=active_features,
        )

    def preview_chunks(
        self,
        file_bytes: bytes,
        filename: str = "uploaded_file.bin",
        min_size: int = 4096,
        avg_size: int = 8192,
        max_size: int = 16384,
    ) -> ChunkPreviewResponse:
        """Chunk an uploaded file using FastCDC and compute structural metadata."""
        if len(file_bytes) == 0:
            return ChunkPreviewResponse(
                filename=filename,
                total_bytes=0,
                chunk_count=0,
                min_chunk_bytes=0,
                avg_chunk_bytes=0.0,
                max_chunk_bytes=0,
                chunks=[],
            )

        raw_chunks = list(fastcdc_chunks(file_bytes, min_size=min_size, avg_size=avg_size, max_size=max_size))
        chunk_details: list[ChunkDetail] = []
        current_offset = 0

        for idx, chunk_b in enumerate(raw_chunks):
            c_len = len(chunk_b)
            fp = hashlib.sha256(chunk_b).hexdigest()
            chunk_details.append(
                ChunkDetail(
                    chunk_index=idx,
                    byte_offset=current_offset,
                    byte_length=c_len,
                    sha256=fp,
                )
            )
            current_offset += c_len

        sizes = [c.byte_length for c in chunk_details]

        return ChunkPreviewResponse(
            filename=filename,
            total_bytes=len(file_bytes),
            chunk_count=len(chunk_details),
            min_chunk_bytes=min(sizes) if sizes else 0,
            avg_chunk_bytes=float(sum(sizes) / len(sizes)) if sizes else 0.0,
            max_chunk_bytes=max(sizes) if sizes else 0,
            chunks=chunk_details,
        )

    def compare_revisions(
        self,
        base_bytes: bytes,
        target_bytes: bytes,
        base_filename: str = "base.txt",
        target_filename: str = "target.txt",
        candidate_strategy: str = "stream_locality",
        resemblance_threshold: float = 0.70,
        min_chunk_size: int = 4096,
        avg_chunk_size: int = 8192,
        max_chunk_size: int = 16384,
        delta_mode: str = "single",
        mrd_window_size: int = 3,
        mrd_max_refs: int = 3,
    ) -> RevisionCompareResponse:
        """Run full DCLC redundancy elimination pipeline between base and target revisions."""
        d_mode = "multi" if delta_mode.lower() in ("multi", "mrd") else "single"

        # 1. Chunk both files
        base_raw_chunks = list(fastcdc_chunks(base_bytes, min_size=min_chunk_size, avg_size=avg_chunk_size, max_size=max_chunk_size))
        target_raw_chunks = list(fastcdc_chunks(target_bytes, min_size=min_chunk_size, avg_size=avg_chunk_size, max_size=max_chunk_size))

        n_base = len(base_raw_chunks)
        n_target = len(target_raw_chunks)

        if n_target == 0:
            return RevisionCompareResponse(
                base_filename=base_filename,
                target_filename=target_filename,
                base_file_size_bytes=len(base_bytes),
                target_file_size_bytes=0,
                base_chunk_count=n_base,
                target_chunk_count=0,
                candidate_strategy=candidate_strategy,
                similarity_threshold=resemblance_threshold,
                exact_duplicates_count=0,
                resemblance_pairs_count=0,
                deltas_accepted_count=0,
                deltas_rejected_count=0,
                reconstruction_success_rate=100.0,
                target_raw_bytes=0,
                target_standalone_compressed_bytes=0,
                target_dclc_stored_bytes=0,
                target_storage_savings_pct=0.0,
                target_compression_ratio=1.0,
                detailed_chunk_comparisons=[],
                delta_mode=d_mode,
                mrd_window_size=mrd_window_size if d_mode == "multi" else None,
                mrd_max_refs=mrd_max_refs if d_mode == "multi" else None,
            )

        # 2. Extract Algorithm 1 initial features and context sequences
        base_feat_list = [extract_initial_feature(c) for c in base_raw_chunks]
        target_feat_list = [extract_initial_feature(c) for c in target_raw_chunks]

        base_seqs = build_all_context_sequences(base_feat_list, radius=4) if n_base > 0 else np.empty((0, 9, 128), dtype=np.float32)
        target_seqs = build_all_context_sequences(target_feat_list, radius=4)

        base_feats = np.stack(base_feat_list) if n_base > 0 else np.empty((0, 128), dtype=np.float32)
        target_feats = np.stack(target_feat_list)

        # 3. Model inference for embeddings
        model = self.get_model()
        with torch.no_grad():
            base_embs = model(torch.from_numpy(base_seqs).float().to(self.device)).cpu().numpy() if n_base > 0 else np.empty((0, 128), dtype=np.float32)
            target_embs = model(torch.from_numpy(target_seqs).float().to(self.device)).cpu().numpy()

        # 4. Initialize candidate indexer
        strat = candidate_strategy.lower()
        if strat in ("rp_lsh", "lsh"):
            indexer: BaseCandidateIndexer = RandomProjectionLSHIndexer(dim=128, num_tables=8, num_bits=6, seed=42)
        elif strat in ("embedding_knn", "knn"):
            indexer = EmbeddingCosineKNNIndexer(top_k=20)
        elif strat in ("hierarchical_cache", "hierarchical", "hierarchical_ann"):
            # Locality-aware hierarchical retrieval: bounded FIFO cache + BallTree tree-based nearest neighbor search (Section 3.3)
            indexer = HierarchicalCacheANNIndexer(
                cache_capacity=64,
                similarity_threshold=resemblance_threshold,
                top_k=20,
            )
        else:
            indexer = StreamLocalityIndexer(window_size=32)

        # Populate base chunks into indexer and exact hash table
        base_exact_hash: dict[str, int] = {}
        for b_idx in range(n_base):
            b_chunk = base_raw_chunks[b_idx]
            fp = hashlib.sha256(b_chunk).hexdigest()
            base_exact_hash[fp] = b_idx
            indexer.add_chunk(
                chunk_id=b_idx,
                initial_feature=base_feats[b_idx],
                embedding=base_embs[b_idx],
                sha256=fp,
            )

        # 5. Process target chunks
        target_raw_total = sum(len(c) for c in target_raw_chunks)
        target_standalone_compressed = sum(len(compress_standalone(c)) for c in target_raw_chunks)
        target_dclc_stored = 0

        exact_dups = 0
        resemblance_hits = 0
        deltas_accepted = 0
        deltas_rejected = 0
        reconstruction_verified_total = 0
        chunk_comparisons: list[ChunkPairComparison] = []

        mrd_groups_count = 0
        mrd_groups_accepted = 0
        mrd_groups_rejected = 0

        if d_mode == "single":
            # --- Single-Reference (SR) Delta Mode ---
            for t_idx in range(n_target):
                t_chunk = target_raw_chunks[t_idx]
                t_len = len(t_chunk)
                t_fp = hashlib.sha256(t_chunk).hexdigest()
                t_standalone_size = len(compress_standalone(t_chunk))
                t_feat = target_feats[t_idx]
                t_emb = target_embs[t_idx]

                # Step 5A: Exact SHA-256 duplicate check
                if t_fp in base_exact_hash:
                    b_ref_idx = base_exact_hash[t_fp]
                    exact_dups += 1
                    target_dclc_stored += 32  # 32-byte pointer overhead
                    chunk_comparisons.append(
                        ChunkPairComparison(
                            target_chunk_index=t_idx,
                            target_sha256=t_fp,
                            target_size_bytes=t_len,
                            is_exact_duplicate=True,
                            candidate_strategy_used="exact_sha256_hash_table",
                            selected_reference_chunk_index=b_ref_idx,
                            selected_reference_sha256=t_fp,
                            neural_cosine_similarity=1.0,
                            similarity_threshold=resemblance_threshold,
                            resemblance_detected=False,
                            delta_accepted=False,
                            delta_size_bytes=0,
                            standalone_compressed_size_bytes=t_standalone_size,
                            reconstruction_verified=True,
                            rejection_reason=None,
                            delta_mode="single",
                        )
                    )
                    continue

                # Step 5B: Non-identical chunk -> Query candidate indexer
                cand_ids = indexer.query_candidates(
                    query_id=-1,  # Not in base index
                    initial_feature=t_feat,
                    embedding=t_emb,
                    sha256=t_fp,
                )

                # Step 5C: Neural cosine similarity scoring among candidates
                best_ref_idx: Optional[int] = None
                best_sim = -1.0

                for c_id in cand_ids:
                    if 0 <= c_id < n_base:
                        sim = float(t_emb @ base_embs[c_id])
                        if sim >= resemblance_threshold and sim > best_sim:
                            best_sim = sim
                            best_ref_idx = c_id

                delta_applied = False
                resemblance_found = (best_ref_idx is not None)

                if resemblance_found:
                    resemblance_hits += 1
                    ref_bytes = base_raw_chunks[best_ref_idx]
                    delta_res = evaluate_and_encode_delta(ref_bytes, t_chunk, compression_level=6)

                    if delta_res.is_accepted:
                        # Assert byte-for-byte correctness
                        rec = decode_delta(ref_bytes, delta_res.delta_container)
                        assert rec == t_chunk, "Fatal: Bit-exact verification failed on delta!"
                        reconstruction_verified_total += 1
                        deltas_accepted += 1
                        target_dclc_stored += delta_res.net_stored_size
                        delta_applied = True

                        chunk_comparisons.append(
                            ChunkPairComparison(
                                target_chunk_index=t_idx,
                                target_sha256=t_fp,
                                target_size_bytes=t_len,
                                is_exact_duplicate=False,
                                candidate_strategy_used=strat,
                                selected_reference_chunk_index=best_ref_idx,
                                selected_reference_sha256=hashlib.sha256(ref_bytes).hexdigest(),
                                neural_cosine_similarity=best_sim,
                                similarity_threshold=resemblance_threshold,
                                resemblance_detected=True,
                                delta_accepted=True,
                                delta_size_bytes=delta_res.delta_total_size,
                                standalone_compressed_size_bytes=t_standalone_size,
                                reconstruction_verified=True,
                                rejection_reason=None,
                                delta_mode="single",
                            )
                        )
                    else:
                        deltas_rejected += 1
                        chunk_comparisons.append(
                            ChunkPairComparison(
                                target_chunk_index=t_idx,
                                target_sha256=t_fp,
                                target_size_bytes=t_len,
                                is_exact_duplicate=False,
                                candidate_strategy_used=strat,
                                selected_reference_chunk_index=best_ref_idx,
                                selected_reference_sha256=hashlib.sha256(ref_bytes).hexdigest(),
                                neural_cosine_similarity=best_sim,
                                similarity_threshold=resemblance_threshold,
                                resemblance_detected=True,
                                delta_accepted=False,
                                delta_size_bytes=delta_res.delta_total_size,
                                standalone_compressed_size_bytes=t_standalone_size,
                                reconstruction_verified=True,
                                rejection_reason=delta_res.rejection_reason,
                                delta_mode="single",
                            )
                        )

                if not delta_applied:
                    # Store standalone compressed
                    target_dclc_stored += t_standalone_size
                    if not resemblance_found:
                        chunk_comparisons.append(
                            ChunkPairComparison(
                                target_chunk_index=t_idx,
                                target_sha256=t_fp,
                                target_size_bytes=t_len,
                                is_exact_duplicate=False,
                                candidate_strategy_used=strat,
                                selected_reference_chunk_index=None,
                                selected_reference_sha256=None,
                                neural_cosine_similarity=None,
                                similarity_threshold=resemblance_threshold,
                                resemblance_detected=False,
                                delta_accepted=False,
                                delta_size_bytes=None,
                                standalone_compressed_size_bytes=t_standalone_size,
                                reconstruction_verified=True,
                                rejection_reason="No candidate exceeded similarity threshold.",
                                delta_mode="single",
                            )
                        )
        else:
            # --- Multi-Reference Delta (MRD) Mode ---
            t_idx = 0
            while t_idx < n_target:
                t_chunk = target_raw_chunks[t_idx]
                t_len = len(t_chunk)
                t_fp = hashlib.sha256(t_chunk).hexdigest()
                t_standalone_size = len(compress_standalone(t_chunk))

                # Step 5A: Exact SHA-256 duplicate check
                if t_fp in base_exact_hash:
                    b_ref_idx = base_exact_hash[t_fp]
                    exact_dups += 1
                    target_dclc_stored += 32
                    chunk_comparisons.append(
                        ChunkPairComparison(
                            target_chunk_index=t_idx,
                            target_sha256=t_fp,
                            target_size_bytes=t_len,
                            is_exact_duplicate=True,
                            candidate_strategy_used="exact_sha256_hash_table",
                            selected_reference_chunk_index=b_ref_idx,
                            selected_reference_sha256=t_fp,
                            neural_cosine_similarity=1.0,
                            similarity_threshold=resemblance_threshold,
                            resemblance_detected=False,
                            delta_accepted=False,
                            delta_size_bytes=0,
                            standalone_compressed_size_bytes=t_standalone_size,
                            reconstruction_verified=True,
                            rejection_reason=None,
                            delta_mode="multi",
                        )
                    )
                    t_idx += 1
                    continue

                # Step 5B: Form consecutive non-duplicate target chunk window
                w_indices = []
                curr = t_idx
                while curr < n_target and len(w_indices) < mrd_window_size:
                    curr_fp = hashlib.sha256(target_raw_chunks[curr]).hexdigest()
                    if curr_fp in base_exact_hash:
                        break  # Stop window at exact duplicate boundary
                    w_indices.append(curr)
                    curr += 1

                if not w_indices:
                    w_indices = [t_idx]
                    curr = t_idx + 1

                # Step 5C: Query candidates for all chunks in the window
                ref_cand_set: list[int] = []
                for j in w_indices:
                    cands = indexer.query_candidates(
                        query_id=-1,
                        initial_feature=target_feats[j],
                        embedding=target_embs[j],
                        sha256=hashlib.sha256(target_raw_chunks[j]).hexdigest(),
                    )
                    for c_id in cands:
                        if 0 <= c_id < n_base:
                            sim = float(target_embs[j] @ base_embs[c_id])
                            if sim >= resemblance_threshold:
                                if c_id not in ref_cand_set:
                                    ref_cand_set.append(c_id)
                                if len(ref_cand_set) >= mrd_max_refs:
                                    break
                    if len(ref_cand_set) >= mrd_max_refs:
                        break

                mrd_groups_count += 1
                target_window_chunks = [target_raw_chunks[j] for j in w_indices]

                if ref_cand_set:
                    ref_chunks = [base_raw_chunks[c] for c in ref_cand_set]
                    ref_fps = [hashlib.sha256(rc).hexdigest() for rc in ref_chunks]
                    mrd_res = evaluate_and_encode_mrd(ref_chunks, target_window_chunks, compression_level=6)

                    if mrd_res.is_accepted and mrd_res.mrd_container:
                        # 1. Byte-for-byte exact group reconstruction assertion
                        rec_group = decode_mrd_group(ref_chunks, mrd_res.mrd_container)
                        assert rec_group == target_window_chunks, "Fatal: Bit-exact verification failed on MRD group!"

                        # 2. Byte-for-byte single-target restoration assertion
                        for sub_idx in range(len(target_window_chunks)):
                            rec_s = decode_mrd_single(ref_chunks, mrd_res.mrd_container, sub_idx)
                            assert rec_s == target_window_chunks[sub_idx], f"Fatal: Single restore failed on MRD chunk {sub_idx}!"

                        reconstruction_verified_total += len(target_window_chunks)
                        deltas_accepted += len(target_window_chunks)
                        resemblance_hits += len(target_window_chunks)
                        mrd_groups_accepted += 1
                        target_dclc_stored += mrd_res.net_stored_size

                        chunk_mrd_share = mrd_res.net_stored_size // len(target_window_chunks)
                        for sub_i, j in enumerate(w_indices):
                            sub_chunk = target_raw_chunks[j]
                            sub_fp = hashlib.sha256(sub_chunk).hexdigest()
                            sub_sa = len(compress_standalone(sub_chunk))
                            max_sim = max([float(target_embs[j] @ base_embs[c]) for c in ref_cand_set], default=0.0)
                            chunk_comparisons.append(
                                ChunkPairComparison(
                                    target_chunk_index=j,
                                    target_sha256=sub_fp,
                                    target_size_bytes=len(sub_chunk),
                                    is_exact_duplicate=False,
                                    candidate_strategy_used=strat,
                                    selected_reference_chunk_index=ref_cand_set[0],
                                    selected_reference_sha256=ref_fps[0],
                                    neural_cosine_similarity=max_sim,
                                    similarity_threshold=resemblance_threshold,
                                    resemblance_detected=True,
                                    delta_accepted=True,
                                    delta_size_bytes=chunk_mrd_share,
                                    standalone_compressed_size_bytes=sub_sa,
                                    reconstruction_verified=True,
                                    rejection_reason=None,
                                    delta_mode="multi",
                                    mrd_group_index=mrd_groups_count,
                                    selected_reference_chunk_indices=ref_cand_set,
                                    selected_reference_sha256s=ref_fps,
                                )
                            )
                    else:
                        mrd_groups_rejected += 1
                        deltas_rejected += len(target_window_chunks)
                        rejection_msg = mrd_res.rejection_reason or "MRD compression exceeded standalone size."
                        for j in w_indices:
                            sub_chunk = target_raw_chunks[j]
                            sub_fp = hashlib.sha256(sub_chunk).hexdigest()
                            sub_sa = len(compress_standalone(sub_chunk))
                            target_dclc_stored += sub_sa
                            sims = [float(target_embs[j] @ base_embs[c]) for c in ref_cand_set]
                            max_sim = max(sims) if sims else None
                            chunk_comparisons.append(
                                ChunkPairComparison(
                                    target_chunk_index=j,
                                    target_sha256=sub_fp,
                                    target_size_bytes=len(sub_chunk),
                                    is_exact_duplicate=False,
                                    candidate_strategy_used=strat,
                                    selected_reference_chunk_index=ref_cand_set[0] if ref_cand_set else None,
                                    selected_reference_sha256=ref_fps[0] if ref_fps else None,
                                    neural_cosine_similarity=max_sim,
                                    similarity_threshold=resemblance_threshold,
                                    resemblance_detected=True,
                                    delta_accepted=False,
                                    delta_size_bytes=None,
                                    standalone_compressed_size_bytes=sub_sa,
                                    reconstruction_verified=True,
                                    rejection_reason=rejection_msg,
                                    delta_mode="multi",
                                    mrd_group_index=mrd_groups_count,
                                    selected_reference_chunk_indices=ref_cand_set,
                                    selected_reference_sha256s=ref_fps,
                                )
                            )
                else:
                    mrd_groups_rejected += 1
                    for j in w_indices:
                        sub_chunk = target_raw_chunks[j]
                        sub_fp = hashlib.sha256(sub_chunk).hexdigest()
                        sub_sa = len(compress_standalone(sub_chunk))
                        target_dclc_stored += sub_sa
                        chunk_comparisons.append(
                            ChunkPairComparison(
                                target_chunk_index=j,
                                target_sha256=sub_fp,
                                target_size_bytes=len(sub_chunk),
                                is_exact_duplicate=False,
                                candidate_strategy_used=strat,
                                selected_reference_chunk_index=None,
                                selected_reference_sha256=None,
                                neural_cosine_similarity=None,
                                similarity_threshold=resemblance_threshold,
                                resemblance_detected=False,
                                delta_accepted=False,
                                delta_size_bytes=None,
                                standalone_compressed_size_bytes=sub_sa,
                                reconstruction_verified=True,
                                rejection_reason="No candidate exceeded similarity threshold.",
                                delta_mode="multi",
                                mrd_group_index=mrd_groups_count,
                            )
                        )

                t_idx = curr

        savings_bytes = target_raw_total - target_dclc_stored
        savings_pct = float(savings_bytes / target_raw_total * 100) if target_raw_total > 0 else 0.0
        compression_ratio = float(target_raw_total / target_dclc_stored) if target_dclc_stored > 0 else 0.0
        recon_rate = 100.0 if (deltas_accepted == reconstruction_verified_total) else 0.0

        indexer_stats = None
        if hasattr(indexer, "stats"):
            indexer_stats = {
                "cache_hits": getattr(indexer.stats, "cache_hits", 0),
                "cache_misses": getattr(indexer.stats, "cache_misses", 0),
                "ann_queries": getattr(indexer.stats, "ann_queries", 0),
                "total_retrieval_time_s": round(getattr(indexer.stats, "total_retrieval_time_s", 0.0), 6),
            }

        return RevisionCompareResponse(
            base_filename=base_filename,
            target_filename=target_filename,
            base_file_size_bytes=len(base_bytes),
            target_file_size_bytes=len(target_bytes),
            base_chunk_count=n_base,
            target_chunk_count=n_target,
            candidate_strategy=strat,
            similarity_threshold=resemblance_threshold,
            exact_duplicates_count=exact_dups,
            resemblance_pairs_count=resemblance_hits,
            deltas_accepted_count=deltas_accepted,
            deltas_rejected_count=deltas_rejected,
            reconstruction_success_rate=recon_rate,
            target_raw_bytes=target_raw_total,
            target_standalone_compressed_bytes=target_standalone_compressed,
            target_dclc_stored_bytes=target_dclc_stored,
            target_storage_savings_pct=savings_pct,
            target_compression_ratio=compression_ratio,
            detailed_chunk_comparisons=chunk_comparisons,
            delta_mode=d_mode,
            mrd_window_size=mrd_window_size if d_mode == "multi" else None,
            mrd_max_refs=mrd_max_refs if d_mode == "multi" else None,
            mrd_groups_count=mrd_groups_count if d_mode == "multi" else 0,
            mrd_groups_accepted_count=mrd_groups_accepted if d_mode == "multi" else 0,
            mrd_groups_rejected_count=mrd_groups_rejected if d_mode == "multi" else 0,
            indexer_stats=indexer_stats,
        )

    def run_storage_benchmark(
        self,
        resemblance_threshold: float = 0.70,
        enable_mrd: bool = False,
    ) -> BenchmarkStorageResponse:
        """Execute end-to-end storage evaluation across held-out test families (085-099)."""
        raw_res = run_pipeline_evaluation(
            checkpoint_path=self.checkpoint_path,
            resemblance_threshold=resemblance_threshold,
            device=self.device,
            enable_mrd=enable_mrd,
        )

        return BenchmarkStorageResponse(
            split_evaluated=raw_res["dataset_scope"]["split"],
            total_chunks_evaluated=raw_res["dataset_scope"]["total_chunks"],
            total_logical_raw_bytes=raw_res["dataset_scope"]["total_logical_raw_bytes"],
            exact_duplicate_targets=raw_res["dataset_scope"]["exact_duplicate_targets"],
            resemblance_targets=raw_res["dataset_scope"]["resemblance_targets"],
            system_0_raw=raw_res["system_0_raw"],
            system_1_standalone_compression=raw_res["system_1_standalone_compression"],
            system_2_exact_dedup=raw_res["system_2_exact_dedup"],
            system_3_full_dclc=raw_res["system_3_full_dclc"],
            system_4_dclc_mrd=raw_res.get("system_4_dclc_mrd"),
        )

    def store_revision(
        self,
        base_bytes: bytes,
        target_bytes: bytes,
        base_name: str = "base.txt",
        target_name: str = "target.txt",
        candidate_strategy: str = "hierarchical_cache",
        resemblance_threshold: float = 0.5,
        delta_mode: str = "single",
        mrd_window_size: int = 3,
        mrd_max_refs: int = 3,
        min_chunk_size: int = 2048,
        avg_chunk_size: int = 4096,
        max_chunk_size: int = 8192,
    ) -> dict[str, Any]:
        """Compress and persist revision in durable storage, returning verified recipe."""
        model = self.get_model()
        return self.storage_manager.store_revision(
            base_bytes=base_bytes,
            target_bytes=target_bytes,
            base_name=base_name,
            target_name=target_name,
            candidate_strategy=candidate_strategy,
            resemblance_threshold=resemblance_threshold,
            delta_mode=delta_mode,
            mrd_window_size=mrd_window_size,
            mrd_max_refs=mrd_max_refs,
            min_chunk_size=min_chunk_size,
            avg_chunk_size=avg_chunk_size,
            max_chunk_size=max_chunk_size,
            model=model,
            device=self.device,
        )

    def list_revisions(self) -> list[dict[str, Any]]:
        """List all stored revisions."""
        return self.storage_manager.list_revisions()

    def get_revision(self, revision_id: str) -> dict[str, Any]:
        """Fetch recipe for revision."""
        return self.storage_manager.get_revision(revision_id)

    def restore_revision(self, revision_id: str) -> tuple[bytes, dict[str, Any]]:
        """Restore revision from disk and return (restored_bytes, recipe)."""
        recipe = self.storage_manager.get_revision(revision_id)
        restored_bytes = self.storage_manager.restore_revision(revision_id)
        return restored_bytes, recipe

