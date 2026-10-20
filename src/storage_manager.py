"""Durable Storage Lifecycle and Revision Management for DCLC (Phase E).

Provides atomic, content-addressed persistence for chunks, delta containers,
and revision recipes. Enables restart-safe restoration without in-memory state.

Architecture:
- chunks/   : Content-addressed raw base chunks and standalone compressed chunks (SHA-256.bin)
- deltas/   : Content-addressed single-reference and multi-reference delta containers (SHA-256.bin)
- revisions/: Revision recipes with complete chunk reconstruction manifests (<revision_id>.json)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional
import zlib

import numpy as np
import torch

from src.candidate_indexer import (
    BaseCandidateIndexer,
    EmbeddingCosineKNNIndexer,
    HierarchicalCacheANNIndexer,
    RandomProjectionLSHIndexer,
    StreamLocalityIndexer,
)
from src.context_builder import build_all_context_sequences
from src.delta_encoder import (
    DeltaCorruptionError,
    compress_standalone,
    decode_delta,
    decode_mrd_group,
    decode_mrd_single,
    evaluate_and_encode_delta,
    evaluate_and_encode_mrd,
)
from src.fastcdc_chunker import fastcdc_chunks
from src.initial_feature_extractor import extract_initial_feature
from src.model import DCLCTransformerContextEncoder

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = Path("data/dclc_store")
_SAFE_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]+$")


class DCLCStorageError(Exception):
    """Base exception for DCLC storage lifecycle errors."""
    pass


class MissingReferenceError(DCLCStorageError):
    """Raised when a referenced chunk or delta container is missing from disk."""
    pass


class StorageCorruptionError(DCLCStorageError):
    """Raised when stored data fails cryptographic checksum or format validation."""
    pass


class RevisionNotFoundError(DCLCStorageError):
    """Raised when the specified revision ID does not exist in the store."""
    pass


class StorageManager:
    """Manages content-addressed chunk/delta storage and durable revision recipes."""

    def __init__(self, base_dir: Path | str = DEFAULT_STORE_DIR):
        self.base_dir = Path(base_dir)
        self.chunks_dir = self.base_dir / "chunks"
        self.deltas_dir = self.base_dir / "deltas"
        self.revisions_dir = self.base_dir / "revisions"

        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.deltas_dir.mkdir(parents=True, exist_ok=True)
        self.revisions_dir.mkdir(parents=True, exist_ok=True)

    def _validate_hex_hash(self, hex_hash: str) -> str:
        """Sanitize and validate SHA-256 hex string to prevent path traversal."""
        if not isinstance(hex_hash, str) or len(hex_hash) != 64 or not _SAFE_ID_REGEX.match(hex_hash):
            raise ValueError(f"Invalid SHA-256 hash identifier: '{hex_hash}'")
        return hex_hash

    def _validate_revision_id(self, revision_id: str) -> str:
        """Sanitize and validate revision ID string to prevent path traversal."""
        if not isinstance(revision_id, str) or not revision_id or not _SAFE_ID_REGEX.match(revision_id):
            raise ValueError(f"Invalid revision identifier: '{revision_id}'")
        return revision_id

    def _atomic_write_file(self, target_path: Path, data: bytes) -> None:
        """Atomically write binary data to disk via temporary file rename."""
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        try:
            with open(temp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _atomic_write_json(self, target_path: Path, data: dict[str, Any]) -> None:
        """Atomically write structured JSON to disk via temporary file rename."""
        temp_path = target_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def store_chunk(self, data: bytes) -> str:
        """Store chunk content-addressed by SHA-256."""
        sha256 = hashlib.sha256(data).hexdigest()
        target_path = self.chunks_dir / f"{sha256}.bin"
        if not target_path.exists():
            self._atomic_write_file(target_path, data)
        return sha256

    def get_chunk(self, sha256: str) -> bytes:
        """Retrieve chunk by SHA-256, verifying cryptographic integrity."""
        valid_sha = self._validate_hex_hash(sha256)
        target_path = self.chunks_dir / f"{valid_sha}.bin"
        if not target_path.exists():
            raise MissingReferenceError(f"Base chunk reference '{valid_sha}' not found on disk.")
        data = target_path.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != valid_sha:
            raise StorageCorruptionError(
                f"Chunk '{valid_sha}' failed integrity check: actual hash is '{actual_sha}'."
            )
        return data

    def store_delta(self, data: bytes) -> str:
        """Store delta container content-addressed by SHA-256."""
        sha256 = hashlib.sha256(data).hexdigest()
        target_path = self.deltas_dir / f"{sha256}.bin"
        if not target_path.exists():
            self._atomic_write_file(target_path, data)
        return sha256

    def get_delta(self, sha256: str) -> bytes:
        """Retrieve delta container by SHA-256, verifying cryptographic integrity."""
        valid_sha = self._validate_hex_hash(sha256)
        target_path = self.deltas_dir / f"{valid_sha}.bin"
        if not target_path.exists():
            raise MissingReferenceError(f"Delta container reference '{valid_sha}' not found on disk.")
        data = target_path.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != valid_sha:
            raise StorageCorruptionError(
                f"Delta container '{valid_sha}' failed integrity check: actual hash is '{actual_sha}'."
            )
        return data

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
        model: Optional[DCLCTransformerContextEncoder] = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Process, compress, and durably persist a target revision against a base file."""
        if delta_mode not in ("single", "multi"):
            raise ValueError(f"Invalid delta_mode '{delta_mode}'; must be 'single' or 'multi'.")
        if mrd_window_size < 1 or mrd_window_size > 16:
            raise ValueError(f"mrd_window_size must be between 1 and 16, got {mrd_window_size}.")
        if mrd_max_refs < 1 or mrd_max_refs > 8:
            raise ValueError(f"mrd_max_refs must be between 1 and 8, got {mrd_max_refs}.")

        target_sha256 = hashlib.sha256(target_bytes).hexdigest()
        base_sha256 = hashlib.sha256(base_bytes).hexdigest()
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        revision_id = f"rev_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{target_sha256[:8]}"

        # FastCDC chunking
        base_raw_chunks = list(fastcdc_chunks(base_bytes, min_size=min_chunk_size, avg_size=avg_chunk_size, max_size=max_chunk_size)) if base_bytes else []
        target_raw_chunks = list(fastcdc_chunks(target_bytes, min_size=min_chunk_size, avg_size=avg_chunk_size, max_size=max_chunk_size)) if target_bytes else []

        n_base = len(base_raw_chunks)
        n_target = len(target_raw_chunks)

        # Store all base chunks durably so they are reliable reference anchors
        base_exact_hash: dict[str, int] = {}
        for b_idx, b_chunk in enumerate(base_raw_chunks):
            fp = self.store_chunk(b_chunk)
            base_exact_hash[fp] = b_idx

        chunk_manifest: list[dict[str, Any]] = []
        stored_bytes_total = 0
        exact_count = 0
        delta_count = 0
        standalone_count = 0

        if n_target == 0:
            recipe = {
                "revision_id": revision_id,
                "target_name": target_name,
                "base_name": base_name,
                "timestamp": timestamp,
                "raw_bytes": 0,
                "stored_bytes": 0,
                "storage_savings_pct": 0.0,
                "compression_ratio": 1.0,
                "sha256": target_sha256,
                "base_sha256": base_sha256,
                "candidate_strategy": candidate_strategy,
                "delta_mode": delta_mode,
                "chunk_count": 0,
                "exact_duplicates_count": 0,
                "delta_chunks_count": 0,
                "standalone_chunks_count": 0,
                "chunks": [],
            }
            self._atomic_write_json(self.revisions_dir / f"{revision_id}.json", recipe)
            return recipe

        # Feature extraction and context sequences
        base_feat_list = [extract_initial_feature(c) for c in base_raw_chunks]
        target_feat_list = [extract_initial_feature(c) for c in target_raw_chunks]

        base_seqs = build_all_context_sequences(base_feat_list, radius=4) if n_base > 0 else np.empty((0, 9, 128), dtype=np.float32)
        target_seqs = build_all_context_sequences(target_feat_list, radius=4)

        base_feats = np.stack(base_feat_list) if n_base > 0 else np.empty((0, 128), dtype=np.float32)
        target_feats = np.stack(target_feat_list)

        if model is not None:
            model.eval()
            with torch.no_grad():
                base_embs = model(torch.from_numpy(base_seqs).float().to(device)).cpu().numpy() if n_base > 0 else np.empty((0, 128), dtype=np.float32)
                target_embs = model(torch.from_numpy(target_seqs).float().to(device)).cpu().numpy()
        else:
            base_embs = np.array([f[:64] for f in base_feat_list], dtype=np.float32) if n_base > 0 else np.empty((0, 64), dtype=np.float32)
            target_embs = np.array([f[:64] for f in target_feat_list], dtype=np.float32)

        # Build candidate indexer
        strat = candidate_strategy.lower()
        if strat in ("rp_lsh", "lsh"):
            indexer: BaseCandidateIndexer = RandomProjectionLSHIndexer(dim=128, num_tables=8, num_bits=6, seed=42)
        elif strat in ("embedding_knn", "knn"):
            indexer = EmbeddingCosineKNNIndexer(top_k=20)
        elif strat in ("hierarchical_cache", "hierarchical", "hierarchical_ann"):
            indexer = HierarchicalCacheANNIndexer(
                cache_capacity=64,
                similarity_threshold=resemblance_threshold,
                top_k=20,
            )
        else:
            indexer = StreamLocalityIndexer(window_size=32)

        for b_idx in range(n_base):
            b_chunk = base_raw_chunks[b_idx]
            fp = hashlib.sha256(b_chunk).hexdigest()
            indexer.add_chunk(
                chunk_id=b_idx,
                initial_feature=base_feats[b_idx],
                embedding=base_embs[b_idx],
                sha256=fp,
            )

        if delta_mode == "single":
            for t_idx in range(n_target):
                t_chunk = target_raw_chunks[t_idx]
                t_len = len(t_chunk)
                t_fp = hashlib.sha256(t_chunk).hexdigest()

                # 1. Exact match
                if t_fp in base_exact_hash:
                    exact_count += 1
                    stored_bytes_total += 32
                    chunk_manifest.append({
                        "chunk_index": t_idx,
                        "type": "exact",
                        "chunk_sha256": t_fp,
                        "base_chunk_sha256": t_fp,
                        "chunk_size": t_len,
                        "stored_size": 32,
                    })
                    continue

                # 2. Candidate similarity retrieval
                cands = indexer.query_candidates(
                    query_id=-1,
                    initial_feature=target_feats[t_idx],
                    embedding=target_embs[t_idx],
                    sha256=t_fp,
                )
                best_ref_idx: Optional[int] = None
                best_sim = -1.0
                for c_id in cands:
                    if 0 <= c_id < n_base:
                        sim = float(target_embs[t_idx] @ base_embs[c_id])
                        if sim >= resemblance_threshold and sim > best_sim:
                            best_sim = sim
                            best_ref_idx = c_id

                delta_applied = False
                if best_ref_idx is not None:
                    ref_bytes = base_raw_chunks[best_ref_idx]
                    delta_res = evaluate_and_encode_delta(ref_bytes, t_chunk, compression_level=6)
                    if delta_res.is_accepted:
                        delta_sha = self.store_delta(delta_res.delta_container)
                        ref_sha = hashlib.sha256(ref_bytes).hexdigest()
                        delta_count += 1
                        stored_bytes_total += delta_res.net_stored_size
                        chunk_manifest.append({
                            "chunk_index": t_idx,
                            "type": "delta_sr",
                            "chunk_sha256": t_fp,
                            "delta_container_sha256": delta_sha,
                            "base_chunk_sha256": ref_sha,
                            "chunk_size": t_len,
                            "stored_size": delta_res.net_stored_size,
                        })
                        delta_applied = True

                if not delta_applied:
                    comp = compress_standalone(t_chunk)
                    comp_sha = self.store_chunk(comp)
                    standalone_count += 1
                    stored_bytes_total += len(comp)
                    chunk_manifest.append({
                        "chunk_index": t_idx,
                        "type": "standalone",
                        "chunk_sha256": t_fp,
                        "compressed_chunk_sha256": comp_sha,
                        "chunk_size": t_len,
                        "stored_size": len(comp),
                    })

        else:
            # Multi-reference delta mode (MRD)
            t_idx = 0
            while t_idx < n_target:
                t_chunk = target_raw_chunks[t_idx]
                t_len = len(t_chunk)
                t_fp = hashlib.sha256(t_chunk).hexdigest()

                if t_fp in base_exact_hash:
                    exact_count += 1
                    stored_bytes_total += 32
                    chunk_manifest.append({
                        "chunk_index": t_idx,
                        "type": "exact",
                        "chunk_sha256": t_fp,
                        "base_chunk_sha256": t_fp,
                        "chunk_size": t_len,
                        "stored_size": 32,
                    })
                    t_idx += 1
                    continue

                w_indices = []
                curr = t_idx
                while curr < n_target and len(w_indices) < mrd_window_size:
                    curr_fp = hashlib.sha256(target_raw_chunks[curr]).hexdigest()
                    if curr_fp in base_exact_hash:
                        break
                    w_indices.append(curr)
                    curr += 1

                if not w_indices:
                    w_indices = [t_idx]
                    curr = t_idx + 1

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

                target_window_chunks = [target_raw_chunks[j] for j in w_indices]

                mrd_applied = False
                if ref_cand_set:
                    ref_chunks = [base_raw_chunks[c] for c in ref_cand_set]
                    ref_fps = [hashlib.sha256(rc).hexdigest() for rc in ref_chunks]
                    mrd_res = evaluate_and_encode_mrd(ref_chunks, target_window_chunks, compression_level=6)

                    if mrd_res.is_accepted and mrd_res.mrd_container:
                        mrd_sha = self.store_delta(mrd_res.mrd_container)
                        delta_count += len(w_indices)
                        stored_bytes_total += mrd_res.net_stored_size
                        chunk_mrd_share = mrd_res.net_stored_size // len(w_indices)

                        for sub_idx, orig_j in enumerate(w_indices):
                            sub_c = target_raw_chunks[orig_j]
                            chunk_manifest.append({
                                "chunk_index": orig_j,
                                "type": "delta_mrd",
                                "chunk_sha256": hashlib.sha256(sub_c).hexdigest(),
                                "delta_container_sha256": mrd_sha,
                                "subchunk_index": sub_idx,
                                "ref_chunk_sha256_list": ref_fps,
                                "chunk_size": len(sub_c),
                                "stored_size": chunk_mrd_share,
                            })
                        mrd_applied = True

                if not mrd_applied:
                    for orig_j in w_indices:
                        sub_c = target_raw_chunks[orig_j]
                        comp = compress_standalone(sub_c)
                        comp_sha = self.store_chunk(comp)
                        standalone_count += 1
                        stored_bytes_total += len(comp)
                        chunk_manifest.append({
                            "chunk_index": orig_j,
                            "type": "standalone",
                            "chunk_sha256": hashlib.sha256(sub_c).hexdigest(),
                            "compressed_chunk_sha256": comp_sha,
                            "chunk_size": len(sub_c),
                            "stored_size": len(comp),
                        })

                t_idx = curr

        raw_bytes = len(target_bytes)
        savings_pct = round(((raw_bytes - stored_bytes_total) / raw_bytes) * 100.0, 2) if raw_bytes > 0 else 0.0
        ratio = round(raw_bytes / stored_bytes_total, 2) if stored_bytes_total > 0 else 1.0

        recipe = {
            "revision_id": revision_id,
            "target_name": target_name,
            "base_name": base_name,
            "timestamp": timestamp,
            "raw_bytes": raw_bytes,
            "stored_bytes": stored_bytes_total,
            "storage_savings_pct": savings_pct,
            "compression_ratio": ratio,
            "sha256": target_sha256,
            "base_sha256": base_sha256,
            "candidate_strategy": candidate_strategy,
            "delta_mode": delta_mode,
            "chunk_count": n_target,
            "exact_duplicates_count": exact_count,
            "delta_chunks_count": delta_count,
            "standalone_chunks_count": standalone_count,
            "chunks": chunk_manifest,
        }

        # Verify immediate restoration before committing recipe to disk
        restored_bytes = self._restore_manifest(chunk_manifest)
        if hashlib.sha256(restored_bytes).hexdigest() != target_sha256:
            raise StorageCorruptionError("Immediate post-encode reconstruction validation failed.")

        # Persist recipe atomically
        self._atomic_write_json(self.revisions_dir / f"{revision_id}.json", recipe)
        return recipe

    def _restore_manifest(self, chunk_manifest: list[dict[str, Any]]) -> bytes:
        """Internal worker to reconstruct raw bytes directly from a chunk manifest."""
        restored_chunks: list[bytes] = []
        mrd_group_cache: dict[str, list[bytes]] = {}

        for entry in chunk_manifest:
            entry_type = entry.get("type")
            expected_sha = entry.get("chunk_sha256")

            if entry_type == "exact":
                base_sha = entry["base_chunk_sha256"]
                chunk_bytes = self.get_chunk(base_sha)

            elif entry_type == "standalone":
                comp_sha = entry["compressed_chunk_sha256"]
                comp_bytes = self.get_chunk(comp_sha)
                try:
                    chunk_bytes = zlib.decompress(comp_bytes)
                except Exception as exc:
                    raise StorageCorruptionError(f"Failed to decompress standalone chunk: {exc}") from exc

            elif entry_type == "delta_sr":
                delta_sha = entry["delta_container_sha256"]
                base_sha = entry["base_chunk_sha256"]
                delta_bytes = self.get_delta(delta_sha)
                base_bytes = self.get_chunk(base_sha)
                try:
                    chunk_bytes = decode_delta(base_bytes, delta_bytes)
                except Exception as exc:
                    raise StorageCorruptionError(f"Failed to decode single-reference delta: {exc}") from exc

            elif entry_type == "delta_mrd":
                delta_sha = entry["delta_container_sha256"]
                subchunk_idx = entry["subchunk_index"]
                ref_shas = entry.get("ref_chunk_sha256_list", [])

                if delta_sha in mrd_group_cache:
                    subchunks = mrd_group_cache[delta_sha]
                    if subchunk_idx >= len(subchunks):
                        raise StorageCorruptionError(
                            f"MRD subchunk index {subchunk_idx} out of range ({len(subchunks)})."
                        )
                    chunk_bytes = subchunks[subchunk_idx]
                else:
                    delta_bytes = self.get_delta(delta_sha)
                    ref_chunks = [self.get_chunk(rsha) for rsha in ref_shas]
                    try:
                        subchunks = decode_mrd_group(ref_chunks, delta_bytes)
                        mrd_group_cache[delta_sha] = subchunks
                        if subchunk_idx >= len(subchunks):
                            raise StorageCorruptionError(
                                f"MRD subchunk index {subchunk_idx} out of range ({len(subchunks)})."
                            )
                        chunk_bytes = subchunks[subchunk_idx]
                    except Exception as exc:
                        raise StorageCorruptionError(f"Failed to decode MRD container: {exc}") from exc
            else:
                raise StorageCorruptionError(f"Unknown chunk manifest entry type: '{entry_type}'.")

            if expected_sha and hashlib.sha256(chunk_bytes).hexdigest() != expected_sha:
                raise StorageCorruptionError(
                    f"Restored chunk {entry.get('chunk_index')} failed SHA-256 integrity validation."
                )

            restored_chunks.append(chunk_bytes)

        return b"".join(restored_chunks)

    def list_revisions(self) -> list[dict[str, Any]]:
        """List all persisted revision recipes, sorted newest first."""
        revisions: list[dict[str, Any]] = []
        for path in self.revisions_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    recipe = json.load(f)
                    revisions.append({
                        "revision_id": recipe.get("revision_id", path.stem),
                        "target_name": recipe.get("target_name", "unknown"),
                        "base_name": recipe.get("base_name", "unknown"),
                        "timestamp": recipe.get("timestamp", ""),
                        "raw_bytes": recipe.get("raw_bytes", 0),
                        "stored_bytes": recipe.get("stored_bytes", 0),
                        "storage_savings_pct": recipe.get("storage_savings_pct", 0.0),
                        "compression_ratio": recipe.get("compression_ratio", 1.0),
                        "sha256": recipe.get("sha256", ""),
                        "delta_mode": recipe.get("delta_mode", "single"),
                        "chunk_count": recipe.get("chunk_count", 0),
                    })
            except Exception as exc:
                logger.warning("Failed to load revision recipe %s: %s", path, exc)

        revisions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return revisions

    def get_revision(self, revision_id: str) -> dict[str, Any]:
        """Fetch full revision recipe by ID."""
        valid_id = self._validate_revision_id(revision_id)
        recipe_path = self.revisions_dir / f"{valid_id}.json"
        if not recipe_path.exists():
            raise RevisionNotFoundError(f"Revision '{valid_id}' not found.")
        try:
            with open(recipe_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise StorageCorruptionError(f"Revision recipe '{valid_id}' is corrupted: {exc}") from exc

    def restore_revision(self, revision_id: str) -> bytes:
        """Fully restore byte-exact revision data from disk manifest, verifying SHA-256."""
        recipe = self.get_revision(revision_id)
        expected_sha = recipe.get("sha256", "")
        chunks = recipe.get("chunks", [])

        restored_bytes = self._restore_manifest(chunks)
        actual_sha = hashlib.sha256(restored_bytes).hexdigest()
        if expected_sha and actual_sha != expected_sha:
            raise StorageCorruptionError(
                f"Restored revision '{revision_id}' checksum mismatch: expected {expected_sha}, got {actual_sha}."
            )
        return restored_bytes
