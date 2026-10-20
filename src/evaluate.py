"""Evaluation and Benchmarking for Trained DCLC Transformer Embeddings.

Phase 3.2 Evaluation Module:
1. Loads trained model checkpoint and prepared dataset/manifest.
2. Generates L2-normalized semantic representation vectors under model.eval().
3. Computes cosine similarity matrices for trained embeddings and baseline initial features.
4. Evaluates:
   - Exact-duplicate detection (ground truth: SHA-256 match).
   - Similar-content resemblance detection (ground truth: version-modified pairs vs unrelated controls).
5. Computes precision, recall, F1, and accuracy with safe division handling.
6. Compares trained Transformer representations against the initial feature extractor baseline.
7. Reports diagnostic statistics on embedding distribution and highlights limitations of
   the small 51-chunk training set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.model import DCLCTransformerContextEncoder


def load_model_and_checkpoint(
    checkpoint_path: Path | str,
    d_model: int = 128,
    output_dim: int = 128,
    device: str | torch.device = "cpu",
    input_norm: Optional[str] = None,
) -> DCLCTransformerContextEncoder:
    """Load the DCLC context-aware Transformer encoder and its trained weights.

    Args:
        checkpoint_path: Path to PyTorch .pt state_dict checkpoint.
        d_model: Hidden dimension of Transformer encoder.
        output_dim: Dimensionality of output projection head.
        device: Device to load model onto ('cpu' or 'cuda').
        input_norm: Input normalization mode ('none', 'layernorm', or None for auto-detection).

    Returns:
        DCLCTransformerContextEncoder instance set to eval() mode.
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)

    # Auto-detect input_norm architecture from checkpoint if not explicitly provided
    if input_norm is None:
        if "input_layer_norm.weight" in state_dict:
            input_norm = "layernorm"
        else:
            input_norm = "none"

    model = DCLCTransformerContextEncoder(
        d_model=d_model,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        output_dim=output_dim,
        normalize_output=True,
        input_norm=input_norm,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_embeddings(
    model: DCLCTransformerContextEncoder,
    context_sequences: np.ndarray,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Generate L2-normalized embeddings for a sequence of context windows.

    Args:
        model: Trained DCLCTransformerContextEncoder in eval() mode.
        context_sequences: Array of shape (N, 2K+1, D), e.g. (N, 9, 128).
        batch_size: Inference batch size.
        device: Device to run inference on.

    Returns:
        Array of shape (N, output_dim) containing L2-normalized embeddings.
    """
    if len(context_sequences) == 0:
        return np.empty((0, model.output_dim), dtype=np.float32)

    model.eval()
    embeddings_list = []
    total = len(context_sequences)

    with torch.no_grad():
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = torch.from_numpy(context_sequences[start_idx:end_idx]).float().to(device)
            out = model(batch)
            embeddings_list.append(out.cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0).astype(np.float32)
    return embeddings


def compute_cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for an array of vectors.

    If vectors are already L2-normalized, this is equivalent to matrix multiplication.
    Otherwise, vectors are normalized before computing the dot product.

    Args:
        vectors: Array of shape (N, D).

    Returns:
        Symmetric array of shape (N, N) with values in [-1.0, 1.0].
    """
    if len(vectors) == 0:
        return np.empty((0, 0), dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    sim = normalized @ normalized.T
    return np.clip(sim, -1.0, 1.0).astype(np.float32)


def compute_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute precision, recall, F1, and accuracy with safe division handling.

    Args:
        y_true: Boolean or binary 1D array of ground-truth positive pairs.
        y_pred: Boolean or binary 1D array of predicted positive pairs.

    Returns:
        Dictionary with precision, recall, f1, accuracy, tp, fp, fn, tn counts.
    """
    y_true_b = np.asarray(y_true, dtype=bool)
    y_pred_b = np.asarray(y_pred, dtype=bool)

    if len(y_true_b) != len(y_pred_b):
        raise ValueError("y_true and y_pred must have identical length")

    tp = int(np.sum(y_true_b & y_pred_b))
    fp = int(np.sum((~y_true_b) & y_pred_b))
    fn = int(np.sum(y_true_b & (~y_pred_b)))
    tn = int(np.sum((~y_true_b) & (~y_pred_b)))

    total = len(y_true_b)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
    }


def evaluate_duplicate_detection(
    similarity_matrix: np.ndarray,
    fingerprints: list[str],
    file_ids: np.ndarray,
    threshold: float = 0.99,
    cross_file_only: bool = True,
) -> dict[str, Any]:
    """Evaluate duplicate detection against exact cryptographic fingerprints (SHA-256).

    Ground-truth positive pair: two chunks have identical SHA-256 hashes.
    Predicted positive pair: cosine similarity >= threshold.

    Args:
        similarity_matrix: (N, N) pairwise cosine similarity matrix.
        fingerprints: List of length N containing SHA-256 hex strings.
        file_ids: 1D array of length N tracking file origins.
        threshold: Cosine similarity cutoff to classify as duplicate.
        cross_file_only: If True, only evaluate pairs from different files (i.e. deduplication across files).

    Returns:
        Dictionary of classification metrics and ground-truth counts.
    """
    n = len(fingerprints)
    if similarity_matrix.shape != (n, n):
        raise ValueError("similarity_matrix shape does not match fingerprints count")

    y_true_list = []
    y_pred_list = []

    for i in range(n):
        for j in range(i + 1, n):
            if cross_file_only and file_ids[i] == file_ids[j]:
                continue
            is_dup = (fingerprints[i] == fingerprints[j])
            is_pred_dup = bool(similarity_matrix[i, j] >= threshold)
            y_true_list.append(is_dup)
            y_pred_list.append(is_pred_dup)

    y_true = np.array(y_true_list, dtype=bool)
    y_pred = np.array(y_pred_list, dtype=bool)

    metrics = compute_precision_recall_f1(y_true, y_pred)
    metrics["threshold"] = threshold
    metrics["total_duplicate_pairs"] = int(np.sum(y_true))
    metrics["total_non_duplicate_pairs"] = int(np.sum(~y_true))
    return metrics


def evaluate_similar_content_detection(
    similarity_matrix: np.ndarray,
    file_ids: np.ndarray,
    chunk_indices: np.ndarray,
    fingerprints: list[str],
    id_v1: int = 1,
    id_v2: int = 2,
    id_unrelated: int = 3,
    threshold: float = 0.70,
) -> dict[str, Any]:
    """Evaluate resemblance detection on known modified chunks vs unrelated negative controls.

    Ground-truth labels:
    - Positive resemblance pair: non-identical chunks originating from the same sequence position
      between document v1 and document v2 (e.g., chunk 1, 2, 7 where content was edited).
    - Negative pair: chunks between document v1 and unrelated document (completely disjoint topics).

    Args:
        similarity_matrix: (N, N) pairwise cosine similarity matrix.
        file_ids: 1D array of file IDs.
        chunk_indices: 1D array of chunk sequence indices.
        fingerprints: List of SHA-256 hex strings.
        id_v1: File ID of version 1.
        id_v2: File ID of version 2.
        id_unrelated: File ID of unrelated document.
        threshold: Cosine similarity cutoff to classify as similar.

    Returns:
        Dictionary with classification metrics and pairwise similarity statistics.
    """
    n = len(file_ids)
    idx_v1 = np.where(file_ids == id_v1)[0]
    idx_v2 = np.where(file_ids == id_v2)[0]
    idx_unrelated = np.where(file_ids == id_unrelated)[0]

    if len(idx_v1) == 0 or len(idx_v2) == 0 or len(idx_unrelated) == 0:
        return {
            "status": "insufficient_files",
            "message": "Required file IDs not present in dataset.",
        }

    # Map chunk index to global index
    v1_map = {chunk_indices[g_idx]: g_idx for g_idx in idx_v1}
    v2_map = {chunk_indices[g_idx]: g_idx for g_idx in idx_v2}

    positive_sims = []
    # Identify non-identical pairs at the same sequence index
    for c_idx in sorted(v1_map.keys()):
        if c_idx in v2_map:
            g1 = v1_map[c_idx]
            g2 = v2_map[c_idx]
            # Non-identical modified chunks only
            if fingerprints[g1] != fingerprints[g2]:
                positive_sims.append(float(similarity_matrix[g1, g2]))

    # Negative pairs: all pairs between v1 and unrelated
    negative_sims = []
    for g1 in idx_v1:
        for gu in idx_unrelated:
            negative_sims.append(float(similarity_matrix[g1, gu]))

    y_true = np.concatenate([
        np.ones(len(positive_sims), dtype=bool),
        np.zeros(len(negative_sims), dtype=bool),
    ])
    y_pred = np.concatenate([
        np.array(positive_sims) >= threshold,
        np.array(negative_sims) >= threshold,
    ])

    metrics = compute_precision_recall_f1(y_true, y_pred)
    metrics["threshold"] = threshold
    metrics["num_positive_pairs"] = len(positive_sims)
    metrics["num_negative_pairs"] = len(negative_sims)
    metrics["positive_sim_mean"] = float(np.mean(positive_sims)) if positive_sims else 0.0
    metrics["positive_sim_min"] = float(np.min(positive_sims)) if positive_sims else 0.0
    metrics["positive_sim_max"] = float(np.max(positive_sims)) if positive_sims else 0.0
    metrics["negative_sim_mean"] = float(np.mean(negative_sims)) if negative_sims else 0.0
    metrics["negative_sim_min"] = float(np.min(negative_sims)) if negative_sims else 0.0
    metrics["negative_sim_max"] = float(np.max(negative_sims)) if negative_sims else 0.0
    return metrics


def run_full_evaluation(
    dataset_path: Path | str,
    manifest_path: Path | str,
    checkpoint_path: Path | str,
    dup_threshold: float = 0.99,
    sim_threshold: float = 0.70,
    device: str = "cpu",
    input_norm: Optional[str] = None,
) -> dict[str, Any]:
    """Execute complete comparative evaluation between trained Transformer and baseline.

    Args:
        dataset_path: Path to dclc_train_data.npz.
        manifest_path: Path to dataset_manifest.json.
        checkpoint_path: Path to trained dclc_encoder.pt.
        dup_threshold: Threshold for duplicate detection.
        sim_threshold: Threshold for resemblance detection.
        device: Device for inference.
        input_norm: Input normalization mode ('none', 'layernorm', or None for auto-detection).

    Returns:
        Structured evaluation report.
    """
    # 1. Load data and manifest
    npz_data = np.load(dataset_path)
    context_sequences = npz_data["context_sequences"].astype(np.float32)
    initial_features = npz_data["initial_features"].astype(np.float32)
    file_ids = npz_data["file_ids"].astype(np.int32)
    chunk_indices = npz_data["chunk_indices"].astype(np.int32)

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    fingerprints = [c["sha256"] for c in manifest["chunks"]]

    # 2. Load model and generate embeddings
    model = load_model_and_checkpoint(checkpoint_path, device=device, input_norm=input_norm)
    trained_embeddings = generate_embeddings(model, context_sequences, device=device)

    # 3. Compute similarity matrices
    sim_matrix_trained = compute_cosine_similarity_matrix(trained_embeddings)
    sim_matrix_baseline = compute_cosine_similarity_matrix(initial_features)

    # 4. Check embedding space collapse / dispersion
    std_per_dim = trained_embeddings.std(axis=0)
    mean_std = float(std_per_dim.mean())
    is_collapsed = bool(mean_std < 1e-3)

    # 5. Evaluate duplicate detection
    dup_eval_trained = evaluate_duplicate_detection(
        sim_matrix_trained, fingerprints, file_ids, threshold=dup_threshold, cross_file_only=True
    )
    dup_eval_baseline = evaluate_duplicate_detection(
        sim_matrix_baseline, fingerprints, file_ids, threshold=dup_threshold, cross_file_only=True
    )

    # 6. Evaluate similar-content detection
    sim_eval_trained = evaluate_similar_content_detection(
        sim_matrix_trained, file_ids, chunk_indices, fingerprints, threshold=sim_threshold
    )
    sim_eval_baseline = evaluate_similar_content_detection(
        sim_matrix_baseline, file_ids, chunk_indices, fingerprints, threshold=sim_threshold
    )

    # 7. Summary metrics table
    report = {
        "metadata": {
            "total_chunks": len(context_sequences),
            "feature_dim": trained_embeddings.shape[1],
            "checkpoint_path": str(checkpoint_path),
            "input_norm": model.input_norm,
            "is_representation_collapsed": is_collapsed,
            "mean_embedding_dimension_std": mean_std,
        },
        "duplicate_detection": {
            "trained_model": dup_eval_trained,
            "baseline_initial_features": dup_eval_baseline,
        },
        "similarity_detection": {
            "trained_model": sim_eval_trained,
            "baseline_initial_features": sim_eval_baseline,
        },
        "limitations": [
            "Dataset consists of only 51 chunks across 4 synthetic files (underfitting / small scope).",
            "Full evaluation requires standard benchmark corpora (BSD, LNX, GCC, STOCK) with diverse delta patterns.",
        ],
    }
    return report


def main() -> None:
    """CLI runner for DCLC model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained DCLC Transformer model against baseline.")
    parser.add_argument("--dataset", type=str, default="data/processed/dclc_train_data.npz", help="Path to .npz dataset")
    parser.add_argument("--manifest", type=str, default="data/processed/dataset_manifest.json", help="Path to manifest JSON")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dclc_encoder.pt", help="Path to trained checkpoint")
    parser.add_argument("--input-norm", type=str, default=None, help="Input norm architecture ('none', 'layernorm', or None for auto-detect)")
    parser.add_argument("--dup-threshold", type=float, default=0.99, help="Duplicate similarity threshold")
    parser.add_argument("--sim-threshold", type=float, default=0.70, help="Resemblance similarity threshold")

    args = parser.parse_args()

    print("=" * 78)
    print(" DCLC Phase 3.2 / 3.4: Trained Model & Representation Evaluation")
    print("=" * 78)

    report = run_full_evaluation(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        dup_threshold=args.dup_threshold,
        sim_threshold=args.sim_threshold,
        input_norm=args.input_norm,
    )

    meta = report["metadata"]
    print(f"\nTotal Evaluated Chunks: {meta['total_chunks']}")
    print(f"Embedding Dimension:   {meta['feature_dim']}")
    print(f"Checkpoint Path:       {meta['checkpoint_path']}")
    print(f"Input Normalization:   {meta['input_norm']}")
    print(f"Representation Status: {'COLLAPSED (Low Variance)' if meta['is_representation_collapsed'] else 'Dispersed'}")
    print(f"Mean Per-Dim Std:      {meta['mean_embedding_dimension_std']:.6f}")

    dup_t = report["duplicate_detection"]["trained_model"]
    dup_b = report["duplicate_detection"]["baseline_initial_features"]

    print("\n" + "-" * 78)
    print(f" 1. EXACT DUPLICATE DETECTION (Threshold >= {args.dup_threshold:.2f}, Cross-File Pairs)")
    print("-" * 78)
    print(f"{'Metric':<25} | {'Trained Transformer':<22} | {'Initial Feat Baseline':<22}")
    print("-" * 78)
    print(f"{'Precision':<25} | {dup_t['precision']:<22.4f} | {dup_b['precision']:<22.4f}")
    print(f"{'Recall':<25} | {dup_t['recall']:<22.4f} | {dup_b['recall']:<22.4f}")
    print(f"{'F1 Score':<25} | {dup_t['f1']:<22.4f} | {dup_b['f1']:<22.4f}")
    print(f"{'Accuracy':<25} | {dup_t['accuracy']:<22.4f} | {dup_b['accuracy']:<22.4f}")
    dup_t_conf = f"{dup_t['tp']}/{dup_t['fp']}/{dup_t['fn']}/{dup_t['tn']}"
    dup_b_conf = f"{dup_b['tp']}/{dup_b['fp']}/{dup_b['fn']}/{dup_b['tn']}"
    print(f"{'TP / FP / FN / TN':<25} | {dup_t_conf:<22} | {dup_b_conf:<22}")

    sim_t = report["similarity_detection"]["trained_model"]
    sim_b = report["similarity_detection"]["baseline_initial_features"]

    print("\n" + "-" * 78)
    print(f" 2. RESEMBLANCE / SIMILAR CONTENT DETECTION (Threshold >= {args.sim_threshold:.2f})")
    print(f"    (Modified v1-vs-v2 aligned chunks vs. v1-vs-unrelated controls)")
    print("-" * 78)
    print(f"{'Metric':<25} | {'Trained Transformer':<22} | {'Initial Feat Baseline':<22}")
    print("-" * 78)
    print(f"{'Precision':<25} | {sim_t['precision']:<22.4f} | {sim_b['precision']:<22.4f}")
    print(f"{'Recall':<25} | {sim_t['recall']:<22.4f} | {sim_b['recall']:<22.4f}")
    print(f"{'F1 Score':<25} | {sim_t['f1']:<22.4f} | {sim_b['f1']:<22.4f}")
    print(f"{'Accuracy':<25} | {sim_t['accuracy']:<22.4f} | {sim_b['accuracy']:<22.4f}")
    print(f"{'Positive Sim (Mean)':<25} | {sim_t['positive_sim_mean']:<22.4f} | {sim_b['positive_sim_mean']:<22.4f}")
    print(f"{'Negative Sim (Mean)':<25} | {sim_t['negative_sim_mean']:<22.4f} | {sim_b['negative_sim_mean']:<22.4f}")
    sim_t_conf = f"{sim_t['tp']}/{sim_t['fp']}/{sim_t['fn']}/{sim_t['tn']}"
    sim_b_conf = f"{sim_b['tp']}/{sim_b['fp']}/{sim_b['fn']}/{sim_b['tn']}"
    print(f"{'TP / FP / FN / TN':<25} | {sim_t_conf:<22} | {sim_b_conf:<22}")

    print("\n" + "-" * 78)
    print(" 3. LIMITATIONS & FINDINGS")
    print("-" * 78)
    for lim in report["limitations"]:
        print(f" - {lim}")
    print("=" * 78)


if __name__ == "__main__":
    main()
