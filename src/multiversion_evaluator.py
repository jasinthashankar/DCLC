"""Multi-Version Evaluation and Benchmarking for DCLC (Phase 4B).

Evaluates representations on the held-out Test split (families 085-099).
Strictly separates:
1. Exact-duplicate detection (SHA-256 match, threshold 0.99).
2. Version-resemblance detection (lineage Jaccard >= 0.50, threshold 0.70).
3. Ambiguous boundary-shift pairs (0.10 <= Jaccard < 0.50) are isolated and
   excluded from supervised classification metrics.
4. Unrelated negative controls (cross-family and intra-family Jaccard < 0.10).

Compares:
- Initial-feature baseline (128-dim normalized initial features).
- Phase 3.4 corrected model (checkpoints/dclc_encoder_corrected.pt).
- Phase 4B version-aware model (checkpoints/dclc_encoder_version_aware.pt).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from src.evaluate import compute_precision_recall_f1, load_model_and_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_distribution_stats(values: list[float] | np.ndarray) -> dict[str, float]:
    """Compute distribution statistics for a sequence of similarity values."""
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def evaluate_test_split(
    manifest_path: str | Path = "data/multi_version/processed/multiversion_manifest.json",
    data_npz_path: str | Path = "data/multi_version/processed/multiversion_train_data.npz",
    version_aware_checkpoint: str | Path = "checkpoints/dclc_encoder_version_aware.pt",
    corrected_checkpoint: str | Path = "checkpoints/dclc_encoder_corrected.pt",
    resemblance_threshold: float = 0.70,
    duplicate_threshold: float = 0.99,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run full comparative evaluation on the held-out test split."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    chunks = manifest["chunks"]
    test_chunks = [c for c in chunks if c["split"] == "test"]
    n_test = len(test_chunks)
    test_cids = [c["global_chunk_id"] for c in test_chunks]

    logger.info(f"Loaded {n_test} test chunks (families 085-099).")

    npz_data = np.load(data_npz_path)
    test_seqs = npz_data["context_sequences"][test_cids].astype(np.float32)
    test_feats = npz_data["initial_features"][test_cids].astype(np.float32)

    # 1. Compute similarity matrices
    # A. Initial features baseline
    norms = np.linalg.norm(test_feats, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    normed_feats = test_feats / norms
    sim_baseline = np.clip(normed_feats @ normed_feats.T, -1.0, 1.0)

    # B. Phase 3.4 corrected checkpoint
    model_corrected = load_model_and_checkpoint(corrected_checkpoint, device=device)
    with torch.no_grad():
        embs_corrected = model_corrected(torch.from_numpy(test_seqs).float().to(device)).cpu().numpy()
    sim_corrected = np.clip(embs_corrected @ embs_corrected.T, -1.0, 1.0)

    # C. Phase 4B version-aware checkpoint (if available)
    sim_va = None
    if Path(version_aware_checkpoint).exists():
        model_va = load_model_and_checkpoint(version_aware_checkpoint, device=device)
        with torch.no_grad():
            embs_va = model_va(torch.from_numpy(test_seqs).float().to(device)).cpu().numpy()
        sim_va = np.clip(embs_va @ embs_va.T, -1.0, 1.0)
    else:
        logger.warning(f"Version-aware checkpoint not found at {version_aware_checkpoint}.")

    # 2. Extract test pair categories
    exact_dup_pairs = []
    resemblance_pairs = []
    ambiguous_pairs = []
    unrelated_same_fam_pairs = []
    cross_fam_pairs = []

    for i in range(n_test):
        ca = test_chunks[i]
        sa = set(ca["base_record_ids"])
        for j in range(i + 1, n_test):
            cb = test_chunks[j]
            sb = set(cb["base_record_ids"])

            if ca["family_id"] != cb["family_id"]:
                cross_fam_pairs.append((i, j))
                continue

            if ca["file_id"] == cb["file_id"]:
                continue  # Same file pairs

            if ca["sha256"] == cb["sha256"]:
                exact_dup_pairs.append((i, j))
                continue

            if not (sa or sb):
                unrelated_same_fam_pairs.append((i, j))
                continue

            jacc = len(sa & sb) / len(sa | sb)
            if jacc >= 0.50:
                resemblance_pairs.append((i, j))
            elif jacc >= 0.10:
                ambiguous_pairs.append((i, j))
            else:
                unrelated_same_fam_pairs.append((i, j))

    logger.info(
        f"Test pairs: {len(exact_dup_pairs)} exact dup, {len(resemblance_pairs)} resemblance, "
        f"{len(ambiguous_pairs)} ambiguous (excluded), {len(unrelated_same_fam_pairs)} unrelated same-fam, "
        f"{len(cross_fam_pairs)} cross-fam."
    )

    systems = {
        "initial_feature_baseline": sim_baseline,
        "phase3_4_corrected_model": sim_corrected,
    }
    if sim_va is not None:
        systems["phase4b_version_aware_model"] = sim_va

    results = {
        "resemblance_threshold": resemblance_threshold,
        "duplicate_threshold": duplicate_threshold,
        "test_counts": {
            "chunks": n_test,
            "exact_duplicate_pairs": len(exact_dup_pairs),
            "resemblance_pairs": len(resemblance_pairs),
            "ambiguous_boundary_shift_pairs": len(ambiguous_pairs),
            "unrelated_same_family_pairs": len(unrelated_same_fam_pairs),
            "cross_family_pairs": len(cross_fam_pairs),
            "total_negative_pairs": len(unrelated_same_fam_pairs) + len(cross_fam_pairs),
        },
        "systems": {},
    }

    # Evaluate each system
    all_neg_pairs = unrelated_same_fam_pairs + cross_fam_pairs

    for sys_name, sim_mat in systems.items():
        # A. Similarity distributions
        sims_dup = [float(sim_mat[i, j]) for i, j in exact_dup_pairs]
        sims_resem = [float(sim_mat[i, j]) for i, j in resemblance_pairs]
        sims_amb = [float(sim_mat[i, j]) for i, j in ambiguous_pairs]
        sims_neg_intra = [float(sim_mat[i, j]) for i, j in unrelated_same_fam_pairs]
        sims_neg_cross = [float(sim_mat[i, j]) for i, j in cross_fam_pairs]
        sims_neg_all = sims_neg_intra + sims_neg_cross

        # B. Resemblance classification (positives: resemblance_pairs, negatives: all_neg_pairs)
        y_true_resem = np.array([True] * len(resemblance_pairs) + [False] * len(all_neg_pairs))
        y_pred_resem = np.array(
            [s >= resemblance_threshold for s in sims_resem]
            + [s >= resemblance_threshold for s in sims_neg_all]
        )
        metrics_resem = compute_precision_recall_f1(y_true_resem, y_pred_resem)

        # C. Exact duplicate classification (positives: exact_dup_pairs, negatives: cross_file non-duplicates)
        non_dup_pairs = resemblance_pairs + ambiguous_pairs + all_neg_pairs
        sims_non_dup = [float(sim_mat[i, j]) for i, j in non_dup_pairs]
        y_true_dup = np.array([True] * len(exact_dup_pairs) + [False] * len(non_dup_pairs))
        y_pred_dup = np.array(
            [s >= duplicate_threshold for s in sims_dup]
            + [s >= duplicate_threshold for s in sims_non_dup]
        )
        metrics_dup = compute_precision_recall_f1(y_true_dup, y_pred_dup)

        results["systems"][sys_name] = {
            "resemblance_metrics": metrics_resem,
            "duplicate_metrics": metrics_dup,
            "distributions": {
                "exact_duplicates": compute_distribution_stats(sims_dup),
                "version_resemblance": compute_distribution_stats(sims_resem),
                "ambiguous_boundary_shifts": compute_distribution_stats(sims_amb),
                "unrelated_same_family": compute_distribution_stats(sims_neg_intra),
                "unrelated_cross_family": compute_distribution_stats(sims_neg_cross),
                "all_negatives": compute_distribution_stats(sims_neg_all),
            },
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DCLC multi-version test split.")
    parser.add_argument("--version-aware-checkpoint", type=str, default="checkpoints/dclc_encoder_version_aware.pt")
    args = parser.parse_args()

    res = evaluate_test_split(version_aware_checkpoint=args.version_aware_checkpoint)
    print(json.dumps(res, indent=2))
