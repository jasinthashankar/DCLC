"""Tests for Phase 3.2 Evaluation Module (src/evaluate.py)."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.evaluate import (
    load_model_and_checkpoint,
    generate_embeddings,
    compute_cosine_similarity_matrix,
    compute_precision_recall_f1,
    evaluate_duplicate_detection,
    evaluate_similar_content_detection,
    run_full_evaluation,
)
from src.model import DCLCTransformerContextEncoder


def test_compute_precision_recall_f1_metrics():
    # Case 1: Perfect classification
    y_true = np.array([True, True, False, False])
    y_pred = np.array([True, True, False, False])
    m = compute_precision_recall_f1(y_true, y_pred)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["tp"] == 2 and m["tn"] == 2 and m["fp"] == 0 and m["fn"] == 0

    # Case 2: Partial overlap
    y_true = np.array([True, True, False, False])
    y_pred = np.array([True, False, True, False])
    m = compute_precision_recall_f1(y_true, y_pred)
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["accuracy"] == 0.5

    # Case 3: Zero predictions (no division by zero)
    y_true = np.array([True, True])
    y_pred = np.array([False, False])
    m = compute_precision_recall_f1(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["accuracy"] == 0.0

    # Case 4: Mismatched length raises ValueError
    with pytest.raises(ValueError, match="identical length"):
        compute_precision_recall_f1(np.array([True]), np.array([True, False]))


def test_compute_cosine_similarity_matrix():
    # Empty input
    assert compute_cosine_similarity_matrix(np.empty((0, 128))).shape == (0, 0)

    # Orthogonal vectors
    v = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=np.float32)
    sim = compute_cosine_similarity_matrix(v)

    assert sim.shape == (3, 3)
    np.testing.assert_allclose(np.diag(sim), [1.0, 1.0, 1.0], atol=1e-6)
    assert sim[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert sim[0, 2] == pytest.approx(-1.0, abs=1e-6)
    assert np.all(sim >= -1.0) and np.all(sim <= 1.0)


def test_load_model_and_checkpoint_nonexistent():
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_model_and_checkpoint("nonexistent_checkpoint.pt")


def test_generate_embeddings_properties(tmp_path: Path):
    # Create small model and random inputs
    model = DCLCTransformerContextEncoder(d_model=32, output_dim=32)
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded_model = load_model_and_checkpoint(ckpt_path, d_model=32, output_dim=32)
    context_seqs = np.random.randn(6, 9, 32).astype(np.float32)

    embeddings = generate_embeddings(loaded_model, context_seqs, batch_size=2)
    assert embeddings.shape == (6, 32)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()

    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones(6), atol=1e-5)

    # Empty context sequence
    empty_emb = generate_embeddings(loaded_model, np.empty((0, 9, 32), dtype=np.float32))
    assert empty_emb.shape == (0, 32)


def test_evaluate_duplicate_detection_logic():
    # 4 chunks:
    # Chunk 0 (File 0): hash A
    # Chunk 1 (File 0): hash B
    # Chunk 2 (File 1): hash A (duplicate of chunk 0)
    # Chunk 3 (File 1): hash C
    fingerprints = ["hash_A", "hash_B", "hash_A", "hash_C"]
    file_ids = np.array([0, 0, 1, 1], dtype=np.int32)

    # Simulated similarity matrix
    sim = np.array([
        [1.0, 0.2, 0.99, 0.1],
        [0.2, 1.0, 0.3, 0.1],
        [0.99, 0.3, 1.0, 0.4],
        [0.1, 0.1, 0.4, 1.0],
    ], dtype=np.float32)

    res = evaluate_duplicate_detection(sim, fingerprints, file_ids, threshold=0.95, cross_file_only=True)
    # Cross-file pairs are: (0,2), (0,3), (1,2), (1,3) -> total 4 pairs
    # True duplicate is (0,2) where fingerprints are both "hash_A" -> 1 TP
    assert res["total_duplicate_pairs"] == 1
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


def test_evaluate_similar_content_detection_logic():
    # File 1 (chunks 0, 1) and File 2 (chunks 2, 3) and Unrelated (chunk 4)
    file_ids = np.array([1, 1, 2, 2, 3], dtype=np.int32)
    chunk_indices = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    fingerprints = ["h0", "h1_v1", "h0", "h1_v2", "h_unrelated"]

    # Sim matrix where:
    # (1, 3) is a modified chunk pair with sim 0.85
    # (0, 4) and (1, 4) are unrelated pairs with sim 0.10
    sim = np.full((5, 5), 0.10, dtype=np.float32)
    np.fill_diagonal(sim, 1.0)
    sim[1, 3] = sim[3, 1] = 0.85

    res = evaluate_similar_content_detection(
        sim, file_ids, chunk_indices, fingerprints, id_v1=1, id_v2=2, id_unrelated=3, threshold=0.70
    )
    assert res["num_positive_pairs"] == 1
    assert res["num_negative_pairs"] == 2
    assert res["tp"] == 1
    assert res["fp"] == 0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


def test_run_full_evaluation_end_to_end():
    # Test end-to-end evaluation using the trained checkpoint and dataset
    ckpt_path = Path("checkpoints/dclc_encoder.pt")
    dataset_path = Path("data/processed/dclc_train_data.npz")
    manifest_path = Path("data/processed/dataset_manifest.json")

    if not (ckpt_path.exists() and dataset_path.exists() and manifest_path.exists()):
        pytest.skip("Trained checkpoint or dataset not present for end-to-end test.")

    report = run_full_evaluation(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        checkpoint_path=ckpt_path,
        dup_threshold=0.99,
        sim_threshold=0.70,
    )

    assert "metadata" in report
    assert "duplicate_detection" in report
    assert "similarity_detection" in report
    assert "limitations" in report

    assert report["metadata"]["total_chunks"] == 51
    assert report["metadata"]["feature_dim"] == 128
    assert "is_representation_collapsed" in report["metadata"]

    # Verify duplicate detection structure
    dup_t = report["duplicate_detection"]["trained_model"]
    dup_b = report["duplicate_detection"]["baseline_initial_features"]
    assert 0.0 <= dup_t["precision"] <= 1.0
    assert 0.0 <= dup_b["precision"] <= 1.0

    # Verify resemblance structure
    sim_t = report["similarity_detection"]["trained_model"]
    sim_b = report["similarity_detection"]["baseline_initial_features"]
    assert 0.0 <= sim_t["recall"] <= 1.0
    assert 0.0 <= sim_b["recall"] <= 1.0


def test_load_model_and_checkpoint_autodetect_input_norm():
    orig_ckpt = Path("checkpoints/dclc_encoder.pt")
    corr_ckpt = Path("checkpoints/dclc_encoder_corrected.pt")

    if not (orig_ckpt.exists() and corr_ckpt.exists()):
        pytest.skip("Checkpoints not available for autodetect test.")

    # 1. Original checkpoint should auto-detect 'none'
    m_orig = load_model_and_checkpoint(orig_ckpt)
    assert m_orig.input_norm == "none"
    assert m_orig.input_layer_norm is None

    # 2. Corrected checkpoint should auto-detect 'layernorm'
    m_corr = load_model_and_checkpoint(corr_ckpt)
    assert m_corr.input_norm == "layernorm"
    assert m_corr.input_layer_norm is not None


def test_run_full_evaluation_corrected_checkpoint():
    corr_ckpt = Path("checkpoints/dclc_encoder_corrected.pt")
    dataset_path = Path("data/processed/dclc_train_data.npz")
    manifest_path = Path("data/processed/dataset_manifest.json")

    if not (corr_ckpt.exists() and dataset_path.exists() and manifest_path.exists()):
        pytest.skip("Corrected checkpoint not available for evaluation test.")

    report = run_full_evaluation(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        checkpoint_path=corr_ckpt,
        dup_threshold=0.99,
        sim_threshold=0.70,
    )

    assert report["metadata"]["input_norm"] == "layernorm"
    assert report["metadata"]["is_representation_collapsed"] is False
    assert report["metadata"]["mean_embedding_dimension_std"] > 0.05

