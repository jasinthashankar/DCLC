"""Tests for the DCLC Context-Aware Transformer Model."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.model import DCLCTransformerContextEncoder, SinusoidalPositionalEncoding


def test_sinusoidal_positional_encoding_shape():
    pos_enc = SinusoidalPositionalEncoding(d_model=128, max_len=32, dropout=0.0)
    x = torch.zeros(2, 9, 128)
    out = pos_enc(x)
    assert out.shape == (2, 9, 128)
    assert torch.isfinite(out).all()


def test_model_input_and_output_shapes():
    model = DCLCTransformerContextEncoder(
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        output_dim=128,
    )
    # Batch of 4 context sequences, each with length 9 and dimension 128
    batch_input = torch.randn(4, 9, 128)
    output = model(batch_input)
    assert output.shape == (4, 128)


def test_model_supports_numpy_and_single_sequence():
    model = DCLCTransformerContextEncoder(d_model=128, output_dim=128)
    # Single sequence as numpy array of shape (9, 128)
    np_input = np.random.randn(9, 128).astype(np.float32)
    output = model(np_input)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 128)


def test_output_embeddings_finite_and_l2_normalized():
    model = DCLCTransformerContextEncoder(
        d_model=128,
        nhead=8,
        num_layers=4,
        normalize_output=True,
    )
    model.eval()
    with torch.no_grad():
        x = torch.randn(5, 9, 128)
        emb = model(x)

    assert torch.isfinite(emb).all()

    # Verify L2 norm is ~1.0 for each embedding in batch
    norms = torch.norm(emb, p=2, dim=-1)
    expected_norms = torch.ones(5)
    torch.testing.assert_close(norms, expected_norms, atol=1e-5, rtol=1e-5)


def test_invalid_input_dimensions_and_features():
    model = DCLCTransformerContextEncoder(d_model=128)

    # 4D tensor is invalid
    with pytest.raises(ValueError, match="Expected input with 3 dimensions"):
        model(torch.randn(2, 2, 9, 128))

    # Feature dimension mismatch (64 instead of 128)
    with pytest.raises(ValueError, match="Expected feature dimension 128"):
        model(torch.randn(2, 9, 64))

    # Center index out of bounds
    with pytest.raises(IndexError, match="out of range"):
        model(torch.randn(2, 9, 128), center_idx=20)


def test_checkpoint_saving_and_loading(tmp_path: Path):
    model = DCLCTransformerContextEncoder(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        output_dim=64,
    )
    test_input = torch.randn(2, 5, 64)

    model.eval()
    with torch.no_grad():
        orig_output = model(test_input)

    # Save checkpoint
    ckpt_path = tmp_path / "test_encoder.pt"
    torch.save(model.state_dict(), ckpt_path)
    assert ckpt_path.exists()

    # Load into a new model instance
    new_model = DCLCTransformerContextEncoder(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        output_dim=64,
    )
    new_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    new_model.eval()

    with torch.no_grad():
        loaded_output = new_model(test_input)

    torch.testing.assert_close(orig_output, loaded_output)


def test_training_smoke_with_loss_and_checkpoint(tmp_path: Path):
    """End-to-end integration: model → InfoNCE loss → backward → checkpoint.

    Verifies that the full Phase 3/3.1 pipeline functions correctly:
    1. Model produces (B, 128) L2-normalized outputs from (B, 9, 128) inputs.
    2. BoundaryAwareNegativeSampler produces valid negative indices.
    3. InfoNCELoss computes a finite loss with the negatives mask.
    4. Backward pass succeeds.
    5. Checkpoint saves and reloads with identical outputs.
    """
    from src.loss import InfoNCELoss, BoundaryAwareNegativeSampler

    # Small synthetic dataset: 8 chunks from 2 files
    n_chunks = 8
    file_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    chunk_indices = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    fingerprints = [f"hash_{i}" for i in range(n_chunks)]
    initial_features = np.random.RandomState(99).randn(n_chunks, 128).astype(np.float32)
    context_sequences = np.random.RandomState(99).randn(n_chunks, 9, 128).astype(np.float32)

    model = DCLCTransformerContextEncoder(d_model=128, output_dim=128)
    loss_fn = InfoNCELoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    sampler = BoundaryAwareNegativeSampler(
        file_ids=file_ids,
        chunk_indices=chunk_indices,
        fingerprints=fingerprints,
        initial_features=initial_features,
        context_radius=4,
        proximity_threshold=0.85,
        max_negatives=15,
        seed=42,
    )

    # ── Train one step ──
    model.train()
    all_seqs = torch.from_numpy(context_sequences).float()
    anchor_emb = model(all_seqs)
    pos_emb = model(all_seqs)  # dropout augmentation (provisional)

    neg_list, mask_list = [], []
    for idx in range(n_chunks):
        neg_idx, _ = sampler.get_valid_negative_indices(idx)
        if len(neg_idx) > 0:
            neg_seqs = torch.from_numpy(context_sequences[neg_idx]).float()
            with torch.no_grad():
                neg_vecs = model(neg_seqs)
            pad_len = 15 - len(neg_idx)
            if pad_len > 0:
                neg_vecs = torch.cat([neg_vecs, torch.zeros(pad_len, 128)])
                mask = torch.cat([torch.ones(len(neg_idx), dtype=torch.bool),
                                  torch.zeros(pad_len, dtype=torch.bool)])
            else:
                neg_vecs = neg_vecs[:15]
                mask = torch.ones(15, dtype=torch.bool)
        else:
            neg_vecs = torch.zeros(15, 128)
            mask = torch.zeros(15, dtype=torch.bool)
        neg_list.append(neg_vecs)
        mask_list.append(mask)

    batch_neg = torch.stack(neg_list)
    batch_mask = torch.stack(mask_list)

    loss = loss_fn(anchor_emb, pos_emb, batch_neg, negatives_mask=batch_mask)
    assert torch.isfinite(loss), "Loss is not finite"
    assert loss.item() > 0.0, "Loss should be positive with valid negatives"

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ── Save and reload ──
    ckpt_path = tmp_path / "test_smoke.pt"
    torch.save(model.state_dict(), ckpt_path)

    model2 = DCLCTransformerContextEncoder(d_model=128, output_dim=128)
    model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model2.eval()
    model.eval()

    test_in = torch.from_numpy(context_sequences[:2]).float()
    with torch.no_grad():
        out1 = model(test_in)
        out2 = model2(test_in)

    torch.testing.assert_close(out1, out2)
    norms = torch.norm(out2, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_configurable_input_norm():
    # 1. Default is 'none'
    model_default = DCLCTransformerContextEncoder(d_model=64, output_dim=64)
    assert model_default.input_norm == "none"
    assert model_default.input_layer_norm is None

    # 2. 'layernorm' creates an nn.LayerNorm
    model_ln = DCLCTransformerContextEncoder(d_model=64, output_dim=64, input_norm="layernorm")
    assert model_ln.input_norm == "layernorm"
    assert isinstance(model_ln.input_layer_norm, torch.nn.LayerNorm)
    assert model_ln.input_layer_norm.normalized_shape == (64,)

    # 3. Invalid option raises ValueError
    with pytest.raises(ValueError, match="Unsupported input_norm"):
        DCLCTransformerContextEncoder(d_model=64, input_norm="unknown_norm")

    # 4. Forward pass with layernorm
    x = torch.randn(3, 9, 64)
    out = model_ln(x)
    assert out.shape == (3, 64)
    assert torch.isfinite(out).all()
    norms = torch.norm(out, p=2, dim=-1)
    torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=1e-5)


def test_checkpoint_compatibility_between_input_norms(tmp_path: Path):
    # Save a model with layernorm
    m_ln = DCLCTransformerContextEncoder(d_model=32, output_dim=32, input_norm="layernorm")
    ckpt_path = tmp_path / "model_ln.pt"
    torch.save(m_ln.state_dict(), ckpt_path)

    # Reload into an instance with layernorm
    m_ln_reloaded = DCLCTransformerContextEncoder(d_model=32, output_dim=32, input_norm="layernorm")
    m_ln_reloaded.load_state_dict(torch.load(ckpt_path, weights_only=True))

    x = torch.randn(2, 5, 32)
    m_ln.eval()
    m_ln_reloaded.eval()
    with torch.no_grad():
        out1 = m_ln(x)
        out2 = m_ln_reloaded(x)
    torch.testing.assert_close(out1, out2)

    # Attempting to load into input_norm='none' fails due to missing key
    m_none = DCLCTransformerContextEncoder(d_model=32, output_dim=32, input_norm="none")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        m_none.load_state_dict(torch.load(ckpt_path, weights_only=True))

