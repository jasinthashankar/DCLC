"""Controlled Experiment Runner for DCLC Input-Scale Correction (Phase 3.4).

Trains the corrected DCLC Transformer architecture (with input normalization)
on the prepared dataset and saves the checkpoint to a separate file:
    checkpoints/dclc_encoder_corrected.pt
preserving the original checkpoint checkpoints/dclc_encoder.pt untouched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.loss import BoundaryAwareNegativeSampler, InfoNCELoss
from src.model import DCLCTransformerContextEncoder


class DCLCDataset(Dataset):
    """Simple PyTorch Dataset wrapper for context sequences."""

    def __init__(self, sequences: np.ndarray):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.sequences[idx]).float(), idx


def run_training_experiment(
    dataset_path: Path | str = "data/processed/dclc_train_data.npz",
    manifest_path: Path | str = "data/processed/dataset_manifest.json",
    output_checkpoint_path: Path | str = "checkpoints/dclc_encoder_corrected.pt",
    input_norm: str = "layernorm",
    num_epochs: int = 20,
    patience: int = 5,
    lr: float = 3e-4,
    temperature: float = 0.07,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train the DCLC encoder under the specified input normalization mode.

    Args:
        dataset_path: Path to .npz data archive.
        manifest_path: Path to dataset manifest JSON.
        output_checkpoint_path: Path where the new checkpoint will be stored.
        input_norm: Input normalization mode ('none' or 'layernorm').
        num_epochs: Max epochs to train.
        patience: Early stopping patience epochs.
        lr: AdamW learning rate.
        temperature: InfoNCE temperature parameter tau.
        seed: Random seed for reproducibility.
        device: Device for training.

    Returns:
        Summary dict containing training history and metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load data
    npz_data = np.load(dataset_path)
    context_sequences = npz_data["context_sequences"].astype(np.float32)
    initial_features = npz_data["initial_features"].astype(np.float32)
    file_ids = npz_data["file_ids"].astype(np.int32)
    chunk_indices = npz_data["chunk_indices"].astype(np.int32)

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    fingerprints = [c["sha256"] for c in manifest["chunks"]]
    total_chunks = len(context_sequences)

    # 2. Build model, sampler, loss
    model = DCLCTransformerContextEncoder(
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        output_dim=128,
        normalize_output=True,
        input_norm=input_norm,
    ).to(device)

    loss_fn = InfoNCELoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    sampler = BoundaryAwareNegativeSampler(
        file_ids=file_ids,
        chunk_indices=chunk_indices,
        fingerprints=fingerprints,
        initial_features=initial_features,
        context_radius=4,
        proximity_threshold=0.85,
        max_negatives=15,
        seed=seed,
    )

    batch_size = min(256, max(4, total_chunks // 2))
    train_loader = DataLoader(
        DCLCDataset(context_sequences),
        batch_size=batch_size,
        shuffle=True,
    )

    output_path = Path(output_checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    theoretical_uniform_loss = math.log(16)
    print(f"Starting training experiment with input_norm='{input_norm}'...")
    print(f"Theoretical uniform chance loss reference ln(16): {theoretical_uniform_loss:.4f}")

    best_loss = float("inf")
    patience_counter = 0
    history = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        batches_trained = 0

        for batch_seqs, batch_indices in train_loader:
            batch_seqs = batch_seqs.to(device)
            optimizer.zero_grad()

            anchor_emb = model(batch_seqs)
            pos_emb = model(batch_seqs)  # stochastic dropout augmentation

            batch_size_cur = len(batch_indices)
            neg_embs_list = []
            masks_list = []

            for b_i in range(batch_size_cur):
                anchor_idx = int(batch_indices[b_i])
                neg_indices, _ = sampler.get_valid_negative_indices(anchor_idx)

                if len(neg_indices) > 0:
                    neg_seqs = torch.from_numpy(context_sequences[neg_indices]).float().to(device)
                    with torch.no_grad():
                        neg_vectors = model(neg_seqs)

                    if len(neg_indices) < 15:
                        pad_len = 15 - len(neg_indices)
                        pad = torch.zeros(pad_len, 128, device=device)
                        neg_vectors = torch.cat([neg_vectors, pad], dim=0)
                        mask = torch.cat([
                            torch.ones(len(neg_indices), dtype=torch.bool),
                            torch.zeros(pad_len, dtype=torch.bool),
                        ])
                    else:
                        mask = torch.ones(15, dtype=torch.bool)
                else:
                    neg_vectors = torch.zeros(15, 128, device=device)
                    mask = torch.zeros(15, dtype=torch.bool)

                neg_embs_list.append(neg_vectors)
                masks_list.append(mask)

            batch_negatives = torch.stack(neg_embs_list).to(device)
            batch_masks = torch.stack(masks_list).to(device)

            if batch_masks.sum().item() == 0:
                continue

            loss = loss_fn(anchor_emb, pos_emb, batch_negatives, negatives_mask=batch_masks)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            batches_trained += 1

        if batches_trained == 0:
            continue

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "batches": batches_trained,
        })
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss: {avg_loss:.4f} (ref ln(16)={theoretical_uniform_loss:.4f})")

        # Early stopping and model checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    print(f"Experiment training finished. Best loss: {best_loss:.4f}")
    print(f"Saved corrected checkpoint to: {output_path.resolve()}")

    return {
        "input_norm": input_norm,
        "best_loss": best_loss,
        "theoretical_uniform_loss": theoretical_uniform_loss,
        "epochs_trained": len(history),
        "history": history,
        "checkpoint_path": str(output_path),
    }


def main() -> None:
    """CLI runner for training experiment."""
    parser = argparse.ArgumentParser(description="Run DCLC training experiment with input normalization.")
    parser.add_argument("--dataset", type=str, default="data/processed/dclc_train_data.npz", help="Dataset path")
    parser.add_argument("--manifest", type=str, default="data/processed/dataset_manifest.json", help="Manifest path")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/dclc_encoder_corrected.pt", help="Checkpoint save path")
    parser.add_argument("--input-norm", type=str, default="layernorm", choices=["none", "layernorm"], help="Input normalization")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    run_training_experiment(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        output_checkpoint_path=args.output_checkpoint,
        input_norm=args.input_norm,
        num_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
