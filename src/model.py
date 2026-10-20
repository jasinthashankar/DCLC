"""Context-Aware Transformer Model for DCLC.

Implements the architecture specified in Section 3.2 (Page 8) of the DCLC paper:
  - Input: Context sequence of chunk feature vectors (e_{i-K} to e_{i+K}), shape (B, 2K+1, 128).
  - Positional Encoding: Sinusoidal positional encoding.
  - Encoder: 4 Transformer encoder layers, 8 attention heads, d_model=128, d_ff=512, GELU activation, dropout=0.1.
  - Output: L2-normalized semantic representation vector h_i in R^128 for the central chunk C_i.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as specified in Section 3.2 of the paper."""

    def __init__(self, d_model: int = 128, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as non-persistent buffer so it moves with model device
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input sequence.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class DCLCTransformerContextEncoder(nn.Module):
    """Transformer-based context encoder for DCLC resemblance detection.

    Paper specifications (Section 3.2, Page 8):
      - 4 encoder layers
      - 8 attention heads
      - Hidden dimension (d_model): 128
      - Feed-forward dimension (dim_feedforward): 512
      - Sinusoidal positional encoding
      - Activation function: GELU
      - Dropout rate: 0.1
      - Output: L2-normalized embedding h_i in R^128 for the central chunk C_i
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        output_dim: int = 128,
        normalize_output: bool = True,
        input_norm: str = "none",
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.normalize_output = normalize_output
        self.input_norm = input_norm.lower()

        if self.input_norm == "none":
            self.input_layer_norm = None
        elif self.input_norm == "layernorm":
            self.input_layer_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(
                f"Unsupported input_norm: '{input_norm}'. Expected 'none' or 'layernorm'."
            )

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # Transformer encoder layers (batch_first=True for shape (B, S, D))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Optional linear projection head before normalization (Figure 2: "Projection head")
        self.projection_head = nn.Linear(d_model, output_dim)

    def forward(
        self,
        context_sequences: torch.Tensor | np.ndarray,
        center_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass to extract semantic vector h_i for the central chunk.

        Args:
            context_sequences: Tensor or ndarray of shape (batch_size, seq_len, d_model),
                               e.g., (B, 9, 128).
            center_idx: Index of the central chunk within the sequence.
                        If None, defaults to seq_len // 2 (index 4 for seq_len=9).

        Returns:
            Normalized semantic vector h_i of shape (batch_size, output_dim).
        """
        if isinstance(context_sequences, np.ndarray):
            x = torch.from_numpy(context_sequences).float()
        else:
            x = context_sequences.float()

        if x.dim() == 2:
            # Single sequence of shape (seq_len, d_model) -> add batch dimension
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(
                f"Expected input with 3 dimensions (batch_size, seq_len, d_model), got {x.dim()}"
            )

        batch_size, seq_len, d_in = x.shape
        if d_in != self.d_model:
            raise ValueError(
                f"Expected feature dimension {self.d_model}, got {d_in}"
            )

        if center_idx is None:
            center_idx = seq_len // 2

        if not (0 <= center_idx < seq_len):
            raise IndexError(
                f"center_idx {center_idx} out of range for sequence length {seq_len}"
            )

        # Move input to same device as model parameters
        device = next(self.parameters()).device
        x = x.to(device)

        # 0. Apply input feature normalization if configured
        if self.input_layer_norm is not None:
            x = self.input_layer_norm(x)

        # 1. Add sinusoidal positional encoding
        x = self.pos_encoder(x)

        # 2. Pass through Transformer encoder layers
        encoded_seq = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # 3. Extract the central chunk's contextualized representation
        center_repr = encoded_seq[:, center_idx, :]  # (batch_size, d_model)

        # 4. Project to output dimension
        out = self.projection_head(center_repr)  # (batch_size, output_dim)

        # 5. L2 normalize final semantic embedding
        if self.normalize_output:
            out = F.normalize(out, p=2, dim=-1)

        return out
