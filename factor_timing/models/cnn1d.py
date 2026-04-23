"""
=============================================================================
SCRIPT NAME: cnn1d.py
=============================================================================

DESCRIPTION:
Paper-spec 1D CNN and CNN-LSTM architectures for factor-return / trajectory
regression.  Both architectures share the same input / feature-extractor,
differing only in how they collapse the temporal dimension before the
regression head.

Paper §3.1 architecture:

  CNN:
    input (batch, timestamps, channels)  — reshape layer
    → Conv1D(filters=64, kernel=5, stride=1, padding='valid', activation=SiLU)
    → MaxPool1D(pool_size=2, strides=None)    # 'None' strides = pool_size
    → Flatten
    → Dense(64, activation=SiLU, dropout=0.25)
    → Dense(1)                                 # regression output

  CNN-LSTM:
    input → Conv1D → MaxPool1D        (same as CNN)
    → LSTM(64)                         # replaces Flatten
    → Dense(64, activation=SiLU, dropout=0.25)
    → Dense(1)

Inputs to the CNN branch are:
  (A) MXX trajectories: shape (batch, window, 1) — 1D cumulative return paths
  (B) JKX images as 1D sequences? No — the paper's CNN on JKX uses a 2D CNN.
      The 2D variant lives in cnn2d.py. This file is the 1D branch only.

Regression output is scaled by the target transform at training time (raw /
σ-standardized / Normal-inverse-CDF / percentile).  Use MSE or MAE loss.

Paper hyperparameters:
  optimizer: Adam, lr=1e-3, β1=0.9, β2=0.99, ε=1e-7
  batch size: 2**15 (32768)
  max epochs: 100, early stopping patience: 7
  train/val split: 70/30 within the training window

VERSION: 1.0
LAST UPDATED: 2026-04-22

DEPENDENCIES:
- torch (>=2.0)
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


@dataclass
class CNN1DConfig:
    window: int                         # length of the input sequence (12 or 24)
    in_channels: int = 1                # MXX branch uses 1 channel
    filters: int = 64                   # Conv1D output channels
    kernel: int = 5                     # Conv1D kernel size
    pool: int = 2                       # MaxPool1D size (stride = pool)
    fc_units: int = 64                  # dense layer width
    dropout: float = 0.25               # dropout on the pre-head dense layer
    head: Literal["regression"] = "regression"


class _FeatureExtractor(nn.Module):
    """Conv1D(64, k=5, SiLU) + MaxPool1D(2) — shared by CNN and CNN-LSTM.

    Input:  (B, T, C_in)       (batch, timestamps, channels)  — matches TF spec
    Output: (B, T', filters)   where T' = (T - kernel + 1) // pool
    """

    def __init__(self, cfg: CNN1DConfig):
        super().__init__()
        # PyTorch Conv1d expects (B, C, T); we transpose at call time
        self.conv = nn.Conv1d(
            in_channels  = cfg.in_channels,
            out_channels = cfg.filters,
            kernel_size  = cfg.kernel,
            stride       = 1,
            padding      = 0,           # 'valid' padding
        )
        self.act  = nn.SiLU()
        self.pool = nn.MaxPool1d(kernel_size=cfg.pool, stride=cfg.pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  — transpose to (B, C, T) for Conv1d
        if x.dim() == 2:                # (B, T) — single-channel shortcut
            x = x.unsqueeze(-1)          # (B, T, 1)
        x = x.transpose(1, 2)            # (B, C, T)
        x = self.conv(x)                 # (B, F, T')
        x = self.act(x)
        x = self.pool(x)                 # (B, F, T'')
        return x.transpose(1, 2)         # (B, T'', F)


class CNN1D(nn.Module):
    """Pure CNN regression head (no LSTM)."""

    def __init__(self, cfg: CNN1DConfig):
        super().__init__()
        self.cfg = cfg
        self.features = _FeatureExtractor(cfg)

        # Compute flatten size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.window, cfg.in_channels)
            feat  = self.features(dummy)           # (1, T'', F)
            flat  = feat.flatten(start_dim=1).shape[1]

        self.dropout = nn.Dropout(cfg.dropout)
        self.dense   = nn.Linear(flat, cfg.fc_units)
        self.silu    = nn.SiLU()
        self.head    = nn.Linear(cfg.fc_units, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.silu(self.dense(x))
        x = self.dropout(x)
        return self.head(x).squeeze(-1)  # (B,)


class CNNLSTM1D(nn.Module):
    """CNN + LSTM(64) regression head."""

    def __init__(self, cfg: CNN1DConfig):
        super().__init__()
        self.cfg = cfg
        self.features = _FeatureExtractor(cfg)
        self.lstm = nn.LSTM(
            input_size  = cfg.filters,
            hidden_size = cfg.fc_units,
            num_layers  = 1,
            batch_first = True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.dense   = nn.Linear(cfg.fc_units, cfg.fc_units)
        self.silu    = nn.SiLU()
        self.head    = nn.Linear(cfg.fc_units, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for n, p in m.named_parameters():
                    if "weight" in n:
                        nn.init.xavier_uniform_(p)
                    elif "bias" in n:
                        nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)             # (B, T'', F)
        out, (h, c) = self.lstm(x)       # out: (B, T'', hidden)
        last = out[:, -1, :]             # final timestep representation
        last = self.silu(self.dense(last))
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


# ---------------------------------------------------------------------------
# 2D CNN branch (JKX image input) — paper uses the same 1D conv recipe on the
# JKX image but reshaped. We treat the JKX branch as a proper 2D CNN because
# it better respects the image geometry (and aligns with the original JKX
# paper by Jiang-Kelly-Xiu 2023). The 2D architecture lives in cnn2d.py.
# ---------------------------------------------------------------------------
