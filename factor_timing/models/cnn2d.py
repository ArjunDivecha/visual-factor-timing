"""
=============================================================================
SCRIPT NAME: cnn2d.py
=============================================================================

DESCRIPTION:
Paper-spec 2D CNN and 2D CNN-LSTM architectures for JKX image regression.

Per Figure 3 of the paper, the JKX branch uses:

  input image (1, H, W)                  # e.g. (1, 72, 36) for window=12
  → Conv2D(filters=64, kernel=(5,5), stride=1, padding='valid', activation=SiLU)
  → MaxPool2D(pool=(2,2))
  → Flatten
  → Dense(64, activation=SiLU, dropout=0.25)
  → Dense(1)                             # regression output

Paper §3.1 text describes the layer as "one-dimensional" but Figure 3 and the
reported flatten size of 73,984 on a 72×72 input confirm it is in fact a 2D
convolution (5×5 kernel) over the (H, W) image.  We follow Figure 3.

The CNN-LSTM variant collapses the W dimension with an LSTM over "time"
(columns of the image), following Murray, Xia & Xiao (2024).

Paper hyperparameters (Adam lr=1e-3, β1=0.9, β2=0.99, ε=1e-7, batch=2^15,
100 epochs max, early stop patience 7) are applied at the training loop
level, not in the model.

VERSION: 1.0
LAST UPDATED: 2026-04-22
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class CNN2DConfig:
    height: int
    width:  int
    in_channels: int = 1
    filters: int    = 64
    kernel:  Tuple[int, int] = (5, 5)
    pool:    Tuple[int, int] = (2, 2)
    fc_units: int  = 64
    dropout: float = 0.25


class CNN2D(nn.Module):
    """Figure-3 2D CNN regressor for JKX factor images."""

    def __init__(self, cfg: CNN2DConfig):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv2d(
            in_channels  = cfg.in_channels,
            out_channels = cfg.filters,
            kernel_size  = cfg.kernel,
            stride       = 1,
            padding      = 0,           # 'valid'
        )
        self.act  = nn.SiLU()
        self.pool = nn.MaxPool2d(kernel_size=cfg.pool, stride=cfg.pool)

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.in_channels, cfg.height, cfg.width)
            flat  = self.pool(self.act(self.conv(dummy))).flatten(start_dim=1).shape[1]

        self.dropout = nn.Dropout(cfg.dropout)
        self.dense   = nn.Linear(flat, cfg.fc_units)
        self.silu    = nn.SiLU()
        self.head    = nn.Linear(cfg.fc_units, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) or (B, H, W) — accept both
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(self.act(self.conv(x)))    # (B, F, H', W')
        x = x.flatten(start_dim=1)
        x = self.silu(self.dense(x))
        x = self.dropout(x)
        return self.head(x).squeeze(-1)          # (B,)


class CNNLSTM2D(nn.Module):
    """2D CNN feature extractor + LSTM over image columns for JKX images.

    Paper §3.1 describes this as replacing the flatten layer with an LSTM that
    further processes the time-series (image-column) dimension.  We apply the
    LSTM along the width axis (which corresponds to months in the JKX image).
    """

    def __init__(self, cfg: CNN2DConfig):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv2d(
            in_channels  = cfg.in_channels,
            out_channels = cfg.filters,
            kernel_size  = cfg.kernel,
            stride       = 1,
            padding      = 0,
        )
        self.act  = nn.SiLU()
        self.pool = nn.MaxPool2d(kernel_size=cfg.pool, stride=cfg.pool)

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.in_channels, cfg.height, cfg.width)
            feat  = self.pool(self.act(self.conv(dummy)))   # (1, F, H', W')
            _, F, Hp, Wp = feat.shape
            self._lstm_seq_len     = Wp          # sequence along width
            self._lstm_input_size  = F * Hp      # features per column

        self.lstm = nn.LSTM(
            input_size  = self._lstm_input_size,
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
            if isinstance(m, (nn.Conv2d, nn.Linear)):
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
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(self.act(self.conv(x)))    # (B, F, H', W')
        B, F, Hp, Wp = x.shape
        # Reshape so LSTM sees one timestep per image column:
        # (B, Wp, F*Hp)
        x = x.permute(0, 3, 1, 2).reshape(B, Wp, F * Hp)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.silu(self.dense(last))
        last = self.dropout(last)
        return self.head(last).squeeze(-1)
