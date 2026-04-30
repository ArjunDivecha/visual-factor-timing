"""
=============================================================================
SCRIPT NAME: monthly_dataset.py
=============================================================================

DESCRIPTION:
Wraps FactorImageDataset to yield WHOLE-MONTH batches:
each draw returns the inputs and labels for ALL valid factors in a single
calendar month, keyed by end_date.  Required for portfolio-level losses
(top-K spread, IC) which compute statistics across the cross-section
within a month.

USAGE:
    ds = FactorImageDataset(...)
    md = MonthlyDataset(ds)
    for X, y in DataLoader(md, batch_size=4, ...):
        # X: (batch_months, max_factors, ...)  zero-padded
        # y: (batch_months, max_factors)
        # mask: (batch_months, max_factors) — 1 where valid, 0 where padded

A custom collate_fn handles ragged month sizes (some months have fewer
factors than others) by zero-padding to the longest month in the batch
and emitting a binary mask.

VERSION: 1.0
LAST UPDATED: 2026-04-25
=============================================================================
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from factor_timing.train.dataset import FactorImageDataset


class MonthlyDataset(Dataset):
    """Returns one whole month per __getitem__.  Item shape:
       (X_month, y_month, end_date) where
         X_month: (n_factors_t, ...) input features for all factors in month t
         y_month: (n_factors_t,)     realized next-month returns
    """

    def __init__(self, base: FactorImageDataset):
        self.base  = base
        self.kind  = base.kind
        # Group base.meta by end_date to know which rows belong to each month
        self._groups = {
            d: g.reset_index(drop=True)
            for d, g in base.meta.groupby("end_date", sort=True)
        }
        self._dates = sorted(self._groups.keys())

    def __len__(self) -> int:
        return len(self._dates)

    def __getitem__(self, i: int):
        d = self._dates[i]
        g = self._groups[d]
        rows = g["row"].astype(int).values
        # Load all images/trajectories for this month
        if self.kind == "jkx":
            arr = np.array(self.base._arr[rows], dtype=np.float32) / 255.0
            X = torch.from_numpy(arr)             # (n, 1, H, W)
        else:
            arr = np.array(self.base._arr[rows], dtype=np.float32)
            X = torch.from_numpy(arr).unsqueeze(-1)  # (n, T, 1)
        y = torch.tensor(g["label_return"].astype(float).values, dtype=torch.float32)
        return X, y, d


def collate_months(batch):
    """Pad ragged months in a batch to (B, max_n, ...) and emit a mask."""
    Xs, ys, dates = zip(*batch)
    max_n = max(x.shape[0] for x in Xs)
    B = len(batch)
    if Xs[0].dim() == 4:           # JKX: (n, 1, H, W)
        H, W = Xs[0].shape[2], Xs[0].shape[3]
        X_pad = torch.zeros(B, max_n, 1, H, W, dtype=torch.float32)
    else:                           # MXX: (n, T, 1)
        T = Xs[0].shape[1]
        X_pad = torch.zeros(B, max_n, T, 1, dtype=torch.float32)
    y_pad = torch.zeros(B, max_n, dtype=torch.float32)
    mask  = torch.zeros(B, max_n, dtype=torch.float32)
    for i, (X, y) in enumerate(zip(Xs, ys)):
        n = X.shape[0]
        X_pad[i, :n] = X
        y_pad[i, :n] = y
        mask[i, :n] = 1.0
    return X_pad, y_pad, mask, list(dates)
