"""
=============================================================================
SCRIPT NAME: dataset.py
=============================================================================

INPUT FILES:
- factor_timing/outputs/cache_{panel}_w{window}/images.npy
- factor_timing/outputs/cache_{panel}_w{window}/index.parquet
- factor_timing/outputs/cache_{panel}_w{window}/targets.parquet
- factor_timing/outputs/cache_{panel}_w{window}_mxx/trajectories.npy
- factor_timing/outputs/cache_{panel}_w{window}_mxx/index.parquet

OUTPUT:
- torch Dataset(s) that serve (X, y, w, meta) batches for training

VERSION: 1.0
LAST UPDATED: 2026-04-22

DESCRIPTION:
Thin torch Dataset wrappers over the JKX memmap and MXX trajectory caches.
Both caches are indexed 1:1 by (factor, end_date) so the `row` column of
each index parquet aligns with the underlying array row.

The dataset yields (X, y, w) where:
  X is either a JKX image tensor (B, 1, H, W) or an MXX trajectory (B, T, 1)
  y is the target value selected by `target_col` (raw / sigma / norm / pct)
  w is the sample weight (weight_ew or weight_ewpm)

A `date_filter` closure restricts the dataset to a train/val/test split.
An optional `factor_filter` restricts to a cluster (for GDELT pooled runs)
or a single factor (for T2 per-factor runs).

DEPENDENCIES:
- numpy, pandas, torch
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ImageKind = Literal["jkx", "mxx"]
TargetCol = Literal["raw", "sigma", "norm", "pct"]
WeightCol = Literal["weight_ew", "weight_ewpm"]


class FactorImageDataset(Dataset):
    """Serves (X, y, w) tuples from the precomputed caches."""

    def __init__(
        self,
        panel:      Literal["t2", "gdelt"],
        window:     int,
        kind:       ImageKind,
        target_col: TargetCol = "raw",
        weight_col: WeightCol = "weight_ew",
        factor_filter: Optional[set] = None,
        date_filter:   Optional[Callable[[pd.Timestamp], bool]] = None,
        out_root: str | Path = "factor_timing/outputs",
        x_dtype: torch.dtype = torch.float32,
    ):
        self.panel = panel
        self.window = window
        self.kind = kind
        self.target_col = target_col
        self.weight_col = weight_col
        self.x_dtype = x_dtype

        cache_dir = Path(out_root) / f"cache_{panel}_w{window}"
        mxx_dir   = Path(out_root) / f"cache_{panel}_w{window}_mxx"
        tgt_path  = cache_dir / "targets.parquet"

        idx  = pd.read_parquet(cache_dir / "index.parquet")
        tgts = pd.read_parquet(tgt_path)
        # `index.parquet` and `targets.parquet` share factor, end_date, row
        self.meta = idx.merge(
            tgts[["factor", "end_date", target_col, weight_col]],
            on=["factor", "end_date"],
            how="inner",
        )

        # Apply filters
        if factor_filter is not None:
            self.meta = self.meta[self.meta["factor"].isin(factor_filter)]
        if date_filter is not None:
            mask = self.meta["end_date"].apply(date_filter)
            self.meta = self.meta[mask]
        # Drop NaN targets (sigma/norm/pct may have NaN when cross-section too thin)
        self.meta = self.meta.dropna(subset=[target_col]).reset_index(drop=True)

        # Load the underlying array as memmap
        if kind == "jkx":
            self._arr = np.load(cache_dir / "images.npy", mmap_mode="r")   # (N, 1, H, W) uint8
        elif kind == "mxx":
            self._arr = np.load(mxx_dir / "trajectories.npy", mmap_mode="r")  # (N, T) float32
        else:
            raise ValueError(kind)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        r   = int(row["row"])
        if self.kind == "jkx":
            arr = np.array(self._arr[r], dtype=np.float32) / 255.0
            x = torch.from_numpy(arr)
            # shape (1, H, W) — already correct
        else:
            arr = np.array(self._arr[r], dtype=np.float32)
            x = torch.from_numpy(arr)
            # shape (T,) — add channel dim
            x = x.unsqueeze(-1)  # (T, 1) — CNN1D expects (B, T, C)
        y = torch.tensor(row[self.target_col], dtype=torch.float32)
        w = torch.tensor(row[self.weight_col], dtype=torch.float32)
        return x.to(self.x_dtype), y, w


def split_dates(
    dates: pd.Series,
    train_end: pd.Timestamp,
    val_end:   pd.Timestamp,
    test_end:  Optional[pd.Timestamp] = None,
):
    """Convenience: produce boolean masks for train / val / test by date."""
    train_mask = dates <= train_end
    val_mask   = (dates > train_end) & (dates <= val_end)
    if test_end is None:
        test_mask = dates > val_end
    else:
        test_mask = (dates > val_end) & (dates <= test_end)
    return train_mask, val_mask, test_mask
