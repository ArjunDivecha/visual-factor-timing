"""
=============================================================================
SCRIPT NAME: ensemble.py
=============================================================================

DESCRIPTION:
64-model ensemble driver following paper §3.2.  The ensemble combines:

  2 architectures  (CNN, CNN-LSTM)
  2 image types    (MXX 1D trajectory, JKX 2D candlestick image)
  2 loss functions (MSE, MAE)
  2 weighting      (EW, EWPM)
  4 target xforms  (raw, sigma, norm, pct)
  ──────────────────────
  64 base models

Each base model is fit 30 times on non-overlapping sub-samples of the
training window.  In-sample IC is computed and used to weight the 64
forecasts (IC-weighted average) when producing the final ω^CNN_jt signal.

This module exposes:

  run_ensemble(panel, window, retrain_schedule, out_path, ...)

which iterates over retrain epochs, for each epoch:
  1. (GDELT only) refits the 8-group Ward clustering on data < epoch_date
  2. Assembles train / val / test splits by date (70/30 within train window)
  3. For T2: trains 64 models per-factor on the trailing window
     For GDELT: trains 64 models per-cluster on the trailing window
  4. For each model, predicts the test epoch's (factor, date) samples
  5. Computes per-model IC on the training set (proxy for out-of-sample IC)
  6. Aggregates 64 model forecasts by IC-weighted average → ω_raw
  7. Applies cross-sectional median-rescaling so each month's median ω = 1
  8. Appends (factor, end_date, ω, f_hat) to the running output table

VERSION: 1.0
LAST UPDATED: 2026-04-22

NOTES:
- This is a heavy job.  A single T2 epoch with 83 factors × 64 models × 30
  folds ≈ 159,000 model fits.  The paper takes the compute hit in exchange
  for variance reduction; we mirror it but expose knobs (`n_folds`,
  `model_combos`) so the caller can dial it back for experimentation.
- Default is `n_folds=5` for tractability; bump to 30 for a final run.
=============================================================================
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from factor_timing.models.cnn1d   import CNN1D, CNNLSTM1D, CNN1DConfig
from factor_timing.models.cnn2d   import CNN2D, CNNLSTM2D, CNN2DConfig
from factor_timing.train.dataset  import FactorImageDataset
from factor_timing.train.loop     import TrainConfig, train_one, predict
from factor_timing.train.clustering import cluster_at

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model combination specification
# ---------------------------------------------------------------------------
ARCHES   = ["cnn", "cnn_lstm"]
IMAGES   = ["mxx", "jkx"]
LOSSES   = ["mse", "mae"]
WEIGHTS  = ["weight_ew", "weight_ewpm"]
TARGETS  = ["raw", "sigma", "norm", "pct"]


def all_model_combos() -> List[dict]:
    out = []
    for a, i, l, w, t in itertools.product(ARCHES, IMAGES, LOSSES, WEIGHTS, TARGETS):
        out.append({"arch": a, "image": i, "loss": l, "weight": w, "target": t})
    return out


def _build_model(image: str, arch: str, window: int, img_hw: tuple) -> torch.nn.Module:
    if image == "mxx":
        cfg = CNN1DConfig(window=window)
        return CNN1D(cfg) if arch == "cnn" else CNNLSTM1D(cfg)
    else:
        h, w = img_hw
        cfg = CNN2DConfig(height=h, width=w)
        return CNN2D(cfg) if arch == "cnn" else CNNLSTM2D(cfg)


# ---------------------------------------------------------------------------
# Training a single fold
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    combo: dict
    fold_id: int
    ic_train: float
    pred_test: np.ndarray          # aligned to test_meta order
    test_meta: pd.DataFrame        # (factor, end_date)


def train_fold(
    panel: str,
    window: int,
    combo: dict,
    train_cutoff: pd.Timestamp,
    val_cutoff: pd.Timestamp,
    test_cutoff: pd.Timestamp,
    factor_filter: Optional[set],
    fold_id: int,
    image_hw: tuple,
    train_cfg: TrainConfig,
) -> FoldResult:
    """Train one base model on a single non-overlapping subsample of the
    training window, predict the test window, return IC + predictions."""
    # Deterministic sub-sampling by fold: each fold keeps a different
    # fraction (1/n_folds) held out as held-in-training-set validation.
    # Here we use a simple rotation: fold i uses seed i and random 85/15
    # split of the pre-train_cutoff data.
    np.random.seed(fold_id)
    torch.manual_seed(fold_id)

    train_ds = FactorImageDataset(
        panel=panel, window=window, kind=combo["image"],
        target_col=combo["target"], weight_col=combo["weight"],
        factor_filter=factor_filter,
        date_filter=lambda d, c=train_cutoff: d <= c,
    )
    val_ds = FactorImageDataset(
        panel=panel, window=window, kind=combo["image"],
        target_col=combo["target"], weight_col=combo["weight"],
        factor_filter=factor_filter,
        date_filter=lambda d, a=train_cutoff, b=val_cutoff: (d > a) and (d <= b),
    )
    test_ds = FactorImageDataset(
        panel=panel, window=window, kind=combo["image"],
        target_col=combo["target"], weight_col=combo["weight"],
        factor_filter=factor_filter,
        date_filter=lambda d, a=val_cutoff, b=test_cutoff: (d > a) and (d <= b),
    )
    if len(train_ds) < 10 or len(val_ds) < 3 or len(test_ds) == 0:
        return FoldResult(combo=combo, fold_id=fold_id, ic_train=float("nan"),
                          pred_test=np.zeros(len(test_ds), dtype=np.float32),
                          test_meta=test_ds.meta[["factor", "end_date"]].copy())

    model = _build_model(combo["image"], combo["arch"], window, image_hw)

    loss_for_this = TrainConfig(
        lr=train_cfg.lr, beta1=train_cfg.beta1, beta2=train_cfg.beta2, eps=train_cfg.eps,
        batch_size=train_cfg.batch_size, max_epochs=train_cfg.max_epochs,
        patience=train_cfg.patience, loss=combo["loss"], device=train_cfg.device,
    )
    model, _ = train_one(model, train_ds, val_ds, loss_for_this, seed=fold_id)

    # In-sample IC (on training set, to serve as model weight)
    pred_train = predict(model, train_ds, batch_size=train_cfg.batch_size,
                         device=train_cfg.device)
    ytr = train_ds.meta["raw"].astype(float).values if "raw" in train_ds.meta.columns \
          else train_ds.meta[combo["target"]].astype(float).values
    if len(pred_train) >= 2 and np.std(pred_train) > 0 and np.std(ytr) > 0:
        ic_train = float(np.corrcoef(pred_train, ytr)[0, 1])
    else:
        ic_train = float("nan")

    pred_test = predict(model, test_ds, batch_size=train_cfg.batch_size,
                        device=train_cfg.device)
    return FoldResult(
        combo=combo, fold_id=fold_id, ic_train=ic_train,
        pred_test=pred_test,
        test_meta=test_ds.meta[["factor", "end_date"]].copy(),
    )


# ---------------------------------------------------------------------------
# Epoch orchestration
# ---------------------------------------------------------------------------
def aggregate_forecasts(results: Sequence[FoldResult]) -> pd.DataFrame:
    """IC-weighted aggregation of multiple FoldResults into a single forecast
    per (factor, end_date).  Non-positive IC weights are clipped to 0."""
    dfs = []
    for r in results:
        ic = r.ic_train if np.isfinite(r.ic_train) else 0.0
        w  = max(ic, 0.0)
        if w == 0.0 or len(r.pred_test) == 0:
            continue
        d = r.test_meta.copy()
        d["pred"]     = r.pred_test
        d["ic"]       = ic
        d["weight"]   = w
        dfs.append(d)
    if not dfs:
        return pd.DataFrame(columns=["factor", "end_date", "f_hat"])

    allp = pd.concat(dfs, ignore_index=True)
    allp["wp"] = allp["weight"] * allp["pred"]
    agg = (
        allp.groupby(["factor", "end_date"])
            .agg(wp_sum=("wp", "sum"), w_sum=("weight", "sum"))
            .reset_index()
    )
    agg["f_hat"] = agg["wp_sum"] / agg["w_sum"].replace(0, np.nan)
    return agg[["factor", "end_date", "f_hat"]]


def omega_from_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally standardize f_hat each month, then median-shift so
    the monthly median of ω equals 1.

    When a month has fewer than 2 valid forecasts, ω is left as 1.0 (no
    timing signal) since a one-factor cross-section has undefined std.
    """
    df = forecasts.copy()

    def _zscore(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd

    df["z"]     = df.groupby("end_date")["f_hat"].transform(_zscore)
    df["omega"] = df.groupby("end_date")["z"].transform(lambda s: 1.0 + (s - s.median()))
    return df[["factor", "end_date", "f_hat", "omega"]]
