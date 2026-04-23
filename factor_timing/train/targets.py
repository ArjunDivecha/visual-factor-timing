"""
=============================================================================
SCRIPT NAME: targets.py
=============================================================================

INPUT FILES:
- factor_timing/outputs/cache_{panel}_w{window}/index.parquet
    (factor, end_date, label_return)

OUTPUT FILES:
- factor_timing/outputs/cache_{panel}_w{window}/targets.parquet
    same rows as index.parquet with 4 target columns and 2 weight columns:
        raw, sigma, norm, pct, weight_ew, weight_ewpm

VERSION: 1.0
LAST UPDATED: 2026-04-22

DESCRIPTION:
Implements the paper's §3.2 four target transformations and two weighting
schemes for the 64-model ensemble.

Target transformations (per sample (j, t) with label r_{j,t+1}):
  raw   — the label itself, r_{j,t+1}
  sigma — (r_{j,t+1} − mean_j) / σ_cs(t+1)
          * mean_j is the factor-level historical mean of labels up to and
            including month t+1 (purely backward-looking; no leakage)
          * σ_cs(t+1) is the cross-sectional std of r_{·,t+1} across all
            factors valid that month
  norm  — Φ⁻¹[rank_{j,t+1} / (N_{t+1} + 1)]
          * Φ⁻¹ is the inverse standard-normal CDF
          * rank is the dense rank across all valid factors in month t+1
  pct   — rank_{j,t+1} / (N_{t+1} + 1)

Weighting schemes:
  weight_ew   — 1 for every sample (ordinary least squares / MAE)
  weight_ewpm — equal weight per month, equal weight per factor within month:
                w_{j,t} = 1 / (T × N_t)   then rescaled to sum to N_total
                so the effective "sample size" stays the same as EW.

Both transforms and weights avoid forward leakage:
- Cross-sectional statistics at month t+1 use only factors that have a
  label at t+1 (no knowledge of future months is used)
- The factor-level mean used in `sigma` is a trailing expanding mean
  computed up to and including the current label month

DEPENDENCIES:
- numpy, pandas, scipy

USAGE:
    python -m factor_timing.train.targets --panel t2 --window 12

NOTES:
- Labels with fewer than 3 valid factors in their month are dropped from
  cross-sectional transforms (cross-sectional std is unreliable).
- All columns are float32.
=============================================================================
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

log = logging.getLogger(__name__)


def _cross_sectional_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Compute σ-standardized, Φ⁻¹-rank, and percentile transforms.

    Input df must have columns [factor, end_date, label_return, raw].
    Groups by end_date and computes the transforms within each month.
    """
    df = df.copy()
    df["sigma"] = np.nan
    df["norm"]  = np.nan
    df["pct"]   = np.nan

    # Trailing expanding mean per factor (used only for sigma)
    df = df.sort_values(["factor", "end_date"]).reset_index(drop=True)
    df["_factor_mean"] = (
        df.groupby("factor")["raw"]
          .apply(lambda s: s.expanding().mean())
          .reset_index(level=0, drop=True)
          .values
    )

    for date, g in df.groupby("end_date", sort=False):
        valid = g.dropna(subset=["raw"])
        N = len(valid)
        if N < 3:
            continue
        r = valid["raw"].values
        mean_j = valid["_factor_mean"].values
        std_cs = r.std(ddof=1)
        if std_cs == 0 or not np.isfinite(std_cs):
            continue

        sigma = (r - mean_j) / std_cs

        # rank in [1..N]; use 'average' ties so tied observations share rank
        ranks = pd.Series(r).rank(method="average").values
        pct   = ranks / (N + 1)
        norm_q = _norm.ppf(pct)

        df.loc[valid.index, "sigma"] = sigma.astype(np.float32)
        df.loc[valid.index, "norm"]  = norm_q.astype(np.float32)
        df.loc[valid.index, "pct"]   = pct.astype(np.float32)

    return df.drop(columns=["_factor_mean"])


def _weighting(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weight_ew and weight_ewpm."""
    df = df.copy()
    n_total = len(df)
    # EW — uniform
    df["weight_ew"] = np.float32(1.0)
    # EWPM — equal per month, equal per factor within month, normalized so
    # that sum equals n_total (same "effective sample size" as EW)
    month_counts = df.groupby("end_date")["factor"].transform("size")
    n_months = df["end_date"].nunique()
    ewpm = (1.0 / (n_months * month_counts)).astype(float)
    ewpm = ewpm * n_total / ewpm.sum()   # rescale
    df["weight_ewpm"] = ewpm.astype(np.float32)
    return df


def build_targets(panel: str, window: int, out_root: str | Path = "factor_timing/outputs") -> Path:
    cache_dir = Path(out_root) / f"cache_{panel}_w{window}"
    idx_path = cache_dir / "index.parquet"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    idx = pd.read_parquet(idx_path)
    idx = idx.rename(columns={"label_return": "raw"})
    idx["raw"] = idx["raw"].astype(np.float32)

    log.info("Building targets for %s w%d  rows=%d", panel, window, len(idx))
    out = _cross_sectional_transforms(idx)
    out = _weighting(out)

    out_path = cache_dir / "targets.parquet"
    cols = ["factor", "end_date", "row",
            "raw", "sigma", "norm", "pct",
            "weight_ew", "weight_ewpm"]
    out[cols].to_parquet(out_path, index=False)
    log.info(
        "Wrote %s  NaN counts: sigma=%d norm=%d pct=%d",
        out_path,
        out["sigma"].isna().sum(),
        out["norm"].isna().sum(),
        out["pct"].isna().sum(),
    )
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel",  choices=["t2", "gdelt"], required=True)
    ap.add_argument("--window", type=int, required=True)
    args = ap.parse_args()
    p = build_targets(args.panel, args.window)
    df = pd.read_parquet(p)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(df[["raw", "sigma", "norm", "pct", "weight_ew", "weight_ewpm"]].describe())
