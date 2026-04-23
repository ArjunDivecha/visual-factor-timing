"""
=============================================================================
SCRIPT NAME: build_mxx_cache.py
=============================================================================

INPUT FILES:
- factor_timing/outputs/{panel}_monthly_ohlc.parquet
    (produced by factor_timing.data.factor_loader)

OUTPUT FILES:
- factor_timing/outputs/cache_{panel}_w{window}_mxx/trajectories.npy
    float32 array (N, window) — each row is CFR_{j,1..window} for one sample
- factor_timing/outputs/cache_{panel}_w{window}_mxx/index.parquet
    (factor, end_date, label_return, row) — row aligns with the float array

VERSION: 1.0
LAST UPDATED: 2026-04-22

DESCRIPTION:
Builds the MXX branch input cache. Per paper §2.1, for each sample
(factor j, month-end t) the MXX input is the trajectory
    CFR_{j,k} = cumulative return from month (t-window) through (t-window+k-1)
             = Close[t-window+k-1] / Close[t-window] - 1        for k = 1..window

In other words, the synthetic-price path is rebased to the window's first
month, converted to a cumulative-return-from-anchor form, and fed as a 1D
sequence into the 1D CNN / CNN-LSTM branch with kernel 5. Figure 1 of the
paper visualizes this as a bar chart but the network consumes the raw
sequence — we skip the bar-chart render since the CNN sees the values
directly and the rendered bars add no information the conv filters can't
already compress.

The sample index matches the JKX cache 1:1 so the two branches can be
concatenated (or ensembled) at the signal-combination step. Samples needing
>window months of history or missing the next-month label are dropped.

DEPENDENCIES:
- numpy, pandas, pyarrow

USAGE:
    python -m factor_timing.imaging.build_mxx_cache --panel t2 --window 12
    python -m factor_timing.imaging.build_mxx_cache --panel gdelt --window 24

NOTES:
- Values are float32 (unlike JKX which is uint8) — the MXX CNN operates on
  continuous values, not rasterised pixels. Cross-sectional standardization
  is applied lazily at train time (not baked into the cache) so different
  fold/retrain configurations can reuse the same cache.
=============================================================================
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OHLC_PARQUET = {
    "t2":    "factor_timing/outputs/t2_monthly_ohlc.parquet",
    "gdelt": "factor_timing/outputs/gdelt_monthly_ohlc.parquet",
}


def build_mxx_cache(
    panel: str,
    window: int,
    out_root: str | Path = "factor_timing/outputs",
) -> dict:
    assert panel in OHLC_PARQUET, panel
    assert window >= 6, f"window must be at least 6, got {window}"

    ohlc_path = Path(OHLC_PARQUET[panel])
    if not ohlc_path.exists():
        raise FileNotFoundError(ohlc_path)

    out_dir = Path(out_root) / f"cache_{panel}_w{window}_mxx"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ohlc_path).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Pre-allocate upper bound
    n_upper = len(df)
    traj_path = out_dir / "trajectories.npy"
    traj_mm = np.lib.format.open_memmap(
        traj_path, mode="w+", dtype=np.float32, shape=(n_upper, window),
    )

    index_rows = []
    row_cursor = 0
    n_hist = n_lbl = n_anchor = 0

    for tkr, g in df.groupby("Ticker", sort=False):
        g = g.reset_index(drop=True)
        close = g["AdjClose"].values.astype(float)
        ret   = g["Return"].values.astype(float)
        dates = g["Date"].values

        for i in range(len(g)):
            if i + 1 < window:
                n_hist += 1
                continue
            if np.isnan(ret[i]):
                n_lbl += 1
                continue
            anchor = close[i + 1 - window]
            if not np.isfinite(anchor) or anchor <= 0:
                n_anchor += 1
                continue
            # CFR_k = close[anchor_idx + k - 1] / anchor - 1, for k=1..window
            window_closes = close[i + 1 - window : i + 1]
            cfr = (window_closes / anchor) - 1.0
            if not np.all(np.isfinite(cfr)):
                n_anchor += 1
                continue

            traj_mm[row_cursor] = cfr.astype(np.float32)
            index_rows.append({
                "factor":       tkr,
                "end_date":     pd.Timestamp(dates[i]),
                "label_return": float(ret[i]),
                "row":          row_cursor,
            })
            row_cursor += 1

    traj_mm.flush()
    del traj_mm

    # Truncate to exact count
    full = np.load(traj_path, mmap_mode="r")
    actual = full[:row_cursor].copy()
    np.save(traj_path, actual)
    log.info("Saved %s  shape=%s", traj_path, actual.shape)

    index_df = pd.DataFrame(index_rows)
    index_path = out_dir / "index.parquet"
    index_df.to_parquet(index_path, index=False)
    log.info(
        "Saved %s  rows=%d  factors=%d  dates=%s..%s",
        index_path, len(index_df),
        index_df["factor"].nunique(),
        str(index_df["end_date"].min().date()),
        str(index_df["end_date"].max().date()),
    )
    log.info("Skipped: history=%d  no-label=%d  anchor-issue=%d", n_hist, n_lbl, n_anchor)

    return {
        "trajectories_path": str(traj_path),
        "index_path":        str(index_path),
        "n_samples":         row_cursor,
        "n_factors":         index_df["factor"].nunique(),
        "shape":             tuple(actual.shape),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel",  choices=["t2", "gdelt"], required=True)
    ap.add_argument("--window", type=int, required=True)
    args = ap.parse_args()
    res = build_mxx_cache(args.panel, args.window)
    print(res)
