"""
=============================================================================
SCRIPT NAME: build_cache.py
=============================================================================

INPUT FILES:
- factor_timing/outputs/{panel}_monthly_ohlc.parquet
    (produced by factor_timing.data.factor_loader)

OUTPUT FILES:
- factor_timing/outputs/cache_{panel}_w{window}/images.npy
    memmap uint8 (N, 1, H, W) containing all valid JKX images
- factor_timing/outputs/cache_{panel}_w{window}/index.parquet
    (factor, end_date, label_return, row) — row indexes into images.npy

VERSION: 1.0
LAST UPDATED: 2026-04-22

DESCRIPTION:
Iterates over every (factor, month-end t) in the OHLC parquet and renders the
JKX-style image for windows of size `window` months (12 or 24). The image
shape is (H, W) = (height, 3*window). Samples with insufficient history or
missing label are skipped. Cluster assignments are NOT baked into the cache —
they are applied lazily at train time by the expanding-window clusterer so
no look-ahead leakage occurs.

The label column `label_return` is the next-month factor return pulled from
the monthly optimizer file during the OHLC build step. For the final month
per factor the label is NaN and the row is dropped here.

DEPENDENCIES:
- numpy, pandas, pyarrow, PIL (optional for preview)
- factor_timing.data.factor_loader (upstream)
- Pattern's pattern.imaging.renderer (JKX renderer)

USAGE:
    python -m factor_timing.imaging.build_cache \
        --panel t2 --window 12
    python -m factor_timing.imaging.build_cache \
        --panel gdelt --window 24

NOTES:
- The renderer's `include_volume=True` path treats the "Volume" column as the
  bottom-strip height, which in our OHLC parquet is monthly realized
  volatility (std of non-zero daily returns within the month).
- Height is fixed at 72 px, ratio 0.80. Width = 3*window.
- All images are uint8 {0, 255}.
=============================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Pattern renderer is our JKX renderer (already validated in earlier steps)
sys.path.insert(0, "/Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern")
from pattern.imaging.renderer import render_window

log = logging.getLogger(__name__)

OHLC_PARQUET = {
    "t2":    "factor_timing/outputs/t2_monthly_ohlc.parquet",
    "gdelt": "factor_timing/outputs/gdelt_monthly_ohlc.parquet",
}

HEIGHT           = 72
OHLC_HEIGHT_RATIO = 0.80


def build_cache(
    panel: str,
    window: int,
    out_root: str | Path = "factor_timing/outputs",
    height: int = HEIGHT,
    ohlc_height_ratio: float = OHLC_HEIGHT_RATIO,
) -> dict:
    """Render all JKX images for a panel/window and write memmap + index."""
    assert panel in OHLC_PARQUET, panel
    assert window >= 6, f"window must be at least 6, got {window}"

    ohlc_path = Path(OHLC_PARQUET[panel])
    if not ohlc_path.exists():
        raise FileNotFoundError(f"Missing OHLC parquet: {ohlc_path}")

    out_dir = Path(out_root) / f"cache_{panel}_w{window}"
    out_dir.mkdir(parents=True, exist_ok=True)

    width = 3 * window
    log.info("Building cache %s panel=%s window=%d H=%d W=%d", out_dir, panel, window, height, width)

    df = pd.read_parquet(ohlc_path)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Pre-allocate upper-bound array for memmap (N_samples ≤ rows-in-panel)
    # We'll slice it to the true count after the loop.
    n_upper = len(df)
    images_path = out_dir / "images.npy"
    # use lock-free uint8 memmap
    images_mm = np.lib.format.open_memmap(
        images_path, mode="w+", dtype=np.uint8, shape=(n_upper, 1, height, width),
    )

    index_rows = []
    row_cursor = 0
    n_skipped_history = 0
    n_skipped_label   = 0
    n_skipped_render  = 0

    for tkr, g in df.groupby("Ticker", sort=False):
        g = g.reset_index(drop=True)
        ohlcv_all = g[["Open", "High", "Low", "Close", "Volume"]].values.astype(float)
        tri_all   = g["AdjClose"].values.astype(float)
        ret_all   = g["Return"].values.astype(float)
        dates_all = g["Date"].values

        for i in range(len(g)):
            # Need `window` rows ending at row i (inclusive)
            if i + 1 < window:
                n_skipped_history += 1
                continue
            # Need next-month label
            if np.isnan(ret_all[i]):
                n_skipped_label += 1
                continue

            ohlcv = ohlcv_all[i + 1 - window : i + 1]
            tri   = tri_all[i + 1 - window : i + 1]

            img = render_window(
                ohlcv=ohlcv, tri=tri,
                window=window, height=height, width=width,
                ohlc_height_ratio=ohlc_height_ratio,
                include_ma=True, include_volume=True,
            )
            if img is None:
                n_skipped_render += 1
                continue

            images_mm[row_cursor] = img
            index_rows.append({
                "factor":       tkr,
                "end_date":     pd.Timestamp(dates_all[i]),
                "label_return": float(ret_all[i]),
                "row":          row_cursor,
            })
            row_cursor += 1

        if row_cursor > 0 and row_cursor % 2000 == 0:
            log.info("  rendered %d images so far", row_cursor)

    # Truncate memmap to actual count
    images_mm.flush()
    del images_mm
    # Re-open and rewrite at exact size (we can't truncate a memmap in place cleanly)
    full = np.load(images_path, mmap_mode="r")
    actual = full[:row_cursor].copy()
    np.save(images_path, actual)
    log.info("Saved %s  shape=%s", images_path, actual.shape)

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
    log.info(
        "Skipped: history=%d  no-label=%d  render-fail=%d",
        n_skipped_history, n_skipped_label, n_skipped_render,
    )

    return {
        "images_path": str(images_path),
        "index_path":  str(index_path),
        "n_images":    row_cursor,
        "n_factors":   index_df["factor"].nunique(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel",  choices=["t2", "gdelt"], required=True)
    ap.add_argument("--window", type=int, required=True)
    args = ap.parse_args()
    res = build_cache(args.panel, args.window)
    print(res)
