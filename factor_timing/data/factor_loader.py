"""
=============================================================================
SCRIPT NAME: factor_loader.py
=============================================================================

INPUT FILES:
- Daily factor return xlsx (T2 or GDELT) - sheet 'Monthly_Net_Returns'
    format: Date column + one column per factor, daily returns in decimal form
- Monthly factor return xlsx (same factor columns) - sheet 'Monthly_Net_Returns'
    format: Date column + one column per factor, monthly net returns

OUTPUT FILES:
- {panel}_monthly_ohlc.parquet: one row per (Factor, MonthEnd) with columns:
    Ticker, Date, Open, High, Low, Close, Volume, AdjClose, Return, RetCount
  where
    Ticker = factor name
    Date = last trading day of month
    Open = synthetic price on first trading day of month
    High = max synthetic price within month
    Low  = min synthetic price within month
    Close = synthetic price on last trading day of month
    Volume = monthly realized vol (std of non-zero daily returns within month)
    AdjClose = synthetic total-return index (same as Close, unscaled)
    Return = next-month forward return pulled from monthly file (label)
    RetCount = number of non-zero daily returns in the month

VERSION: 1.0
LAST UPDATED: 2026-04-22
AUTHOR: factor-timing replication

DESCRIPTION:
Implements the paper's Section 2.2.2 step (i)-(ii): for each factor, cumulate
daily returns into a synthetic $1 price series and extract monthly OHLC bars
+ monthly realized volatility (replacing the volume strip used in the original
JKX image, since factor long-short portfolios have no natural volume).

Non-trading days (weekends, holidays) appear in the daily file as zero-return
rows across all factors. These rows are filtered before computing vol so the
estimator isn't biased down by structural zeros. They are kept in the synthetic
price series (they just carry the previous price forward).

DEPENDENCIES:
- pandas, numpy, openpyxl, pyarrow

USAGE:
    from factor_timing.data.factor_loader import build_monthly_ohlc
    df = build_monthly_ohlc(
        daily_path="/path/to/daily/T2_Optimizer.xlsx",
        monthly_path="/path/to/monthly/T2_Optimizer.xlsx",
        out_path="factor_timing/outputs/t2_monthly_ohlc.parquet",
    )

NOTES:
- Factor names must match exactly between daily and monthly files. The loader
  asserts this and emits the intersection.
- The first month of each factor may have incomplete data (starts mid-month);
  it is kept but RetCount flags its sample count.
- Sheet name is hardcoded to 'Monthly_Net_Returns' because both daily and
  monthly optimizer files use that (misleading) sheet name.
=============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SHEET_NAME = "Monthly_Net_Returns"


def _load_returns(path: str | Path, sheet: str = SHEET_NAME) -> pd.DataFrame:
    """Load a factor-return xlsx into a long-ish DataFrame indexed by Date.

    Returns a DataFrame with a DatetimeIndex (named 'Date') and one column per
    factor. All values are numeric (NaN preserved).
    """
    df = pd.read_excel(path, sheet_name=sheet)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "Date"
    # Coerce factor columns to float (any stray strings become NaN)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _synthetic_price(daily_returns: pd.Series, scale: float = 100.0) -> pd.Series:
    """Cumulate daily returns into a synthetic $1-indexed price series.

    The T2 and GDELT optimizer files store returns in PERCENT (e.g. 1.23 for
    +1.23%), so we divide by `scale` (default 100) before compounding. NaNs
    are treated as 0 (carry forward). First observation anchors at 1.0.
    """
    r = daily_returns.fillna(0.0).astype(float) / float(scale)
    p = (1.0 + r).cumprod()
    return p


def _monthly_ohlc_for_factor(
    daily_prices: pd.Series,
    daily_returns: pd.Series,
) -> pd.DataFrame:
    """Collapse a factor's daily synthetic-price series into monthly OHLC + vol.

    Only non-zero daily returns are used for the vol estimator (structural
    weekends/holidays drop out). All daily rows are used for OHLC so the
    first/last/min/max of the calendar month are correctly identified.
    """
    # Group key = end-of-month timestamp (calendar month, not business)
    grouper = pd.Grouper(freq="ME")

    price_groups = daily_prices.groupby(grouper)
    # Calendar month: Open = first obs, Close = last obs, High = max, Low = min
    ohlc = pd.DataFrame({
        "Open":  price_groups.first(),
        "High":  price_groups.max(),
        "Low":   price_groups.min(),
        "Close": price_groups.last(),
    })

    # Realized vol: std of non-zero daily returns within each month.
    # Returned in the same percent units as the input (matches the synthetic
    # price scale when interpreting image features).
    def _month_vol(x: pd.Series) -> float:
        mask = x.fillna(0.0) != 0.0
        if mask.sum() < 2:
            return np.nan
        return float(x[mask].std(ddof=1))

    def _month_count(x: pd.Series) -> int:
        return int((x.fillna(0.0) != 0.0).sum())

    vol    = daily_returns.groupby(grouper).apply(_month_vol)
    counts = daily_returns.groupby(grouper).apply(_month_count)

    ohlc["Volume"]   = vol.reindex(ohlc.index).values
    ohlc["AdjClose"] = ohlc["Close"].values
    ohlc["RetCount"] = counts.reindex(ohlc.index).astype("Int64").values

    # Drop months with fewer than 2 trading days
    ohlc = ohlc[ohlc["RetCount"].fillna(0).astype(int) >= 2]
    return ohlc


def build_monthly_ohlc(
    daily_path: str | Path,
    monthly_path: str | Path,
    out_path: Optional[str | Path] = None,
    factors: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Core entry point. See module docstring for contract.

    Args:
        daily_path: path to daily xlsx (T2 or GDELT _Optimizer.xlsx)
        monthly_path: path to monthly xlsx with the label returns
        out_path: if given, write parquet to this path
        factors: restrict to this subset (default = intersection of daily ∩ monthly)

    Returns:
        Long DataFrame (Ticker, Date, Open, High, Low, Close, Volume,
        AdjClose, Return, RetCount) sorted by (Ticker, Date).
    """
    log.info("Loading daily returns from %s", daily_path)
    daily = _load_returns(daily_path)
    log.info("  shape=%s  range=%s..%s", daily.shape, daily.index.min(), daily.index.max())

    log.info("Loading monthly returns from %s", monthly_path)
    monthly = _load_returns(monthly_path)
    log.info("  shape=%s  range=%s..%s", monthly.shape, monthly.index.min(), monthly.index.max())

    # Factor intersection
    daily_facs   = set(daily.columns)
    monthly_facs = set(monthly.columns)
    common = sorted(daily_facs & monthly_facs)
    if factors is not None:
        common = [f for f in factors if f in common]
    log.info("Using %d factors (daily∩monthly)", len(common))
    if len(daily_facs ^ monthly_facs) > 0:
        log.warning("  daily-only: %d, monthly-only: %d, will be dropped",
                    len(daily_facs - monthly_facs), len(monthly_facs - daily_facs))

    # Month-end index for the monthly label file (align to ME frequency)
    monthly_me = monthly.copy()
    monthly_me.index = monthly_me.index.to_period("M").to_timestamp("M").normalize()

    rows = []
    for fac in common:
        dly_ret = daily[fac].dropna()
        if len(dly_ret) < 20:
            log.warning("  %s: only %d daily obs — skipping", fac, len(dly_ret))
            continue
        dly_px = _synthetic_price(dly_ret)
        ohlc = _monthly_ohlc_for_factor(dly_px, dly_ret)
        ohlc.index = ohlc.index.normalize()

        # Attach next-month label from monthly file
        # Align: ohlc index is month-end t; label = monthly[fac] at month-end t+1
        if fac in monthly_me.columns:
            lbl = monthly_me[fac].copy()
            lbl.index = lbl.index.normalize()
            next_ret = lbl.shift(-1).reindex(ohlc.index)
        else:
            next_ret = pd.Series(np.nan, index=ohlc.index)
        ohlc["Return"] = next_ret.values

        ohlc = ohlc.reset_index().rename(columns={"Date": "Date"})
        ohlc.insert(0, "Ticker", fac)
        rows.append(ohlc)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Reorder columns
    cols = ["Ticker", "Date", "Open", "High", "Low", "Close",
            "Volume", "AdjClose", "Return", "RetCount"]
    out = out[cols]

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        log.info("Wrote %s  rows=%d  factors=%d", out_path, len(out), out["Ticker"].nunique())

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser(description="Build monthly OHLC+vol cache from daily factor returns")
    ap.add_argument("--panel", choices=["t2", "gdelt"], required=True)
    ap.add_argument("--daily",   default=None, help="override daily xlsx path")
    ap.add_argument("--monthly", default=None, help="override monthly xlsx path")
    ap.add_argument("--out",     default=None, help="override output parquet path")
    args = ap.parse_args()

    defaults = {
        "t2": dict(
            daily   = "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy Daily/T2_Optimizer.xlsx",
            monthly = "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy/T2_Optimizer.xlsx",
            out     = "/Users/arjundivecha/Dropbox/AAA Backup/A Working/pattern 2/factor_timing/outputs/t2_monthly_ohlc.parquet",
        ),
        "gdelt": dict(
            daily   = "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/GDELT Factor Timing Fuzzy Daily/T2-Factor-Timing-Daily/GDELT_Optimizer.xlsx",
            monthly = "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_Optimizer.xlsx",
            out     = "/Users/arjundivecha/Dropbox/AAA Backup/A Working/pattern 2/factor_timing/outputs/gdelt_monthly_ohlc.parquet",
        ),
    }
    cfg = defaults[args.panel]
    daily   = args.daily   or cfg["daily"]
    monthly = args.monthly or cfg["monthly"]
    out     = args.out     or cfg["out"]

    df = build_monthly_ohlc(daily, monthly, out)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Factors: {df['Ticker'].nunique()}")
    print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"Return NaN: {df['Return'].isna().sum()} / {len(df)}  "
          f"(last month per factor is expected to be NaN)")
