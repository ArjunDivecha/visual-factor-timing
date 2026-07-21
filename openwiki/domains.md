---
type: "Reference"
title: "Domain concepts"
openwiki_generated: true
---

# Domain concepts

This repository uses a small set of recurring domain objects. Understanding them makes the rest of the code much easier to navigate.

## Panels

The main panels are **T2** and **GDELT**. The README also mentions ECON as a results panel, but the core source path in this repository is centered on T2 and GDELT.

Each panel is represented by external monthly and daily optimizer workbooks. `factor_loader.py` consumes those workbooks and creates a normalized monthly OHLC representation.

## Factors

A factor is a named time series column in the optimizer workbooks. Many scripts operate on one `(factor, end_date)` sample at a time.

The code distinguishes between:

- **Per-factor training** — the T2 convention in `run_single.py`.
- **Pooled clustered training** — the GDELT convention, where factors may be grouped by expanding-window Ward clustering in `clustering.py`.

## Monthly OHLC and labels

`factor_loader.py` constructs a synthetic price series from daily returns and then computes monthly:

- Open
- High
- Low
- Close
- Volume, which in this repository is monthly realized volatility
- AdjClose
- Return, which is the next-month label return
- RetCount, the number of non-zero trading-day returns used in the month

That monthly OHLC file is the core bridge between raw inputs and model-ready data.

## JKX and MXX representations

The codebase uses two main input encodings:

- **JKX** — a 2D candlestick-style image representation built by `imaging/build_cache.py` and consumed by the 2D CNNs.
- **MXX** — a 1D cumulative-return trajectory representation consumed by the 1D CNNs and the monthly/portfolio runners.

The architecture and dataset code treat these as separate branches, not as interchangeable formats.

## Targets and weights

The standard regression pipeline supports multiple target transforms built in `train/targets.py`:

- `raw`
- `sigma`
- `norm`
- `pct`

It also supports two weighting schemes:

- `weight_ew`
- `weight_ewpm`

## Model families

The repository currently has:

- 1D CNN and CNN-LSTM models for MXX
- 2D CNN and CNN-LSTM models for JKX

The model files are intentionally small; most training behavior lives in the loops and runners.

## Ensemble and forecasting logic

`train/ensemble.py` and the CLI runners implement a repeated pattern:

1. Fit one or more models.
2. Score each model with in-sample IC.
3. Keep only positive-IC models for aggregation.
4. IC-weight the forecasts into `f_hat`.
5. Convert cross-sectionally standardized forecasts into monthly `omega` weights.
6. Multiply `omega` by next-month returns to produce timed returns.

This is the core mechanism behind the ensemble and backtest outputs.

## Source references

- `factor_timing/data/factor_loader.py`
- `factor_timing/imaging/build_cache.py`
- `factor_timing/train/dataset.py`
- `factor_timing/train/targets.py`
- `factor_timing/train/ensemble.py`
- `factor_timing/train/monthly_dataset.py`
- `factor_timing/train/clustering.py`
