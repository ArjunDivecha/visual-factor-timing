---
type: Reference
title: Domain concepts
description: Core domain objects in factor_timing — panels, factors, monthly OHLC labels, the JKX and MXX input encodings, targets and weights, model families, and the IC-weighted ensemble forecast-to-omega-to-timed-return mechanism.
openwiki:
  roles: [domain, architecture]
  source_paths: [factor_timing/data/factor_loader.py, factor_timing/imaging/build_cache.py, factor_timing/imaging/build_mxx_cache.py, factor_timing/train/targets.py, factor_timing/train/ensemble.py, factor_timing/train/monthly_dataset.py, factor_timing/train/clustering.py]
---

# Domain concepts

This repository uses a small set of recurring domain objects. Understanding them makes the rest of the code much easier to navigate. The forecast-to-weight mechanism is the canonical home for the ensemble logic; see also [Models, targets, and objectives](domain/models-targets.md) for the model and target axes and [Pipeline architecture](architecture/pipeline.md) for the data flow.

## Panels

The main panels are **T2** and **GDELT**. The README also mentions ECON as a results panel, but the core source path in this repository is centered on T2 and GDELT.

Each panel is represented by external monthly and daily optimizer workbooks. `factor_loader.py` consumes those workbooks and creates a normalized monthly OHLC representation.

## Factors

A factor is a named time series column in the optimizer workbooks. Many scripts operate on one `(factor, end_date)` sample at a time, and that `(factor, end_date)` pair is the shared alignment key across caches, targets, datasets, and labels.

The code distinguishes between:

- **Per-factor training** — the T2 convention in `run_single.py`.
- **Pooled clustered training** — the GDELT convention, where factors are grouped by expanding-window Ward clustering in `clustering.py` (leakage-aware; see [Pipeline architecture](architecture/pipeline.md)).

## Monthly OHLC and labels

`factor_loader.py` constructs a synthetic price series from daily returns and then computes monthly:

- Open
- High
- Low
- Close
- Volume, which in this repository is monthly realized volatility (std of non-zero daily returns within the month)
- AdjClose
- Return, which is the next-month label return
- RetCount, the number of non-zero trading-day returns used in the month

That monthly OHLC file is the core bridge between raw inputs and model-ready data.

## JKX and MXX representations

The codebase uses two main input encodings:

- **JKX** — a 2D candlestick-style image representation built by `imaging/build_cache.py` and consumed by the 2D CNNs. Images are `uint8` shaped `(N, 1, 72, 3*window)`.
- **MXX** — a 1D cumulative-return trajectory representation built by `imaging/build_mxx_cache.py` and consumed by the 1D CNNs and the monthly/portfolio runners. Trajectories are `float32` shaped `(N, window)` where each row is the cumulative return rebased to the window's first month.

The architecture and dataset code treat these as separate branches, not as interchangeable formats. Both caches are indexed 1:1 by `(factor, end_date)` so the two branches can be ensembled at the signal-combination step.

## Targets and weights

The standard regression pipeline supports multiple target transforms built in `train/targets.py` (see [Models, targets, and objectives](domain/models-targets.md)):

- `raw`
- `sigma`
- `norm`
- `pct`

It also supports two weighting schemes:

- `weight_ew`
- `weight_ewpm`

These are separate axes from image type and architecture, so they participate in the 64-combo sweep.

## Model families

The repository currently has:

- 1D CNN and CNN-LSTM models for MXX (`models/cnn1d.py`)
- 2D CNN and CNN-LSTM models for JKX (`models/cnn2d.py`)

The model files are intentionally small; most training behavior lives in the loops and runners.

## Ensemble and forecasting logic

`train/ensemble.py` and the CLI runners implement a repeated pattern that turns per-fold forecasts into monthly timing weights:

1. Fit one or more models (`train_fold`).
2. Score each model with in-sample IC on the training set.
3. Keep only positive-IC models for aggregation (`aggregate_forecasts` clips non-positive IC weights to 0).
4. IC-weight the surviving forecasts into `f_hat` per `(factor, end_date)`.
5. Cross-sectionally z-score `f_hat` each month and median-shift so the monthly median `omega` equals 1 (`omega_from_forecasts`).
6. Multiply `omega` by the next-month `label_return` to produce `timed_return`.

This is the core mechanism behind the ensemble and backtest outputs and is shared between `train/ensemble.py` and the CLI `run_ensemble` (which duplicates `_median_omega`/`_ic_weighted`). See [Workflow runners](workflows/runners.md) for the orchestration that drives it.

## Source references

- `factor_timing/data/factor_loader.py`
- `factor_timing/imaging/build_cache.py`
- `factor_timing/imaging/build_mxx_cache.py`
- `factor_timing/train/targets.py`
- `factor_timing/train/ensemble.py`
- `factor_timing/train/monthly_dataset.py`
- `factor_timing/train/clustering.py`
