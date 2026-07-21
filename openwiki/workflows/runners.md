---
type: "Reference"
title: "Workflow runners"
openwiki_generated: true
---

# Workflow runners

## Overview
The `factor_timing.cli` package contains the operational entrypoints for the project. Most workflows are meant to be run with `python -m ...` from the repository root.

## Single-run training
### `factor_timing.cli.run_single`
Purpose:
- train one configuration across factors and seed folds,
- log progress to `progress.jsonl`,
- emit `forecasts.parquet` and a T2_Optimizer-style Excel workbook.

What it uses:
- `FactorImageDataset` from `train/dataset.py`,
- `train_one()` from `train/loop.py`,
- model builders and forecast aggregation from `train/ensemble.py`.

Notable defaults from the source:
- panel `t2`
- window `12`
- arch `cnn`
- image `mxx`
- loss `mse`
- weight `weight_ew`
- target `raw`
- `n_folds=5`
- `val_frac=0.15`

Why it matters:
- this is the core atomic unit that other runners reuse.

## Ensemble aggregation
### `factor_timing.cli.run_ensemble`
Purpose:
- run all 64 `(arch, image, loss, weight, target)` combinations for one `(panel, window)` pair,
- aggregate the forecasts into MXX-only, JKX-only, and combined ensembles,
- write manifest and Parquet/Excel outputs.

Important behavior:
- IC weights are derived from mean in-sample training IC,
- non-positive or invalid IC weights are dropped,
- cross-sectional omega normalization uses median shifting so monthly median omega equals 1,
- the manifest is written incrementally so partial runs survive interruption.

## Hyperparameter sweeps
### `factor_timing.cli.run_sweep`
Purpose:
- run one or more `window:train_end` points,
- execute the full 64-combo ensemble at each point,
- write a rolling summary CSV/XLSX.

This is the widest orchestration mode in the repo and is useful when comparing lookback windows and train cutoffs under one consistent evaluation harness.

The sweep runner's `_strategy_metrics()` function computes, for each `window:train_end` point:
- monthly average IC and rank IC,
- HML quintile long-short Sharpe,
- Top-3 long/short Sharpe,
- Top-15 long-only Sharpe,
- baseline equal-weight Sharpe for comparison.

## Dashboard generation
### `factor_timing.dashboard.build`
Purpose:
- read a run directory,
- summarize progress metrics,
- generate a self-contained HTML dashboard,
- optionally refresh every 5 seconds in watch mode.

The dashboard expects `progress.jsonl` and `summary.json` in a run directory and is useful for monitoring long training jobs.

## Output conventions
Most runners write under `factor_timing/outputs/`:
- `runs/<run_id>/` for per-run artifacts,
- `runs/ENSEMBLE_<group>/` for aggregated ensemble outputs,
- `sweeps/<sweep_id>/` for cross-point summaries,
- `cache_*` and `cache_*_mxx/` for cached inputs.

## Operational caveats
- Several runners embed absolute Dropbox paths for source spreadsheets and output workbooks.
- `WANDB_API_KEY` controls whether W&B logging is active.
- The repo currently has no packaging metadata, so runners assume the working directory is the repo root.
- Some docstrings mention paths or file names that no longer match the current tree; the source code is the more reliable reference.
