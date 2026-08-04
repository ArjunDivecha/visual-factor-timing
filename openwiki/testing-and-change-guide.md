---
type: Reference
title: Testing and change guide
description: No automated test suite exists for factor_timing; this page lists what to verify per area, practical smoke-test commands, and the narrow validation that proves each change without a full sweep.
openwiki:
  roles: [testing, operations]
  source_paths: [factor_timing/data/factor_loader.py, factor_timing/train/targets.py, factor_timing/imaging/build_cache.py, factor_timing/imaging/build_mxx_cache.py, factor_timing/train/dataset.py, factor_timing/train/monthly_dataset.py, factor_timing/models/cnn1d.py, factor_timing/models/cnn2d.py, factor_timing/train/loop.py, factor_timing/train/ensemble.py, factor_timing/cli/run_single.py, factor_timing/cli/run_ensemble.py, factor_timing/cli/run_sweep.py, factor_timing/dashboard/build.py]
---

# Testing and change guide

This repository does not ship a formal automated test suite, so future changes should be verified with targeted script runs, smoke checks, and careful inspection of generated artifacts. There is no `pytest`/unit-test harness to lean on; the smoke commands below are the narrowest checks available.

## What to verify when changing each area

### Data loading and preprocessing

Files:

- `factor_timing/data/factor_loader.py`
- `factor_timing/train/targets.py`

Checks:

- Confirm the output parquet schema still matches what the downstream dataset code expects (`Ticker, Date, Open, High, Low, Close, Volume, AdjClose, Return, RetCount`).
- Confirm month-end alignment and next-month label shifting still behave the same (`monthly_me` index normalization + `lbl.shift(-1)`).
- Confirm no secret-bearing or environment-only files are read into docs or generated outputs.
- Narrow command: `python -m factor_timing.data.factor_loader --panel t2 --out /tmp/ohlc.parquet` (requires external xlsx at the hard-coded Dropbox paths or `--daily`/`--monthly` overrides).

### Caching and dataset code

Files:

- `factor_timing/imaging/build_cache.py`
- `factor_timing/imaging/build_mxx_cache.py`
- `factor_timing/train/dataset.py`
- `factor_timing/train/monthly_dataset.py`

Checks:

- Verify cache indices still align with the underlying NumPy arrays (`row` column maps to array row).
- Verify JKX `(N, 1, 72, 3*window)` uint8 and MXX `(N, window)` float32 sample shapes still match the model inputs.
- Verify monthly batching still produces valid masks for ragged months (`collate_months`).
- Narrow command: `python -m factor_timing.imaging.build_cache --panel t2 --window 12` then `python -m factor_timing.imaging.build_mxx_cache --panel t2 --window 12`.

### Targets and weights

Files:

- `factor_timing/train/targets.py`

Checks:

- Confirm `targets.parquet` columns (`factor, end_date, row, raw, sigma, norm, pct, weight_ew, weight_ewpm`) still match what `FactorImageDataset` merges on.
- Confirm cross-sectional transforms drop months with fewer than three valid factors and remain leakage-free.
- Narrow command: `python -m factor_timing.train.targets --panel t2 --window 12`.

### Models and training loops

Files:

- `factor_timing/models/cnn1d.py`
- `factor_timing/models/cnn2d.py`
- `factor_timing/train/loop.py`

Checks:

- Run a tiny training job and confirm the model converges without shape errors.
- Confirm early stopping still triggers after the intended minimum epoch floor (`min_delta` + `patience`).
- Confirm `TrainConfig` runtime defaults (`max_epochs=200`, `patience=15`) match intent; the module docstring still states 100/7.
- Narrow command: `python -m factor_timing.cli.run_single --n-folds 1`.

### Ensemble and backtest runners

Files:

- `factor_timing/train/ensemble.py`
- `factor_timing/cli/run_single.py`
- `factor_timing/cli/run_ensemble.py`
- `factor_timing/cli/run_sweep.py`

Checks:

- Make sure generated run directories contain `summary.json`, `progress.jsonl`, and `forecasts.parquet`.
- Confirm Excel output still matches the optimizer workbook schema (`Monthly_Net_Returns`, `Omega_Weights`, `Expected_Return` sheets; month-end forecast dates converted to first-of-month).
- Confirm the IC-weighting logic still drops non-positive IC combinations (`aggregate_forecasts` and the duplicated `_ic_weighted`/`_median_omega`).
- Narrow command: `python -m factor_timing.cli.run_ensemble --panel t2 --window 12 --n-folds 1` (still heavy: 64 combos); for a cheaper check use `python -m factor_timing.cli.run_single --n-folds 1`.

### Dashboard

File:

- `factor_timing/dashboard/build.py`

Checks:

- Create a run directory with progress logs and confirm the HTML dashboard renders.
- If `--watch` is used, confirm repeated rebuilds do not corrupt the output file.
- Narrow command: `python -m factor_timing.dashboard.build --run-id <run_id>`.

## Practical smoke-test suggestions

Use small, bounded runs rather than full sweeps when validating a change. Good smoke tests:

- One `run_single` configuration with `--n-folds 1`.
- One dashboard build from an existing run directory.
- One cache build for each panel/window combination you touched (JKX via `build_cache`, MXX via `build_mxx_cache`).
- One `targets` build for the affected panel/window.

There is no test that runs the full 64-combo ensemble cheaply; treat any `run_ensemble`/`run_sweep` invocation as an expensive integration check and gate it behind having first validated the single-run path.

## Recent history that matters

The git history indicates the project recently added:

- stronger early stopping (`min_delta` threshold, `patience=15`) and arbitrary window sizes
- a 64-combo ensemble orchestrator
- W&B-backed single-run and dashboard support
- `MonthlyDataset` support for cross-sectional training

Those additions mean shape assumptions, split logic, and the duplicated omega/IC-weighting logic are the most likely places for regressions.

## Source references

- `factor_timing/train/loop.py`
- `factor_timing/train/monthly_dataset.py`
- `factor_timing/train/ensemble.py`
- `factor_timing/cli/*.py`
- `factor_timing/dashboard/build.py`
- `README.md`
