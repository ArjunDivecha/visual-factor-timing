# Testing and change guide

This repository does not appear to ship a formal automated test suite, so future changes should be verified with targeted script runs, smoke checks, and careful inspection of generated artifacts.

## What to verify when changing each area

### Data loading and preprocessing

Files:

- `factor_timing/data/factor_loader.py`
- `factor_timing/train/targets.py`
- `factor_timing/train/alt_targets.py`

Checks:

- Confirm the output parquet schema still matches what the downstream dataset code expects.
- Confirm month-end alignment and next-month label shifting still behave the same.
- Confirm no secret-bearing or environment-only files are read into docs or generated outputs.

### Caching and dataset code

Files:

- `factor_timing/imaging/build_cache.py`
- `factor_timing/train/dataset.py`
- `factor_timing/train/monthly_dataset.py`

Checks:

- Verify cache indices still align with the underlying NumPy arrays.
- Verify JKX and MXX sample shapes still match the model inputs.
- Verify monthly batching still produces valid masks for ragged months.

### Models and training loops

Files:

- `factor_timing/models/cnn1d.py`
- `factor_timing/models/cnn2d.py`
- `factor_timing/train/loop.py`
- `factor_timing/train/loop_portfolio.py`
- `factor_timing/train/portfolio_loss.py`

Checks:

- Run a tiny training job and confirm the model converges without shape errors.
- Confirm early stopping still triggers after the intended minimum epoch floor.
- Confirm the portfolio-loss path still masks padded factor slots correctly.

### Ensemble and backtest runners

Files:

- `factor_timing/train/ensemble.py`
- `factor_timing/cli/run_single.py`
- `factor_timing/cli/run_ensemble.py`
- `factor_timing/cli/run_sweep.py`
- `factor_timing/cli/run_full_period.py`
- `factor_timing/cli/run_walkforward.py`
- `factor_timing/cli/run_walkforward_topk.py`
- `factor_timing/cli/run_walkforward_alt_targets.py`

Checks:

- Make sure generated run directories contain `summary.json`, `progress.jsonl`, and `forecasts.parquet`.
- Confirm Excel output still matches the optimizer workbook schema.
- Confirm the IC-weighting logic still drops non-positive IC combinations.
- Confirm walk-forward runners still respect the intended cold-start period and epoch scheduling.

### Dashboard

File:

- `factor_timing/dashboard/build.py`

Checks:

- Create a run directory with progress logs and confirm the HTML dashboard renders.
- If `--watch` is used, confirm repeated rebuilds do not corrupt the output file.

## Practical smoke-test suggestions

Use small, bounded runs rather than full sweeps when validating a change.

Good smoke tests:

- One `run_single` configuration with low `n_folds`
- One `run_walkforward` configuration on a small window
- One dashboard build from an existing run directory
- One cache build for each panel/window combination you touched

## Recent history that matters

The git history indicates the project recently added:

- stronger early stopping and arbitrary window sizes
- a 64-combo ensemble orchestrator
- W&B-backed single-run and dashboard support
- MonthlyDataset support for cross-sectional training
- portfolio-level loss experiments
- alternative targets and walk-forward runners

Those additions mean shape assumptions and split logic are the most likely places for regressions.

## Source references

- `factor_timing/train/loop.py`
- `factor_timing/train/loop_portfolio.py`
- `factor_timing/train/monthly_dataset.py`
- `factor_timing/train/portfolio_loss.py`
- `factor_timing/train/ensemble.py`
- `factor_timing/cli/*.py`
- `factor_timing/dashboard/build.py`
- `README.md`
