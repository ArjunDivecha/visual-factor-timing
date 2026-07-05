# Workflows

This page captures the main ways the repository is used in practice and the source files that govern each workflow.

## 1) Build monthly data and caches

Canonical flow:

1. Load daily and monthly optimizer workbooks with `factor_timing.data.factor_loader`.
2. Render JKX image caches with `factor_timing.imaging.build_cache`.
3. Build targets and weights with `factor_timing.train.targets`.
4. Consume the resulting caches through `factor_timing.train.dataset.FactorImageDataset`.

This is the prerequisite for every training and backtest runner.

## 2) Single configuration training

`factor_timing.cli.run_single` is the main “train one model family on all factors” runner.

Behavior to know:

- It trains per factor for the T2 convention.
- It loops over `n_folds` seeded fits per factor.
- It writes `progress.jsonl`, `epochs.jsonl`, `summary.json`, `forecasts.parquet`, and an Excel workbook.
- W&B is optional and gated by both the import and `WANDB_API_KEY`.

Use this runner when debugging the core training loop or validating a new model/loss/target choice.

## 3) 64-combo ensemble runs

`factor_timing.cli.run_ensemble` iterates over the 64 combinations of architecture, image type, loss, weight, and target.

The flow is:

- Call `run_single` for each combo.
- Read its `progress.jsonl` to compute the mean training IC.
- IC-weight the surviving forecasts into MXX-only, JKX-only, and combined ensembles.
- Write parquet, Excel, and manifest outputs under a shared `ENSEMBLE_<group>` directory.

This is the canonical path for reproducing the paper-style ensemble outputs.

## 4) Hyperparameter sweeps

`factor_timing.cli.run_sweep` repeats ensemble runs across multiple `window:train_end` points and collects a summary table.

This is the right workflow when the question is not “which combo is best?” but “which window and training cutoff are most robust?”

## 5) Full-period comparisons

`factor_timing.cli.run_full_period` trains each combo on the full available history and produces a full-period in-sample comparison.

This workflow exists to compare strategies over the entire panel history, but the README and code both warn that the numbers are in-sample and may overstate deployable performance.

## 6) Walk-forward evaluation

Three walk-forward runners exist:

- `run_walkforward.py` — standard regression baseline.
- `run_walkforward_topk.py` — top-K spread loss variant.
- `run_walkforward_alt_targets.py` — alternative-target family comparison.

These runners expand the train window over time, predict the next block of months, and then concatenate all predictions into a continuous OOS series.

The walk-forward path is the most important one when you want a conservative estimate of real-time usefulness.

## 7) Run monitoring

`factor_timing.dashboard.build` turns a run directory into a single-file HTML dashboard. It is designed to monitor `progress.jsonl` during long jobs.

If a run is already underway, this is the first place to look for progress visibility.

## Change-oriented notes for future agents

- If you change data schema or cache layout, update the loader, dataset, and every runner that reads cached parquet files.
- If you change train/val splitting rules, re-check both `run_single.py` and the walk-forward scripts; they implement slightly different conventions.
- If you change how `omega` is derived from forecasts, update both the ensemble logic and the backtest metrics.
- If you change the portfolio-loss path, ensure the monthly dataset collation still handles ragged month sizes correctly.

## Source references

- `factor_timing/data/factor_loader.py`
- `factor_timing/imaging/build_cache.py`
- `factor_timing/train/targets.py`
- `factor_timing/train/dataset.py`
- `factor_timing/train/loop.py`
- `factor_timing/train/ensemble.py`
- `factor_timing/cli/run_single.py`
- `factor_timing/cli/run_ensemble.py`
- `factor_timing/cli/run_sweep.py`
- `factor_timing/cli/run_full_period.py`
- `factor_timing/cli/run_walkforward.py`
- `factor_timing/cli/run_walkforward_topk.py`
- `factor_timing/cli/run_walkforward_alt_targets.py`
- `factor_timing/dashboard/build.py`
