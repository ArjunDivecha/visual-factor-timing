---
type: Reference
title: Workflows
description: Canonical end-to-end workflows for factor_timing, from building monthly data and caches through single-run training, 64-combo ensemble aggregation, hyperparameter sweeps, and run monitoring.
openwiki:
  roles: [workflow, operations]
  source_paths: [factor_timing/data/factor_loader.py, factor_timing/imaging/build_cache.py, factor_timing/imaging/build_mxx_cache.py, factor_timing/train/targets.py, factor_timing/train/dataset.py, factor_timing/train/loop.py, factor_timing/train/ensemble.py, factor_timing/cli/run_single.py, factor_timing/cli/run_ensemble.py, factor_timing/cli/run_sweep.py, factor_timing/dashboard/build.py]
---

# Workflows

This page captures the main ways the repository is used in practice and the source files that govern each workflow. For the runner internals and orchestration flow, see [Workflow runners](workflows/runners.md).

## 1) Build monthly data and caches

Canonical flow (each step is a prerequisite for the next):

1. Load daily and monthly optimizer workbooks with `python -m factor_timing.data.factor_loader --panel {t2|gdelt}`, writing `{panel}_monthly_ohlc.parquet`.
2. Render the JKX image cache with `python -m factor_timing.imaging.build_cache --panel {panel} --window {window}`, writing `cache_{panel}_w{window}/images.npy` and `index.parquet`.
3. Render the MXX trajectory cache with `python -m factor_timing.imaging.build_mxx_cache --panel {panel} --window {window}`, writing `cache_{panel}_w{window}_mxx/trajectories.npy` and `index.parquet`.
4. Build the target/weight table with `python -m factor_timing.train.targets --panel {panel} --window {window}`, writing `cache_{panel}_w{window}/targets.parquet`.
5. Consume the resulting caches through `factor_timing.train.dataset.FactorImageDataset`.

This is the prerequisite for every training and backtest runner. Steps 2 and 3 are independent of each other (both read the OHLC parquet) but step 4 depends on the JKX index, and the dataset layer reads all three artifacts.

## 2) Single configuration training

`python -m factor_timing.cli.run_single` is the main "train one model family on all factors" runner (defaults: `panel=t2 window=12 arch=cnn image=mxx loss=mse weight=weight_ew target=raw n_folds=5 train_end=2018-12-31 val_frac=0.15`).

Behavior to know:

- It trains per factor for the T2 convention.
- It loops over `n_folds` seeded fits per factor, computing in-sample IC per fold.
- It writes `progress.jsonl`, `epochs.jsonl`-equivalent history, `summary.json`, `forecasts.parquet`, and a T2_Optimizer-style Excel workbook (`_write_excel` pivots into `Monthly_Net_Returns`, `Omega_Weights`, `Expected_Return` sheets).
- W&B is optional and gated by both the import and `WANDB_API_KEY`.

Use this runner when debugging the core training loop or validating a new model/loss/target choice.

## 3) 64-combo ensemble runs

`python -m factor_timing.cli.run_ensemble --panel {panel} --window {window}` iterates over the 64 combinations of architecture, image type, loss, weight, and target produced by `_combos()`.

The flow is:

- Call `run_single` for each combo.
- Read its `progress.jsonl` to compute the mean training IC across folds (`ic_train_mean`).
- IC-weight the surviving forecasts into MXX-only, JKX-only, and combined ensembles via `_ic_weighted` (combos with non-positive or NaN IC are skipped).
- Cross-sectionally standardize and median-shift into `omega` via `_median_omega` (monthly median `omega = 1`).
- Attach `label_return` from the JKX index and compute `timed_return = omega * label_return`.
- Write `ensemble_{mxx|jkx|combined}.parquet`, the per-ensemble Excel workbooks, and a `manifest.json` that is updated incrementally so partial runs survive interruption.

This is the canonical path for reproducing the paper-style ensemble outputs.

## 4) Hyperparameter sweeps

`python -m factor_timing.cli.run_sweep --panel {panel} --points w12:2023-03-31 w24:2023-03-31 ...` repeats ensemble runs across multiple `window:train_end` points and collects a summary table.

This is the right workflow when the question is not "which combo is best?" but "which window and training cutoff are most robust?" For each point it runs the full 64-combo ensemble (`run_ensemble_for_point`) and computes strategy metrics via `_strategy_metrics` (see [Workflow runners](workflows/runners.md) for the full metric list). The sweep writes `summary.csv` and `summary.xlsx` under `factor_timing/outputs/sweeps/{sweep_id}/`, with a partial `summary.csv` after each point so progress is visible mid-sweep.

## 5) Run monitoring

`python -m factor_timing.dashboard.build --run-id <run_id>` turns a run directory into a single-file HTML dashboard. With `--watch` it rebuilds every 5 seconds (REFRESH_SECONDS) and is designed to monitor `progress.jsonl` during long jobs.

If a run is already underway, this is the first place to look for progress visibility.

## Change-oriented notes for future agents

- If you change data schema or cache layout, update the loader, both cache builders, the targets builder, the dataset, and every runner that reads cached parquet files.
- If you change train/val splitting rules, re-check `run_single.run` (which derives `val_start`/`val_end` from `val_frac`) and the ensemble logic; they implement the split conventions.
- If you change how `omega` is derived from forecasts, update both `train/ensemble.omega_from_forecasts` and `cli/run_ensemble._median_omega` (which duplicates the logic) and the sweep's reuse of `_median_omega`.

## Source references

- `factor_timing/data/factor_loader.py`
- `factor_timing/imaging/build_cache.py`
- `factor_timing/imaging/build_mxx_cache.py`
- `factor_timing/train/targets.py`
- `factor_timing/train/dataset.py`
- `factor_timing/train/loop.py`
- `factor_timing/train/ensemble.py`
- `factor_timing/cli/run_single.py`
- `factor_timing/cli/run_ensemble.py`
- `factor_timing/cli/run_sweep.py`
- `factor_timing/dashboard/build.py`
