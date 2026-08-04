---
type: Reference
title: OpenWiki quickstart
description: Entry point for the visual-factor-timing knowledge base, with a compact task-routing table and links to architecture, workflows, domain, and operations pages.
openwiki:
  roles: [repository, architecture]
  source_paths: [README.md, factor_timing/data/factor_loader.py, factor_timing/imaging/build_cache.py, factor_timing/train/dataset.py, factor_timing/train/loop.py, factor_timing/train/ensemble.py, factor_timing/cli/run_single.py, factor_timing/cli/run_ensemble.py, factor_timing/cli/run_sweep.py, factor_timing/dashboard/build.py]
---

# OpenWiki quickstart

This repository implements a factor-timing research pipeline for the T2 and GDELT panels. The code builds monthly factor data, renders two kinds of model inputs (JKX candlestick images and MXX cumulative-return trajectories), trains 1D/2D CNN and CNN-LSTM networks, and aggregates predictions into timing weights (`omega`) and backtest outputs. The top-level `README.md` describes the project as a replication of Jia, Li, Zhang & Zhao (2025), *Timing the Factor Zoo via Deep Visualization*.

## What this project does

At a high level, the codebase supports three layers of work:

1. **Data preparation** — turn daily/monthly optimizer workbooks into monthly OHLC parquet, then into cached JKX images, MXX trajectories, and target tables.
2. **Modeling** — train per-sample or cross-sectional regression models over the [64-combo grid](domain/models-targets.md) of architecture, image type, loss, weight, and target.
3. **Evaluation** — [aggregate forecasts](workflows/runners.md) into IC-weighted ensembles, normalize cross-sectionally into monthly `omega` weights, and summarize IC, Sharpe, HML, and Top-K spreads.

## Task routing

| Change area or intent | Relevant wiki page | Exact source entry points | Important symbols or types | Focused tests | Minimal validation command |
|---|---|---|---|---|---|
| Add/change a data transform or OHLC schema | [Pipeline architecture](architecture/pipeline.md) | `factor_timing/data/factor_loader.py` | `build_monthly_ohlc`, `_synthetic_price`, `_monthly_ohlc_for_factor` | none (no test suite) | `python -m factor_timing.data.factor_loader --panel t2 --out /tmp/ohlc.parquet` (needs external xlsx) |
| Change image/trajectory cache layout | [Pipeline architecture](architecture/pipeline.md) | `factor_timing/imaging/build_cache.py`, `factor_timing/imaging/build_mxx_cache.py` | `build_cache`, `build_mxx_cache` | none | `python -m factor_timing.imaging.build_cache --panel t2 --window 12` |
| Add/change a target or weight transform | [Models, targets, and objectives](domain/models-targets.md) | `factor_timing/train/targets.py` | `_cross_sectional_transforms`, `_weighting`, `build_targets` | none | `python -m factor_timing.train.targets --panel t2 --window 12` |
| Change the per-sample dataset/filter behavior | [Pipeline architecture](architecture/pipeline.md) | `factor_timing/train/dataset.py` | `FactorImageDataset`, `split_dates` | none | build caches then run a `run_single` with `--n-folds 1` |
| Change cross-sectional batching | [Pipeline architecture](architecture/pipeline.md) | `factor_timing/train/monthly_dataset.py` | `MonthlyDataset`, `collate_months` | none | import + instantiate over a small `FactorImageDataset` |
| Change the training loop / early stopping / scheduler | [Models, targets, and objectives](domain/models-targets.md) | `factor_timing/train/loop.py` | `TrainConfig`, `train_one`, `predict`, `_default_device` | none | `python -m factor_timing.cli.run_single --n-folds 1` |
| Change a model architecture | [Models, targets, and objectives](domain/models-targets.md) | `factor_timing/models/cnn1d.py`, `factor_timing/models/cnn2d.py` | `CNN1D`, `CNNLSTM1D`, `CNN2D`, `CNNLSTM2D`, `CNN1DConfig`, `CNN2DConfig` | none | a tiny `run_single` run with the new image/arch |
| Change ensemble aggregation / omega logic | [Workflow runners](workflows/runners.md) | `factor_timing/train/ensemble.py` | `aggregate_forecasts`, `omega_from_forecasts`, `all_model_combos`, `_build_model`, `train_fold` | none | one `run_ensemble` run on a tiny panel window |
| Add/change the 64-combo CLI orchestration | [Workflow runners](workflows/runners.md) | `factor_timing/cli/run_ensemble.py` | `main`, `_combos`, `_combo_key`, `_median_omega`, `_ic_weighted` | none | `python -m factor_timing.cli.run_ensemble --panel t2 --window 12` |
| Add/change single-run training or its outputs | [Workflow runners](workflows/runners.md) | `factor_timing/cli/run_single.py` | `run`, `RunSpec`, `train_factor_folds`, `_write_excel` | none | `python -m factor_timing.cli.run_single --n-folds 1` |
| Change sweep strategy metrics or point iteration | [Workflow runners](workflows/runners.md) | `factor_timing/cli/run_sweep.py` | `main`, `_strategy_metrics`, `run_ensemble_for_point` | none | `python -m factor_timing.cli.run_sweep --panel gdelt --points w12:2023-03-31` |
| Change GDELT pooled clustering | [Pipeline architecture](architecture/pipeline.md) | `factor_timing/train/clustering.py` | `cluster_at`, `expanding_window_clusters`, `_corr_distance` | none | call `cluster_at` on a wide returns frame |
| Change run monitoring | [Workflow runners](workflows/runners.md) | `factor_timing/dashboard/build.py` | `build_dashboard`, `_read_log`, `_read_summary` | none | `python -m factor_timing.dashboard.build --run-id <run_id>` |

## Start here

- [Architecture overview](architecture.md)
- [Pipeline architecture](architecture/pipeline.md)
- [Core workflows](workflows.md)
- [Workflow runners](workflows/runners.md)
- [Domain concepts](domains.md)
- [Models, targets, and objectives](domain/models-targets.md)
- [Outputs and operational caveats](operations/outputs-and-caveats.md)
- [Testing and change guide](testing-and-change-guide.md)

## Major entrypoints

- `python -m factor_timing.data.factor_loader`
- `python -m factor_timing.imaging.build_cache`
- `python -m factor_timing.imaging.build_mxx_cache`
- `python -m factor_timing.train.targets`
- `python -m factor_timing.cli.run_single`
- `python -m factor_timing.cli.run_ensemble`
- `python -m factor_timing.cli.run_sweep`
- `python -m factor_timing.dashboard.build`

## Repository shape

- `factor_timing/data/` — raw-to-monthly data preparation.
- `factor_timing/imaging/` — image/trajectory cache builders.
- `factor_timing/models/` — 1D and 2D CNN and CNN-LSTM model definitions.
- `factor_timing/train/` — datasets, training loops, clustering, targets, and ensemble aggregation.
- `factor_timing/cli/` — experiment runners and sweep orchestration.
- `factor_timing/dashboard/` — run monitoring dashboard.
- `factor_timing/outputs/` — generated caches, run artifacts, and sweeps.

## Important constraints

- The repository expects external Excel inputs outside the repo; do not assume the data is bundled here. Several modules embed absolute Dropbox paths to those workbooks (see [Outputs and operational caveats](operations/outputs-and-caveats.md)).
- Most scripts resolve paths relative to the project root and expect to be launched from the repository directory.
- Several runners write into `factor_timing/outputs/` and may take a long time to finish; a single T2 64-combo epoch with 30 folds is ~159,000 model fits.
- W&B logging is optional and is gated by both the import and `WANDB_API_KEY`.
- The repository has no packaging metadata, so there is no `pip install`-style import path; use `python -m ...` from the repo root.

## If you are changing the code

- For data-format changes, start with `factor_timing/data/factor_loader.py` and `factor_timing/train/dataset.py`.
- For model changes, inspect `factor_timing/models/` and the relevant training loop in `factor_timing/train/`.
- For evaluation changes, inspect the runners in `factor_timing/cli/` and the Excel writer logic in `run_single.py` (`_write_excel`).
- For debugging long runs, use `factor_timing/dashboard/build.py`.

## Backlog

None. All substantial components have coverage after this update.

## Source evidence

This quickstart is grounded in:

- `README.md`
- `factor_timing/data/factor_loader.py`
- `factor_timing/imaging/build_cache.py`
- `factor_timing/imaging/build_mxx_cache.py`
- `factor_timing/train/dataset.py`
- `factor_timing/train/loop.py`
- `factor_timing/train/ensemble.py`
- `factor_timing/train/clustering.py`
- `factor_timing/cli/*.py`
- `factor_timing/dashboard/build.py`
