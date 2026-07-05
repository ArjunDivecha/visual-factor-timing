# OpenWiki quickstart

This repository implements a factor-timing research pipeline for the T2 and GDELT panels. The code builds monthly factor data, renders two kinds of model inputs, trains neural nets, and aggregates predictions into timing weights and backtest outputs. The top-level README describes the project as a replication of Jia, Li, Zhang & Zhao (2025), *Timing the Factor Zoo via Deep Visualization*.

## What this project does

At a high level, the codebase supports three layers of work:

1. **Data preparation** — turn daily/monthly optimizer workbooks into monthly OHLC and cached model inputs.
2. **Modeling** — train 1D/2D CNN and CNN-LSTM variants on MXX and JKX representations.
3. **Evaluation** — produce factor weights, backtest them as timed returns, and summarize IC, Sharpe, HML, and top-K spreads.

## Start here

- [Architecture overview](architecture.md)
- [Core workflows](workflows.md)
- [Domain concepts](domains.md)
- [Testing and change guide](testing-and-change-guide.md)

## Major entrypoints

- `python -m factor_timing.data.factor_loader`
- `python -m factor_timing.imaging.build_cache`
- `python -m factor_timing.imaging.build_mxx_cache`
- `python -m factor_timing.cli.run_single`
- `python -m factor_timing.cli.run_ensemble`
- `python -m factor_timing.cli.run_sweep`
- `python -m factor_timing.dashboard.build`

## Repository shape

- `factor_timing/data/` — raw-to-monthly data preparation.
- `factor_timing/imaging/` — image/trajectory cache builders.
- `factor_timing/models/` — CNN and CNN-LSTM model definitions.
- `factor_timing/train/` — datasets, training loops, clustering, and targets.
- `factor_timing/cli/` — experiment runners and backtests.
- `factor_timing/dashboard/` — run monitoring dashboard.
- `factor_timing/outputs/` — generated caches, run artifacts, and sweeps.

## Important constraints

- The repository expects external Excel inputs outside the repo; do not assume the data is bundled here.
- Most scripts resolve paths relative to the project root and expect to be launched from the repository directory.
- Several runners write into `factor_timing/outputs/` and may take a long time to finish.
- W&B logging is optional and depends on environment configuration.

## If you are changing the code

- For data-format changes, start with `factor_timing/data/factor_loader.py` and `factor_timing/train/dataset.py`.
- For model changes, inspect `factor_timing/models/` and the relevant training loop in `factor_timing/train/`.
- For evaluation changes, inspect the runners in `factor_timing/cli/` and the Excel writer logic in `run_single.py`.
- For debugging long runs, use `factor_timing/dashboard/build.py`.

## Source evidence

This quickstart is grounded in:

- `README.md`
- `factor_timing/data/factor_loader.py`
- `factor_timing/imaging/build_cache.py`
- `factor_timing/train/dataset.py`
- `factor_timing/train/loop.py`
- `factor_timing/train/ensemble.py`
- `factor_timing/cli/*.py`
- `factor_timing/dashboard/build.py`

