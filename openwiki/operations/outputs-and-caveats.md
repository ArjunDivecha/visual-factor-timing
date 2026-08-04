---
type: Reference
title: Outputs and operational caveats
description: Output directory layout, per-run artifacts, W&B gating, hard-coded external Dropbox paths, README documentation drift, and pre-edit checks for factor_timing runners.
openwiki:
  roles: [operations, repository]
  source_paths: [factor_timing/cli/run_single.py, factor_timing/cli/run_ensemble.py, factor_timing/dashboard/build.py, factor_timing/imaging/build_cache.py, factor_timing/data/factor_loader.py]
---

# Outputs and operational caveats

## Output directory layout
The repository writes most generated artifacts under `factor_timing/outputs/`.
Common subtrees include:
- `cache_{panel}_w{window}/` — JKX cache (`images.npy`, `index.parquet`, `targets.parquet`),
- `cache_{panel}_w{window}_mxx/` — MXX trajectory cache (`trajectories.npy`, `index.parquet`),
- `runs/<run_id>/` — single-run artifacts,
- `runs/ENSEMBLE_<group>/` — aggregated ensemble outputs,
- `sweeps/<sweep_id>/` — sweep summaries.

## Run artifacts
A typical run directory contains some subset of:
- `progress.jsonl`
- `summary.json`
- `forecasts.parquet`
- `dashboard.html`
- Excel workbooks in the paper's output format (`_write_excel` writes `Monthly_Net_Returns`, `Omega_Weights`, `Expected_Return` sheets)

The dashboard builder reads `progress.jsonl` and `summary.json` to render charts and status tables (see [Workflow runners](../workflows/runners.md)).

## W&B logging
The CLI runners conditionally use Weights & Biases when `WANDB_API_KEY` is available. The import is guarded (`try: import wandb`) and `run_single` checks `spec.wandb_enabled and _HAS_WANDB and os.environ.get("WANDB_API_KEY")` before initializing. The source indicates that W&B is optional, not required.

## Hard-coded external paths
Several runners and builders embed absolute Dropbox paths to the source Excel workbooks and output directories. Examples appear in:
- `factor_timing/cli/run_single.py` (`MONTHLY_PANEL`, `DATA_ROOT`, `RUNS_ROOT`)
- `factor_timing/data/factor_loader.py` (`defaults` dict in the CLI)
- `factor_timing/imaging/build_cache.py` (`OHLC_PARQUET` and the `pattern.imaging.renderer` `sys.path.insert` to an absolute Dropbox Pattern path)

These paths are environment-specific and should be treated as operational assumptions, not portable defaults. The CLI exposes `--daily`, `--monthly`, `--out` overrides on `factor_loader` and `--panel`/`--window` on the cache builders.

## Documentation drift to remember
The top-level README still mentions some files that are not present in the current tree, such as:
- `imaging/mxx_renderer.py`
- `models/cnn.py`
- `models/cnn_lstm.py`
- `backtest/...`

The source tree currently uses:
- `imaging/build_mxx_cache.py`
- `models/cnn1d.py`
- `models/cnn2d.py`
- `train/` modules for training/evaluation logic

When the README and code disagree, prefer the source files.

## Known workflow caveats
- The repository does not currently have packaging metadata (`pyproject.toml` / `setup.py`), so the normal run pattern is `python -m ...` from the repo root.
- Some docstrings are more descriptive than authoritative; they may contain paths or claims that have not been fully updated (e.g. `loop.py`'s docstring lists `max_epochs=100` / `patience=7` while the runtime defaults are 200/15).
- The repo may contain generated outputs under `factor_timing/outputs/`; avoid treating those as canonical source unless they are specifically needed as evidence.

## What future agents should check before editing
1. Confirm whether a runner is used by current workflows or only retained for historical experiments.
2. Check whether new code should write into `factor_timing/outputs/` or a caller-supplied location.
3. Verify whether the intended model branch is MXX, JKX, or both.
4. Confirm whether a change affects cached file schemas, because the dataset loaders are schema-sensitive.
