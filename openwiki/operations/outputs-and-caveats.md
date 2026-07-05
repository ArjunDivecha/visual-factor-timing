# Outputs and operational caveats

## Output directory layout
The repository writes most generated artifacts under `factor_timing/outputs/`.
Common subtrees include:
- `cache_{panel}_w{window}/` — JKX cache, index, targets,
- `cache_{panel}_w{window}_mxx/` — MXX trajectory cache,
- `runs/<run_id>/` — single-run artifacts,
- `runs/ENSEMBLE_<group>/` — aggregated ensemble outputs,
- `sweeps/<sweep_id>/` — sweep summaries.

## Run artifacts
A typical run directory contains some subset of:
- `progress.jsonl`
- `epochs.jsonl`
- `summary.json`
- `forecasts.parquet`
- `dashboard.html`
- Excel workbooks in the paper’s output format

The dashboard builder reads `progress.jsonl` and `summary.json` to render charts and status tables.

## W&B logging
The CLI runners conditionally use Weights & Biases when `WANDB_API_KEY` is available. The source indicates that W&B is optional, not required.

## Hard-coded external paths
Several runners and builders embed absolute Dropbox paths to the source Excel workbooks and output directories. Examples appear in:
- `factor_timing/cli/run_single.py`
- `factor_timing/cli/run_ensemble.py`
- `factor_timing/dashboard/build.py`

These paths are environment-specific and should be treated as operational assumptions, not portable defaults.

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
- Some docstrings are more descriptive than authoritative; they may contain paths or claims that have not been fully updated.
- The repo contains many generated outputs under `factor_timing/outputs/`; avoid treating those as canonical source unless they are specifically needed as evidence.

## What future agents should check before editing
1. Confirm whether a runner is used by current workflows or only retained for historical experiments.
2. Check whether new code should write into `factor_timing/outputs/` or a caller-supplied location.
3. Verify whether the intended model branch is MXX, JKX, or both.
4. Confirm whether a change affects cached file schemas, because the dataset loaders are schema-sensitive.
