---
type: Reference
title: Models, targets, and objectives
description: The factor_timing model branches (1D MXX, 2D JKX CNN and CNN-LSTM), the four regression targets and two weighting schemes that form the 64-combo grid, runtime training defaults, and the f_hat/omega/timed_return output semantics.
openwiki:
  roles: [architecture, domain, testing]
  source_paths: [factor_timing/models/cnn1d.py, factor_timing/models/cnn2d.py, factor_timing/train/targets.py, factor_timing/train/loop.py, factor_timing/train/ensemble.py]
  symbols: [CNN1D, CNNLSTM1D, CNN1DConfig, CNN2D, CNNLSTM2D, CNN2DConfig, TrainConfig, train_one, predict, aggregate_forecasts, omega_from_forecasts, _build_model]
  invariants: ["Regression head width is inferred with a dummy forward pass, so models adapt to window length", "IC weights <= 0 contribute nothing to the ensemble", "Runtime TrainConfig defaults (max_epochs=200, patience=15) differ from the loop.py docstring (100/7)"]
---

# Models, targets, and objectives

## Model branches
The repository has two input geometries and matching model families. See [Pipeline architecture](../architecture/pipeline.md) for how the caches feed these models and [Workflow runners](../workflows/runners.md) for the orchestration that trains and aggregates them.

### MXX branch
The MXX path models cumulative-return trajectories directly as 1D sequences.
- `factor_timing/models/cnn1d.py` defines `CNN1D` and `CNNLSTM1D`.
- `factor_timing/train/dataset.py` returns tensors shaped `(T, 1)` for MXX samples (the CNN1D feature extractor transposes to `(B, C, T)` for Conv1d).

### JKX branch
The JKX path models candlestick-style monthly images.
- `factor_timing/models/cnn2d.py` defines `CNN2D` and `CNNLSTM2D`.
- `factor_timing/train/dataset.py` returns tensors shaped `(1, H, W)` for JKX samples (loaded as uint8 and divided by 255).

Both model modules use the same design pattern:
- 64 filters,
- kernel size 5 (1D) or 5x5 (2D),
- SiLU activation,
- dropout 0.25,
- MaxPool (size 2 / 2x2),
- a regression head whose width is inferred with a dummy forward pass so the same class adapts to `window` length.

`CNNLSTM2D` runs its LSTM along the image-width axis (one timestep per image column), treating the width as the "time" dimension that corresponds to months in the JKX image.

## Regression targets used in the main ensemble grid
The base ensemble grid in `train/ensemble.py` uses four target transforms (`TARGETS`):
- `raw`
- `sigma`
- `norm`
- `pct`

`FactorImageDataset` selects the active target column by name, and `train/loop.py` trains with either weighted MSE or weighted MAE (`_weighted_loss`). The target/weight axes are separate from image type and architecture, which is what yields the 64-combo grid (`ARCHES x IMAGES x LOSSES x WEIGHTS x TARGETS`).

## Target transformations
`factor_timing/train/targets.py` computes the four target columns (`raw`, `sigma`, `norm`, `pct`) and the two weight columns (`weight_ew`, `weight_ewpm`) written to `targets.parquet`. The transforms are leakage-aware:

- `sigma` uses a trailing expanding factor mean and the cross-sectional std of the same label month.
- `norm`/`pct` rank within the valid factors of each label month (months with fewer than three valid factors are skipped).

The important operational point is that target transforms are a separate axis from image type and architecture, so they participate in the 64-combo sweep and in `train_fold`'s in-sample IC computation.

## Runtime training defaults
`train/loop.py` exposes `TrainConfig`. The runtime defaults differ from the module docstring (which restates the paper's §3.2 hyperparameters):
- `lr=1e-3`, `beta1=0.9`, `beta2=0.99`, `eps=1e-7` (matches paper)
- `batch_size=32768` (2**15), capped by dataset size in `train_one`
- `max_epochs=200` (docstring says 100)
- `patience=15` (docstring says 7)
- `min_delta=1e-4`, `lr_reduce_patience=4`, `lr_reduce_factor=0.5`, `min_lr=1e-5`

`_default_device()` prefers Apple Silicon MPS > CUDA > CPU. Early stopping requires a `min_delta` improvement to reset the counter, and the ReduceLROnPlateau scheduler runs an independent plateau counter.

`run_single.RunSpec` overrides `batch_size=256` (much smaller than the paper's 2**15 because per-factor datasets are only thousands of samples) and keeps `max_epochs=200`, `patience=15`.

## What the outputs mean
The main training flows ultimately produce (see [Domain concepts](../domains.md) for the canonical mechanism):
- `f_hat`: IC-weighted forecast score / expected return proxy,
- `omega`: normalized cross-sectional timing weight (monthly median = 1),
- `timed_return`: `omega` multiplied by next-month `label_return`.

These values are then used in Excel workbooks, parquet outputs, and evaluation summaries (OOS IC, HML Sharpe, Top-K Sharpes).

## Practical cautions
- Do not assume the regression target and the evaluation metric are identical. `train_fold` predicts the selected target column but in-sample IC may be computed against `raw` when available, and the sweep's `_strategy_metrics` ranks by `omega` against `label_return`.
- The `aggregate_forecasts` in `train/ensemble.py` and `_ic_weighted`/`_median_omega` in `cli/run_ensemble.py` are duplicated implementations of the same IC-weighting and omega logic; change both together.
