---
type: "Reference"
title: "Models, targets, and objectives"
openwiki_generated: true
---

# Models, targets, and objectives

## Model branches
The repository has two input geometries and matching model families.

### MXX branch
The MXX path models cumulative-return trajectories directly as 1D sequences.
- `factor_timing/models/cnn1d.py` defines the 1D CNN and CNN-LSTM variants.
- `factor_timing/train/dataset.py` returns tensors shaped like `(T, 1)` for MXX samples.

### JKX branch
The JKX path models candlestick-style monthly images.
- `factor_timing/models/cnn2d.py` defines the 2D CNN and CNN-LSTM variants.
- `factor_timing/train/dataset.py` returns tensors shaped like `(1, H, W)` for JKX samples.

Both model modules use the same design pattern:
- 64 filters,
- kernel size 5,
- SiLU activation,
- dropout 0.25,
- a regression head whose size is inferred with a dummy forward pass.

## Regression targets used in the main ensemble grid
The base ensemble grid in `train/ensemble.py` uses four target transforms:
- `raw`
- `sigma`
- `norm`
- `pct`

`FactorImageDataset` selects the active target column by name, and `train/loop.py` trains with either weighted MSE or weighted MAE.

## Target transformations
`factor_timing/train/targets.py` contains the target transform utilities referenced by the README and training code. In practice, the main pointwise runners rely on the dataset’s target columns and the loop’s loss selection.

The important operational point is that target transforms are a separate axis from image type and architecture, so they participate in the 64-combo sweep.

## What the outputs mean
The main training flows ultimately produce:
- `f_hat`: forecast score or expected return proxy,
- `omega`: normalized cross-sectional timing weight,
- `timed_return`: omega multiplied by next-month label return.

These values are then used in Excel workbooks, parquet outputs, and evaluation summaries.

## Practical cautions
- Do not assume the regression target and the evaluation metric are identical. Some runners predict one transformed target but rank by another scalar score.
