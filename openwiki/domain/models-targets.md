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

## Alternative target experimentation
`factor_timing/train/alt_targets.py` expands the target space beyond the base four transforms.
It defines:
- `raw`
- `sign`
- `multi_horizon`
- `drawdown_prob`
- `quantile`
- `topk_member`
- `volatility`

It also provides:
- per-target head definitions,
- target-specific loss functions,
- `AltTargetDataset`, which merges alt targets into a base image dataset.

This module is important because it shows where the repository is experimenting beyond the paper-default setup.

## Portfolio-level / top-K objective
`factor_timing/train/portfolio_loss.py` and `factor_timing/train/loop_portfolio.py` implement a different learning objective:
- the model is scored on the spread between soft top-K and soft bottom-K portfolios,
- training operates on whole-month batches rather than individual samples,
- a cosine-annealed temperature sharpens the portfolio selection over training.

This is the right path when the goal is portfolio construction rather than pointwise return prediction.

## What the outputs mean
The main training flows ultimately produce:
- `f_hat`: forecast score or expected return proxy,
- `omega`: normalized cross-sectional timing weight,
- `timed_return`: omega multiplied by next-month label return.

These values are then used in Excel workbooks, parquet outputs, and evaluation summaries.

## Practical cautions
- Do not assume the regression target and the evaluation metric are identical. Some runners predict one transformed target but rank by another scalar score.
- `drawdown_prob` and `volatility` are risk-oriented targets and invert or reshape the score semantics.
- The top-K portfolio objective is month-level, so it needs `MonthlyDataset` and a mask-aware collate function.
