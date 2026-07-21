---
type: "Reference"
title: "Pipeline architecture"
openwiki_generated: true
---

# Pipeline architecture

## Overview
The repository’s core pipeline is:
1. build monthly factor panels from external optimizer spreadsheets,
2. convert them into cacheable model inputs,
3. load the cached inputs through dataset wrappers,
4. train neural nets on either per-sample or cross-sectional objectives,
5. aggregate forecasts into timing weights and evaluation outputs.

This flow is implemented in small, composable modules rather than one monolithic training script.

## Data construction
`factor_timing/data/factor_loader.py` is the upstream normalization stage. It:
- loads daily and monthly Excel workbooks,
- compounds daily percentage returns into synthetic prices,
- computes monthly OHLC bars and realized volatility,
- aligns next-month returns as labels,
- writes panel-specific parquet outputs such as `t2_monthly_ohlc.parquet` and `gdelt_monthly_ohlc.parquet`.

Why it exists:
- the original spreadsheets are not model-ready,
- the monthly OHLC/vol tables become the common input for both image branches and the target builder.

## Cache generation
There are two cache builders:
- `factor_timing/imaging/build_cache.py` for JKX-style image caches,
- `factor_timing/imaging/build_mxx_cache.py` for MXX trajectory caches.

The source tree and downstream loaders indicate two different input geometries:
- **JKX**: 2D candlestick-style images stored as `images.npy` under `cache_{panel}_w{window}/`.
- **MXX**: 1D cumulative-return trajectories stored as `trajectories.npy` under `cache_{panel}_w{window}_mxx/`.

Each cache is indexed by `factor` and `end_date`, with a matching `index.parquet` to align rows back to metadata and labels.

## Dataset layer
`factor_timing/train/dataset.py` wraps the caches into `FactorImageDataset`.
It returns `(X, y, w)` tuples and supports:
- JKX or MXX inputs,
- target selection (`raw`, `sigma`, `norm`, `pct`),
- sample-weight selection (`weight_ew`, `weight_ewpm`),
- factor filters and date filters for train/val/test splits.

`factor_timing/train/monthly_dataset.py` groups samples by month for cross-sectional training. It pads ragged month sizes and emits a mask so losses can operate on full cross-sections.

## Training layer
`factor_timing/train/loop.py` is the standard regression loop.
It provides:
- `TrainConfig`,
- `train_one()` with Adam, weighted MSE/MAE, ReduceLROnPlateau, and early stopping,
- `predict()` for batched inference.

## Ensemble layer
`factor_timing/train/ensemble.py` defines the combinatorial experiment grid:
- 2 architectures,
- 2 image types,
- 2 losses,
- 2 weighting schemes,
- 4 target transforms.

That yields 64 base model combinations. The ensemble logic:
- trains folds for each combination,
- computes in-sample IC as a model weight,
- drops non-positive or invalid weights,
- averages forecasts into `f_hat`,
- rescales cross-sections into monthly `omega` values.

## Pooled GDELT training
`factor_timing/train/clustering.py` is used for leakage-aware pooled training.
It computes expanding-window Ward clusters from historical factor-return correlations and refits clusters using only data available before each retrain date.

This matters because the GDELT panel is trained in pooled groups, while T2 is trained per factor.

## Model definitions
The model branch is split by input geometry:
- `factor_timing/models/cnn1d.py` handles MXX trajectories with 1D CNN and CNN-LSTM variants.
- `factor_timing/models/cnn2d.py` handles JKX images with 2D CNN and CNN-LSTM variants.

The code sizes linear layers with a dummy forward pass, so the same model classes adapt to window length.

## Data flow summary
`factor_timing/outputs/` acts as the handoff layer between stages:
- cache builders write `cache_*` directories,
- trainers read those caches,
- CLI runners write `runs/`, `sweeps/`, and dashboard artifacts.

## What to watch when editing this area
- Keep `(factor, end_date)` alignment consistent across caches, datasets, and labels.
- Avoid mixing the JKX and MXX geometries; they share training code but not input shapes.
- Be careful with leakage: clustering and train/val/test splits are explicitly date-bounded.
- If you change target or weight column names, update both `dataset.py` and all runner CLIs.
