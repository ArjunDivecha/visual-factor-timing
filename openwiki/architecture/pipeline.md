---
type: Reference
title: Pipeline architecture
description: Stage-by-stage data-to-signal pipeline for factor_timing, from raw optimizer workbooks through cached inputs, datasets, training, and IC-weighted forecast aggregation into monthly omega timing weights.
openwiki:
  roles: [architecture, domain, workflow]
  source_paths: [factor_timing/data/factor_loader.py, factor_timing/imaging/build_cache.py, factor_timing/imaging/build_mxx_cache.py, factor_timing/train/targets.py, factor_timing/train/dataset.py, factor_timing/train/loop.py, factor_timing/train/ensemble.py, factor_timing/train/clustering.py, factor_timing/train/monthly_dataset.py]
  symbols: [build_monthly_ohlc, build_cache, build_mxx_cache, build_targets, FactorImageDataset, MonthlyDataset, split_dates, TrainConfig, train_one, predict, aggregate_forecasts, omega_from_forecasts, all_model_combos, _build_model, train_fold, cluster_at, expanding_window_clusters]
  invariants: ["(factor, end_date) alignment is the shared key across OHLC parquet, both caches, targets, and dataset metadata", "JKX images are uint8 (N,1,72,3*window); MXX trajectories are float32 (N,window)", "Both caches drop the last month per factor (NaN label) and samples with <window months of history", "Clustering and train/val/test splits are date-bounded to avoid look-ahead leakage", "IC weights are clipped at 0 (non-positive IC combos contribute nothing to the ensemble)"]
---

# Pipeline architecture

## Overview
The repository's core pipeline is:
1. build monthly factor panels from external optimizer spreadsheets,
2. convert them into cacheable model inputs (JKX images and MXX trajectories) plus a target table,
3. load the cached inputs through dataset wrappers,
4. train neural nets on either per-sample or cross-sectional objectives,
5. aggregate forecasts into timing weights and evaluation outputs.

This flow is implemented in small, composable modules rather than one monolithic training script.

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    A["external xlsx<br/>daily + monthly optimizer"] --> B["factor_loader.build_monthly_ohlc<br/>{panel}_monthly_ohlc.parquet"]
    B --> C["build_cache.build_cache<br/>JKX uint8 images.npy + index.parquet"]
    B --> D["build_mxx_cache.build_mxx_cache<br/>MXX float32 trajectories.npy + index.parquet"]
    C --> E["targets.build_targets<br/>targets.parquet (raw,sigma,norm,pct,weight_ew,weight_ewpm)"]
    C --> F["FactorImageDataset<br/>(JKX branch)"]
    E --> F
    D --> G["FactorImageDataset<br/>(MXX branch)"]
    E --> G
    F --> H["loop.train_one<br/>per-sample regression"]
    G --> H
    H --> I["ensemble.aggregate_forecasts<br/>+ omega_from_forecasts"]
    I --> J["f_hat, omega, timed_return"]
    B --> K["clustering.cluster_at<br/>(GDELT pooled, leakage-aware)"]
    K --> H
```

## Data construction
`factor_timing/data/factor_loader.py` is the upstream normalization stage. It:
- loads daily and monthly Excel workbooks (sheet `Monthly_Net_Returns` in both, despite the name),
- compounds daily percentage returns (stored in percent, divided by 100) into a synthetic $1-indexed price series,
- computes monthly OHLC bars where `Volume` is monthly realized volatility (std of non-zero daily returns, with structural zero-return non-trading days filtered out),
- aligns next-month returns from the monthly file as the `Return` label column,
- drops months with fewer than two trading days,
- writes panel-specific parquet outputs such as `t2_monthly_ohlc.parquet` and `gdelt_monthly_ohlc.parquet`.

Why it exists:
- the original spreadsheets are not model-ready,
- the monthly OHLC/vol tables become the common input for both image branches and the target builder.

## Cache generation
There are two cache builders, each producing a parallel cache that is indexed 1:1 by `(factor, end_date)`:
- `factor_timing/imaging/build_cache.py` for JKX-style image caches. Images are `uint8` arrays shaped `(N, 1, H, W)` with `H=72` and `W=3*window`, stored as `images.npy` under `cache_{panel}_w{window}/`. Rendering uses the external `pattern.imaging.renderer.render_window` JKX renderer (imported via an absolute Dropbox `sys.path` insert).
- `factor_timing/imaging/build_mxx_cache.py` for MXX trajectory caches. Trajectories are `float32` arrays shaped `(N, window)` where each row is `CFR_{j,k} = Close[anchor+k-1]/Close[anchor] - 1` (cumulative return from the window's first month), stored as `trajectories.npy` under `cache_{panel}_w{window}_mxx/`.

Both caches write an `index.parquet` with `(factor, end_date, label_return, row)` so the `row` column aligns a metadata record to its array row. Samples with fewer than `window` months of history or a missing next-month label are skipped. Cross-sectional standardization is not baked into the MXX cache so different fold/retrain configurations reuse the same cache.

## Target and weight construction
`factor_timing/train/targets.py` reads the JKX `index.parquet`, renames `label_return` to `raw`, and writes `targets.parquet` with the four target transforms and two weight columns consumed by `FactorImageDataset`:
- `raw` — the label itself.
- `sigma` — `(r - trailing_expanding_factor_mean) / cross_sectional_std`, backward-looking only.
- `norm` — `Phi^-1(rank / (N + 1))` using `scipy.stats.norm.ppf`, computed across valid factors each month.
- `pct` — `rank / (N + 1)`.
- `weight_ew` — uniform (1.0).
- `weight_ewpm` — equal weight per month, equal weight per factor within month, rescaled to preserve the EW effective sample size.

Months with fewer than three valid factors are dropped from the cross-sectional transforms. See [Models, targets, and objectives](../domain/models-targets.md) for how these axes participate in the 64-combo grid.

## Dataset layer
`factor_timing/train/dataset.py` wraps the caches into `FactorImageDataset`.
It returns `(X, y, w)` tuples and supports:
- JKX or MXX inputs (JKX is loaded as `(1, H, W)` and divided by 255; MXX is loaded as `(T,)` and unsqueezed to `(T, 1)`),
- target selection (`raw`, `sigma`, `norm`, `pct`),
- sample-weight selection (`weight_ew`, `weight_ewpm`),
- `factor_filter` and `date_filter` closures for train/val/test splits or cluster/ factor-level pooling.

`factor_timing/train/monthly_dataset.py` groups samples by month for cross-sectional training. `MonthlyDataset` returns one whole month per `__getitem__`, and `collate_months` pads ragged month sizes and emits a binary mask so losses can operate on full cross-sections.

## Training layer
`factor_timing/train/loop.py` is the standard regression loop.
It provides:
- `TrainConfig` (runtime defaults differ from the module docstring: `max_epochs=200`, `patience=15`, `min_delta=1e-4`, `lr_reduce_patience=4`, `batch_size=32768` capped by dataset size),
- `train_one()` with Adam, weighted MSE/MAE, ReduceLROnPlateau, and early stopping that requires a `min_delta` improvement to reset the counter,
- `predict()` for batched inference.

## Ensemble layer
`factor_timing/train/ensemble.py` defines the combinatorial experiment grid:
- 2 architectures (`cnn`, `cnn_lstm`),
- 2 image types (`mxx`, `jkx`),
- 2 losses (`mse`, `mae`),
- 2 weighting schemes (`weight_ew`, `weight_ewpm`),
- 4 target transforms (`raw`, `sigma`, `norm`, `pct`).

That yields 64 base model combinations via `all_model_combos()`. The ensemble logic:
- trains folds for each combination (`train_fold` reuses `FactorImageDataset` with date-bounded filters and per-fold seeds),
- computes in-sample IC on the training set as a model weight,
- `aggregate_forecasts` drops non-positive or invalid IC weights (clips to 0),
- averages surviving forecasts into `f_hat`,
- `omega_from_forecasts` z-scores `f_hat` cross-sectionally each month then median-shifts so each month's median `omega` equals 1; months with fewer than two valid forecasts get `omega = 1.0`.

## Pooled GDELT training
`factor_timing/train/clustering.py` is used for leakage-aware pooled training.
It computes expanding-window Ward clusters from historical factor-return correlations (`cluster_at` uses data strictly before `as_of`; `expanding_window_clusters` iterates over retrain dates). Distance is angular on `|correlation|`: `D_ij = sqrt(2*(1 - |rho_ij|))`. Factors failing the `min_factor_coverage` (default 0.8) filter at an epoch are omitted for that epoch.

This matters because the GDELT panel is trained in pooled groups, while T2 is trained per factor.

## Model definitions
The model branch is split by input geometry:
- `factor_timing/models/cnn1d.py` handles MXX trajectories with `CNN1D` and `CNNLSTM1D` variants (Conv1D 64 filters, kernel 5, MaxPool 2, SiLU, dropout 0.25).
- `factor_timing/models/cnn2d.py` handles JKX images with `CNN2D` and `CNNLSTM2D` variants (Conv2D 64 filters, 5x5 kernel, MaxPool 2x2, SiLU, dropout 0.25; the CNN-LSTM variant runs an LSTM along the image-width axis).

Both model modules size linear layers with a dummy forward pass, so the same model classes adapt to window length.

## What to watch when editing this area
- Keep `(factor, end_date)` alignment consistent across OHLC parquet, both caches, targets, dataset metadata, and labels.
- Avoid mixing the JKX and MXX geometries; they share training code but not input shapes or dtype (uint8 vs float32).
- Be careful with leakage: clustering and train/val/test splits are explicitly date-bounded.
- If you change target or weight column names, update `targets.py`, `dataset.py`, and every runner CLI.
- `build_cache.py` depends on the external `pattern.imaging.renderer` module via an absolute path insert; that renderer is not vendored in this repo.
