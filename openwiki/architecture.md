---
type: Reference
title: Architecture
description: High-level architecture of the factor_timing data-to-backtest pipeline, with a stage map and pointers to the detailed pipeline page.
openwiki:
  roles: [architecture, repository]
---

# Architecture

The repository is organized as a data-to-backtest pipeline for factor timing. The detailed stage-by-stage flow lives in [Pipeline architecture](architecture/pipeline.md); this page gives the overview and the cross-system wiring.

## Pipeline stages

### 1) Raw factor workbooks

The project starts from external Excel workbooks containing daily and monthly factor returns for the T2 and GDELT panels. `factor_timing/data/factor_loader.py` loads those files, compounds daily returns into synthetic prices, computes monthly OHLC bars, and attaches next-month returns as labels.

### 2) Cached learning inputs

Two cache builders turn the monthly OHLC into model-ready arrays:

- `factor_timing/imaging/build_cache.py` renders JKX-style OHLC images into a `uint8` memmap cache (`images.npy`) with an `index.parquet`.
- `factor_timing/imaging/build_mxx_cache.py` renders MXX cumulative-return trajectories into a `float32` array (`trajectories.npy`) with its own `index.parquet`.

`factor_timing/train/targets.py` then joins the JKX index with the next-month label and writes `targets.parquet` containing the four target columns and two weight columns.

`factor_timing/train/dataset.py` finally joins the index with `targets.parquet` and serves samples from either the JKX cache or the MXX trajectory cache.

### 3) Model families

The repository has separate architecture files for 1D and 2D branches (see [Models, targets, and objectives](domain/models-targets.md)):

- `factor_timing/models/cnn1d.py` — MXX trajectory branch, with `CNN1D` and `CNNLSTM1D` variants.
- `factor_timing/models/cnn2d.py` — JKX image branch, with `CNN2D` and `CNNLSTM2D` variants.

Both branches use small CNN encoders with a SiLU activation and a dense head whose width is inferred with a dummy forward pass.

### 4) Training loops

`factor_timing/train/loop.py` provides the standard per-sample regression loop, including Adam, ReduceLROnPlateau, early stopping with a `min_delta` threshold, and batched prediction.

Additional training modules:

- `factor_timing/train/monthly_dataset.py` groups samples by month for cross-sectional training.
- `factor_timing/train/clustering.py` provides expanding-window Ward clustering for pooled GDELT training (leakage-aware — clusters are refit at each retrain date using only data available before that date).

### 5) Experiment runners

The CLI layer orchestrates the training/evaluation combinations (see [Workflow runners](workflows/runners.md)):

- `run_single.py` — one factor-timing configuration over all factors.
- `run_ensemble.py` — 64-combo IC-weighted ensemble aggregation.
- `run_sweep.py` — repeat ensembles over multiple window/train-end points and collect strategy metrics.

`run_single` is the atomic unit reused by `run_ensemble`, which in turn is reused by `run_sweep`.

### 6) Monitoring and outputs

`factor_timing/dashboard/build.py` reads `progress.jsonl` and `summary.json` from a run directory and writes a self-contained `dashboard.html`. The output tree under `factor_timing/outputs/` stores run artifacts, sweeps, and caches (see [Outputs and operational caveats](operations/outputs-and-caveats.md)).

## Why the code is structured this way

The repository reflects two competing needs:

- **Research fidelity** — reproduce paper-style workflows with explicit image-based factor timing and IC-weighted aggregation.
- **Practical experimentation** — keep a modular training loop so new targets, losses, or strategies can be swapped in without rewriting data preparation.

The codebase has evolved from single-run factor loaders and JKX image caches into ensemble orchestration, stronger early stopping, larger windows, and cross-sectional dataset support.

## Source map

- Data prep: `factor_timing/data/factor_loader.py`
- Image caching: `factor_timing/imaging/build_cache.py`
- Trajectory caching: `factor_timing/imaging/build_mxx_cache.py`
- Target/weight builder: `factor_timing/train/targets.py`
- Trajectory/image datasets: `factor_timing/train/dataset.py`
- Standard training loop: `factor_timing/train/loop.py`
- Ensemble logic: `factor_timing/train/ensemble.py`
- Cross-sectional training: `factor_timing/train/monthly_dataset.py`
- Clustering: `factor_timing/train/clustering.py`
- CLI runners: `factor_timing/cli/*.py`
- Dashboard: `factor_timing/dashboard/build.py`
