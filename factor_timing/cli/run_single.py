"""
=============================================================================
SCRIPT NAME: run_single.py
=============================================================================

DESCRIPTION:
Orchestrates a single-combo end-to-end factor-timing run.

Defaults (matching the first pilot the user asked for):
  panel=t2  window=12  arch=cnn  image=mxx  loss=mse  weight=weight_ew
  target=raw  n_folds=5  train_end=2018-12-31  val_frac=0.15

Training is per-factor (T2 convention).  Each factor is fit n_folds times
with different seeds.  IC-weighted averaging of the n_folds forecasts
produces f_hat per (factor, end_date); cross-sectional z-score + median
shift turns f_hat into ω.

Writes progress to factor_timing/outputs/runs/{run_id}/progress.jsonl
(appended after every fold so the dashboard can tail it).  Final outputs
go to the same run directory:
    summary.json          run metadata
    forecasts.parquet     long-form (factor, end_date, f_hat, omega, timed_return)
    excel/<run_id>.xlsx   T2_Optimizer-style wide workbook with 3 sheets

VERSION: 1.0
LAST UPDATED: 2026-04-22

USAGE:
    python -m factor_timing.cli.run_single            # defaults
    python -m factor_timing.cli.run_single --n-folds 1
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Optional W&B integration — enabled when WANDB_API_KEY is set
try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

from factor_timing.train.dataset  import FactorImageDataset
from factor_timing.train.loop     import TrainConfig, train_one, predict, _default_device
from factor_timing.train.ensemble import (
    _build_model,
    aggregate_forecasts,
    omega_from_forecasts,
    FoldResult,
)

log = logging.getLogger(__name__)

DATA_ROOT = Path("factor_timing/outputs")
RUNS_ROOT = Path("factor_timing/outputs/runs")
MONTHLY_PANEL = {
    "t2":    "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy/T2_Optimizer.xlsx",
    "gdelt": "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_Optimizer.xlsx",
}


@dataclass
class RunSpec:
    panel:      str   = "t2"
    window:     int   = 12
    arch:       str   = "cnn"
    image:      str   = "mxx"
    loss:       str   = "mse"
    weight:     str   = "weight_ew"
    target:     str   = "raw"
    n_folds:    int   = 5
    train_end:  str   = "2018-12-31"
    val_frac:   float = 0.15      # of training window
    max_epochs: int   = 200       # let the model train longer
    patience:   int   = 15        # require a real plateau before stopping
    batch_size: int   = 256       # smaller than paper's 2^15 because dataset is only ~thousands
    device:     Optional[str] = None
    wandb_project: str = "visual-factor-timing"
    wandb_enabled: bool = True


def _append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _image_hw(window: int) -> tuple:
    return (72, 3 * window)


def _date_before(d, cutoff): return d <= cutoff
def _date_between(d, lo, hi): return (d > lo) and (d <= hi)
def _date_after(d, cutoff):  return d > cutoff


def train_factor_folds(
    spec: RunSpec,
    factor: str,
    train_cutoff: pd.Timestamp,
    val_cutoff:   pd.Timestamp,
    test_cutoff:  pd.Timestamp,
    train_cfg: TrainConfig,
    progress_log: Path,
    run_id: str,
    pbar: tqdm,
    wb=None,
    fit_counter_ref=None,
) -> list:
    """Train n_folds models on this factor; return per-fold FoldResults."""
    results = []
    hw = _image_hw(spec.window)

    # Build datasets once (they reuse the same memmap)
    train_ds = FactorImageDataset(spec.panel, spec.window, spec.image,
        target_col=spec.target, weight_col=spec.weight,
        factor_filter={factor},
        date_filter=lambda d, c=train_cutoff: _date_before(d, c))
    val_ds = FactorImageDataset(spec.panel, spec.window, spec.image,
        target_col=spec.target, weight_col=spec.weight,
        factor_filter={factor},
        date_filter=lambda d, a=train_cutoff, b=val_cutoff: _date_between(d, a, b))
    test_ds = FactorImageDataset(spec.panel, spec.window, spec.image,
        target_col=spec.target, weight_col=spec.weight,
        factor_filter={factor},
        date_filter=lambda d, c=val_cutoff: _date_after(d, c))

    if len(train_ds) < 20 or len(val_ds) < 3 or len(test_ds) == 0:
        _append_log(progress_log, {
            "ts": time.time(), "run_id": run_id, "factor": factor,
            "event": "skip",
            "reason": f"thin splits: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        })
        pbar.update(spec.n_folds)
        return results

    for fold in range(spec.n_folds):
        t0 = time.time()
        torch.manual_seed(fold)
        np.random.seed(fold)

        model = _build_model(spec.image, spec.arch, spec.window, hw)
        fold_cfg = TrainConfig(
            lr=train_cfg.lr, beta1=train_cfg.beta1, beta2=train_cfg.beta2, eps=train_cfg.eps,
            batch_size=train_cfg.batch_size, max_epochs=train_cfg.max_epochs,
            patience=train_cfg.patience, loss=spec.loss, device=train_cfg.device,
        )
        model, stats = train_one(model, train_ds, val_ds, fold_cfg, seed=fold)

        # IC on training set (model weight later)
        pred_tr = predict(model, train_ds, batch_size=train_cfg.batch_size, device=train_cfg.device)
        ytr = train_ds.meta[spec.target].astype(float).values
        ic_tr = float(np.corrcoef(pred_tr, ytr)[0,1]) if len(pred_tr) > 1 and np.std(pred_tr) > 0 else float("nan")

        pred_test = predict(model, test_ds, batch_size=train_cfg.batch_size, device=train_cfg.device)
        fr = FoldResult(
            combo={"arch": spec.arch, "image": spec.image, "loss": spec.loss,
                   "weight": spec.weight, "target": spec.target},
            fold_id=fold, ic_train=ic_tr,
            pred_test=pred_test,
            test_meta=test_ds.meta[["factor", "end_date"]].copy(),
        )
        results.append(fr)

        dt = time.time() - t0
        _append_log(progress_log, {
            "ts": time.time(), "run_id": run_id, "factor": factor, "fold": fold,
            "event": "fit",
            "ic_train":   ic_tr,
            "val_loss":   stats["best_val_loss"],
            "epochs":     stats["epochs"],
            "best_epoch": stats.get("best_epoch", -1),
            "final_lr":   stats.get("final_lr", float("nan")),
            "wall_s":     dt,
        })
        if wb is not None:
            if fit_counter_ref is not None:
                fit_counter_ref[0] += 1
                step = fit_counter_ref[0]
            else:
                step = None
            wb.log({
                "fit/ic_train":    ic_tr,
                "fit/val_loss":    stats["best_val_loss"],
                "fit/epochs":      stats["epochs"],
                "fit/best_epoch":  stats.get("best_epoch", -1),
                "fit/final_lr":    stats.get("final_lr", float("nan")),
                "fit/wall_s":      dt,
                "fit/factor":      factor,
                "fit/fold":        fold,
            }, step=step)
        pbar.update(1)

    return results


def run(spec: RunSpec) -> Path:
    device = spec.device or _default_device()
    run_id = f"{spec.panel}_w{spec.window}_{spec.arch}_{spec.image}_{spec.loss}_{spec.weight}_{spec.target}_f{spec.n_folds}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log = run_dir / "progress.jsonl"

    train_cutoff = pd.Timestamp(spec.train_end)
    # Val = last val_frac of training window; easiest: val is a date range
    # preceding the test period. We'll compute val_cutoff so that
    # (val_cutoff - some offset) ≤ train_cutoff ≤ val_cutoff, using a
    # convention of val_frac of the total pre-test history as validation.
    # Simpler: treat train_cutoff as end-of-training-and-val, and carve a
    # val slice off the tail of the training window.
    # For the pilot we use: train ≤ val_start, val in (val_start, train_cutoff],
    # test in (train_cutoff, ...].
    # So redefine:
    idx = pd.read_parquet(DATA_ROOT / f"cache_{spec.panel}_w{spec.window}" / "index.parquet")
    all_dates = sorted(idx["end_date"].unique())
    n_pre = sum(d <= train_cutoff for d in all_dates)
    val_start_idx = int(n_pre * (1 - spec.val_frac))
    val_start = all_dates[val_start_idx - 1]     # last training date
    val_end   = train_cutoff                     # paper's train_cutoff becomes val_end
    test_cutoff = pd.Timestamp("2099-12-31")     # run to end of series

    cfg = TrainConfig(
        batch_size=spec.batch_size, max_epochs=spec.max_epochs,
        patience=spec.patience, loss=spec.loss, device=device,
    )

    factors = sorted(idx["factor"].unique())
    total_fits = len(factors) * spec.n_folds

    meta = asdict(spec)
    meta.update({
        "run_id": run_id, "device": device,
        "train_cutoff": str(val_start.date()),   # data available for training ends here
        "val_cutoff":   str(val_end.date()),     # validation ends here
        "n_factors":    len(factors),
        "total_fits":   total_fits,
        "started":      time.time(),
    })
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2, default=str))

    # ── Initialise W&B run (optional) ───────────────────────────────────
    wb = None
    if spec.wandb_enabled and _HAS_WANDB and os.environ.get("WANDB_API_KEY"):
        wb = wandb.init(
            project=spec.wandb_project,
            name=run_id,
            config=meta,
            dir=str(run_dir),
            reinit=True,
        )
        print(f"  wandb: {wb.url}")

    print(f"▶ run_id = {run_id}")
    print(f"  device = {device}")
    print(f"  factors = {len(factors)}   folds/factor = {spec.n_folds}   total fits = {total_fits}")
    print(f"  train ≤ {val_start.date()}   val ≤ {val_end.date()}   test > {val_end.date()}")
    print(f"  progress log: {progress_log}")

    all_results: list = []
    fit_counter = [0]
    pbar = tqdm(total=total_fits, desc="fits", ncols=90)
    for fac in factors:
        frs = train_factor_folds(
            spec, fac, val_start, val_end, test_cutoff, cfg, progress_log, run_id, pbar,
            wb=wb, fit_counter_ref=fit_counter,
        )
        all_results.extend(frs)
    pbar.close()

    # Aggregate
    forecasts = aggregate_forecasts(all_results)
    om = omega_from_forecasts(forecasts)

    # Attach realized next-month return (label_return from the index parquet)
    idx_lbl = idx[["factor", "end_date", "label_return"]]
    out = om.merge(idx_lbl, on=["factor", "end_date"], how="left")
    out["timed_return"] = out["omega"] * out["label_return"]
    out.to_parquet(run_dir / "forecasts.parquet", index=False)

    # Write T2_Optimizer-style excel
    excel_path = run_dir / f"{run_id}.xlsx"
    _write_excel(out, excel_path, spec.panel)

    meta["ended"]       = time.time()
    meta["elapsed_min"] = (meta["ended"] - meta["started"]) / 60.0
    meta["excel_path"]  = str(excel_path)
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2, default=str))

    # Cross-factor summary metrics
    n_factors_out = out["factor"].nunique()
    n_months_out  = out["end_date"].nunique()
    # Out-of-sample IC: correlation of f_hat with realized next-month return
    valid = out.dropna(subset=["f_hat", "label_return"])
    oos_ic = (float(np.corrcoef(valid["f_hat"], valid["label_return"])[0, 1])
              if len(valid) > 1 and valid["f_hat"].std() > 0 else float("nan"))
    # Simple high-vs-low quintile on omega, equal-weighted monthly returns
    qdf = out.dropna(subset=["omega", "label_return"]).copy()
    qdf["quintile"] = qdf.groupby("end_date")["omega"].transform(
        lambda s: pd.qcut(s, q=5, labels=False, duplicates="drop") if s.nunique() >= 5 else np.nan
    )
    hml = (qdf.groupby(["end_date", "quintile"])["label_return"].mean()
               .unstack("quintile"))
    if hml.shape[1] >= 5:
        hml_series = hml[4.0] - hml[0.0]
        sharpe = float(hml_series.mean() / hml_series.std(ddof=1) * np.sqrt(12))
    else:
        hml_series = pd.Series(dtype=float)
        sharpe = float("nan")

    if wb is not None:
        wb.summary.update({
            "oos_ic":           oos_ic,
            "hml_sharpe_ann":   sharpe,
            "n_factors_out":    n_factors_out,
            "n_months_out":     n_months_out,
            "elapsed_min":      meta["elapsed_min"],
        })
        # Log the full excel as an artifact
        art = wandb.Artifact(name=f"{run_id}_outputs", type="factor_timing_run")
        art.add_file(str(excel_path))
        art.add_file(str(run_dir / "forecasts.parquet"))
        wb.log_artifact(art)
        wb.finish()

    print(f"\n✔ done in {meta['elapsed_min']:.1f} min")
    print(f"  oos IC (f_hat vs realized): {oos_ic:.4f}")
    print(f"  HML Sharpe (annualized):    {sharpe:.3f}")
    print(f"  forecasts: {run_dir/'forecasts.parquet'}")
    print(f"  excel:     {excel_path}")
    return run_dir


def _write_excel(out: pd.DataFrame, path: Path, panel: str) -> None:
    """Wide workbook matching T2_Optimizer schema. Date col = first-of-month.

    The cache uses month-end dates; the optimizer file uses first-of-month.
    We convert so the output merges cleanly with the existing optimizer.
    """
    out = out.copy()
    # end_date is the forecast-making month-end (t). The label is next-month.
    # Output Date is first-of-month for month (t+1), matching T2_Optimizer convention.
    out["Date"] = out["end_date"] + pd.offsets.MonthBegin(1)

    def _pivot(col: str) -> pd.DataFrame:
        w = out.pivot_table(index="Date", columns="factor", values=col, aggfunc="first")
        w.index.name = "Date"
        return w.sort_index().reset_index()

    timed = _pivot("timed_return")
    omega = _pivot("omega")
    fhat  = _pivot("f_hat")

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        timed.to_excel(xw, sheet_name="Monthly_Net_Returns", index=False)
        omega.to_excel(xw, sheet_name="Omega_Weights",       index=False)
        fhat.to_excel(xw,  sheet_name="Expected_Return",     index=False)
    log.info("Wrote excel %s  shape (timed)=%s", path, timed.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel",   default="t2",  choices=["t2", "gdelt"])
    ap.add_argument("--window",  type=int, default=12)
    ap.add_argument("--arch",    default="cnn", choices=["cnn", "cnn_lstm"])
    ap.add_argument("--image",   default="mxx", choices=["mxx", "jkx"])
    ap.add_argument("--loss",    default="mse", choices=["mse", "mae"])
    ap.add_argument("--weight",  default="weight_ew", choices=["weight_ew", "weight_ewpm"])
    ap.add_argument("--target",  default="raw", choices=["raw", "sigma", "norm", "pct"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--train-end", default="2018-12-31")
    ap.add_argument("--device",  default=None, help="override device (cpu/mps/cuda)")
    args = ap.parse_args()

    spec = RunSpec(
        panel=args.panel, window=args.window, arch=args.arch, image=args.image,
        loss=args.loss, weight=args.weight, target=args.target,
        n_folds=args.n_folds, train_end=args.train_end, device=args.device,
    )
    run(spec)
