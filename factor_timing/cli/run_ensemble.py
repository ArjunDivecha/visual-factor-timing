"""
=============================================================================
SCRIPT NAME: run_ensemble.py
=============================================================================

DESCRIPTION:
Runs the full 64-combo IC-weighted ensemble for one (panel × window). Loops
all combinations of (arch × image × loss × weight × target), each with
n_folds seed fits, reusing the single-combo machinery. After all 64 combos
finish, aggregates their per-combo in-sample ICs to build three final
forecasts:

  MXX-only ensemble       IC-weighted over the 32 MXX combos
  JKX-only ensemble       IC-weighted over the 32 JKX combos
  Combined ensemble       IC-weighted over all 64 combos

Emits three excel files in T2_Optimizer format with sheets
Monthly_Net_Returns / Omega_Weights / Expected_Return.

All 64 base combos log to W&B as individual runs grouped under a common
group name.  A final aggregation run logs the three ensemble-level summary
metrics (OOS IC, HML Sharpe, n_factors_out).

VERSION: 1.0
LAST UPDATED: 2026-04-22

USAGE:
    python -m factor_timing.cli.run_ensemble --panel t2 --window 12
=============================================================================
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from factor_timing.cli.run_single import (
    RunSpec, RUNS_ROOT, _write_excel, DATA_ROOT,
)
from factor_timing.train.loop     import _default_device

log = logging.getLogger(__name__)

ARCHES   = ["cnn", "cnn_lstm"]
IMAGES   = ["mxx", "jkx"]
LOSSES   = ["mse", "mae"]
WEIGHTS  = ["weight_ew", "weight_ewpm"]
TARGETS  = ["raw", "sigma", "norm", "pct"]


def _combos():
    for a, i, l, w, t in itertools.product(ARCHES, IMAGES, LOSSES, WEIGHTS, TARGETS):
        yield {"arch": a, "image": i, "loss": l, "weight": w, "target": t}


def _combo_key(c: dict) -> str:
    return f"{c['arch']}_{c['image']}_{c['loss']}_{c['weight']}_{c['target']}"


def _median_omega(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally standardize, median-shift so monthly median = 1."""
    df = forecasts.copy()
    def _z(s):
        sd = s.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd
    df["z"]     = df.groupby("end_date")["f_hat"].transform(_z)
    df["omega"] = df.groupby("end_date")["z"].transform(lambda s: 1.0 + (s - s.median()))
    return df[["factor", "end_date", "f_hat", "omega"]]


def _ic_weighted(combo_df_list, keys):
    """Combine multiple per-combo (factor, end_date, f_hat, ic_train_mean) into
    one IC-weighted ensemble forecast."""
    frames = []
    for df, ic in zip(combo_df_list, keys):
        if ic is None or not np.isfinite(ic) or ic <= 0:
            continue
        d = df.copy()
        d["w"]  = ic
        d["wp"] = ic * d["f_hat"]
        frames.append(d[["factor", "end_date", "w", "wp"]])
    if not frames:
        return pd.DataFrame(columns=["factor", "end_date", "f_hat"])
    allp = pd.concat(frames, ignore_index=True)
    agg  = (allp.groupby(["factor", "end_date"])
                .agg(wp_sum=("wp", "sum"), w_sum=("w", "sum"))
                .reset_index())
    agg["f_hat"] = agg["wp_sum"] / agg["w_sum"].replace(0, np.nan)
    return agg[["factor", "end_date", "f_hat"]]


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel",     default="t2",  choices=["t2", "gdelt"])
    ap.add_argument("--window",    type=int, default=12)
    ap.add_argument("--n-folds",   type=int, default=5)
    ap.add_argument("--train-end", default="2018-12-31")
    ap.add_argument("--project",   default="visual-factor-timing")
    ap.add_argument("--group",     default=None, help="W&B group (default auto)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip combos whose forecasts.parquet already exists")
    args = ap.parse_args()

    device = _default_device()
    group = args.group or f"{args.panel}_w{args.window}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.environ.setdefault("WANDB_RUN_GROUP", group)

    agg_root = RUNS_ROOT / f"ENSEMBLE_{group}"
    agg_root.mkdir(parents=True, exist_ok=True)

    # Import run lazily so env is set first
    from factor_timing.cli.run_single import run as run_single

    combos = list(_combos())
    combo_results: list = []
    manifest_path = agg_root / "manifest.json"
    manifest = {
        "panel": args.panel, "window": args.window, "n_folds": args.n_folds,
        "train_end": args.train_end, "device": device, "group": group,
        "combos": [], "started": time.time(),
    }

    print(f"▶ ensemble group = {group}")
    print(f"  device = {device}   combos = {len(combos)}   folds/combo/factor = {args.n_folds}")
    print(f"  outputs will aggregate to: {agg_root}")

    for i, combo in enumerate(combos, start=1):
        key = _combo_key(combo)
        print(f"\n═══ {i}/{len(combos)}  {key} ═══")
        spec = RunSpec(
            panel=args.panel, window=args.window,
            arch=combo["arch"], image=combo["image"],
            loss=combo["loss"], weight=combo["weight"],
            target=combo["target"], n_folds=args.n_folds,
            train_end=args.train_end, device=device,
            wandb_project=args.project, wandb_enabled=True,
        )
        run_dir = run_single(spec)
        forecasts = pd.read_parquet(run_dir / "forecasts.parquet")
        # Mean in-sample IC across folds & factors — scalar weight for this combo
        prog = pd.read_json(run_dir / "progress.jsonl", lines=True)
        fits = prog[prog["event"] == "fit"]
        ic_weight = float(fits["ic_train"].mean()) if not fits.empty else float("nan")
        combo_results.append({
            "combo": combo, "key": key, "ic_train_mean": ic_weight,
            "forecasts": forecasts[["factor", "end_date", "f_hat"]].copy(),
            "run_dir": str(run_dir),
        })
        manifest["combos"].append({
            "key": key, "combo": combo, "ic_train_mean": ic_weight,
            "run_dir": str(run_dir),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    # ── Aggregate ─────────────────────────────────────────────────────────
    mxx = [r for r in combo_results if r["combo"]["image"] == "mxx"]
    jkx = [r for r in combo_results if r["combo"]["image"] == "jkx"]

    # Pull next-month label once for outputs
    idx_lbl = (pd.read_parquet(DATA_ROOT / f"cache_{args.panel}_w{args.window}" / "index.parquet")
                 [["factor", "end_date", "label_return"]])

    for name, subset in [("mxx", mxx), ("jkx", jkx), ("combined", combo_results)]:
        if not subset:
            continue
        fc = _ic_weighted([r["forecasts"] for r in subset],
                          [r["ic_train_mean"] for r in subset])
        if fc.empty:
            continue
        om = _median_omega(fc)
        out = om.merge(idx_lbl, on=["factor", "end_date"], how="left")
        out["timed_return"] = out["omega"] * out["label_return"]

        out.to_parquet(agg_root / f"ensemble_{name}.parquet", index=False)
        excel = agg_root / f"{args.panel.upper()}_w{args.window}_{name.upper()}.xlsx"
        _write_excel(out, excel, args.panel)

        # Metrics
        valid = out.dropna(subset=["f_hat", "label_return"])
        oos_ic = (float(np.corrcoef(valid["f_hat"], valid["label_return"])[0,1])
                  if len(valid) > 1 and valid["f_hat"].std() > 0 else float("nan"))
        qdf = out.dropna(subset=["omega", "label_return"]).copy()
        qdf["quintile"] = qdf.groupby("end_date")["omega"].transform(
            lambda s: pd.qcut(s, q=5, labels=False, duplicates="drop") if s.nunique() >= 5 else np.nan)
        hml = (qdf.groupby(["end_date", "quintile"])["label_return"].mean().unstack("quintile"))
        if hml.shape[1] >= 5:
            hml_series = hml[4.0] - hml[0.0]
            sharpe = float(hml_series.mean() / hml_series.std(ddof=1) * np.sqrt(12))
        else:
            sharpe = float("nan")

        print(f"\n  [{name}]  OOS IC={oos_ic:.4f}   HML Sharpe (ann)={sharpe:.3f}   factors={out['factor'].nunique()}   excel={excel.name}")
        manifest[f"{name}_oos_ic"] = oos_ic
        manifest[f"{name}_hml_sharpe"] = sharpe
        manifest[f"{name}_excel"] = str(excel)

    manifest["ended"] = time.time()
    manifest["elapsed_min"] = (manifest["ended"] - manifest["started"]) / 60.0
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n✔ ensemble done in {manifest['elapsed_min']:.1f} min  →  {agg_root}")


if __name__ == "__main__":
    main()
