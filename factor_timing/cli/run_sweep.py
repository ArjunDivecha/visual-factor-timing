"""
=============================================================================
SCRIPT NAME: run_sweep.py
=============================================================================

DESCRIPTION:
Runs multiple 64-combo ensembles sequentially and produces a combined
summary table with key metrics (OOS IC, HML Sharpe, Top-3 and Top-15
long-only Sharpes). Intended for hyperparameter sweeps over window and
train-end combinations.

Usage:
    python -m factor_timing.cli.run_sweep \
        --config factor_timing/sweeps/gdelt_window_train_end.yaml

Or with inline specs (simpler for one-off sweeps):
    python -m factor_timing.cli.run_sweep \
        --panel gdelt --n-folds 5 \
        --points w12:2022-03-31 w12:2023-03-31 w24:2022-03-31 w24:2023-03-31 w6:2023-03-31 w36:2023-03-31

After all runs complete, writes factor_timing/outputs/sweeps/{sweep_id}/
  summary.csv    per-run metrics (OOS IC, HML Sharpe, Top-K Sharpes)
  summary.xlsx   human-readable workbook

VERSION: 1.0
LAST UPDATED: 2026-04-23
=============================================================================
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from factor_timing.cli.run_single   import RunSpec, run as run_single, RUNS_ROOT
from factor_timing.cli.run_ensemble import _combos, _combo_key, _ic_weighted, _median_omega, DATA_ROOT, _write_excel


def _strategy_metrics(ensemble_df: pd.DataFrame) -> dict:
    """Compute IC, HML, Top-3 L/S, Top-15 L, Top-20 L metrics."""
    v = ensemble_df.dropna(subset=["f_hat", "label_return", "omega"])
    if len(v) < 10:
        return {}

    monthly_ic = v.groupby("end_date").apply(
        lambda g: np.corrcoef(g["f_hat"], g["label_return"])[0,1]
                  if len(g) > 2 and g["f_hat"].std() > 0 and g["label_return"].std() > 0 else np.nan,
        include_groups=False)
    rank_ic = v[["f_hat","label_return"]].rank().corr().iloc[0,1]

    def sharpe(r: pd.Series) -> Tuple[float, float, float]:
        r = r.dropna()
        if len(r) < 3: return float("nan"), float("nan"), float("nan")
        mean_ann = r.mean() * 12
        std_ann  = r.std(ddof=1) * np.sqrt(12)
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        return (mean_ann / std_ann if std_ann > 0 else float("nan")), t, mean_ann

    # HML quintile
    qdf = v.copy()
    qdf["q"] = qdf.groupby("end_date")["omega"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop") if s.nunique() >= 5 else np.nan)
    hml = qdf.groupby(["end_date","q"])["label_return"].mean().unstack("q")
    h_series = (hml[4.0] - hml[0.0]) if hml.shape[1] >= 5 else pd.Series(dtype=float)
    hml_sr, hml_t, hml_ret = sharpe(h_series)

    # Top-3 / Bot-3 L/S
    def ls3(g):
        if len(g) < 6: return np.nan
        return g.nlargest(3, "omega")["label_return"].mean() - g.nsmallest(3, "omega")["label_return"].mean()
    ls3_series = v.groupby("end_date", group_keys=False).apply(ls3, include_groups=False)
    ls3_sr, ls3_t, ls3_ret = sharpe(ls3_series)

    # Top-3 long only
    def top3(g):
        if len(g) < 3: return np.nan
        return g.nlargest(3, "omega")["label_return"].mean()
    top3_series = v.groupby("end_date", group_keys=False).apply(top3, include_groups=False)
    top3_sr, top3_t, top3_ret = sharpe(top3_series)

    # Top-15 long only
    def top15(g):
        if len(g) < 15: return np.nan
        return g.nlargest(15, "omega")["label_return"].mean()
    top15_series = v.groupby("end_date", group_keys=False).apply(top15, include_groups=False)
    top15_sr, top15_t, top15_ret = sharpe(top15_series)

    # Baseline EW
    base_series = v.groupby("end_date")["label_return"].mean()
    base_sr, base_t, base_ret = sharpe(base_series)

    return {
        "n_months":          v["end_date"].nunique(),
        "n_factors":         v["factor"].nunique(),
        "avg_month_IC":      round(monthly_ic.mean(), 4),
        "rank_IC":           round(rank_ic, 4),
        "baseline_ret%":     round(base_ret, 2),
        "baseline_Sharpe":   round(base_sr, 3),
        "HML_ret%":          round(hml_ret, 2),
        "HML_Sharpe":        round(hml_sr, 3),
        "HML_t":             round(hml_t, 2),
        "Top3_long_ret%":    round(top3_ret, 2),
        "Top3_long_Sharpe":  round(top3_sr, 3),
        "Top3_long_t":       round(top3_t, 2),
        "LS3_ret%":          round(ls3_ret, 2),
        "LS3_Sharpe":        round(ls3_sr, 3),
        "LS3_t":             round(ls3_t, 2),
        "Top15_long_ret%":   round(top15_ret, 2),
        "Top15_long_Sharpe": round(top15_sr, 3),
        "Top15_long_t":      round(top15_t, 2),
    }


def run_ensemble_for_point(panel: str, window: int, train_end: str,
                           n_folds: int, project: str) -> Path:
    """Run all 64 combos for one sweep point, aggregate MXX/JKX/Combined, return agg dir."""
    group = f"{panel}_w{window}_{train_end.replace('-','')}_{time.strftime('%H%M%S')}"
    import os
    os.environ["WANDB_RUN_GROUP"] = group
    agg_root = RUNS_ROOT / f"ENSEMBLE_{group}"
    agg_root.mkdir(parents=True, exist_ok=True)

    combos = list(_combos())
    combo_results = []

    print(f"\n{'='*78}")
    print(f"  ▶ sweep point  panel={panel}  window={window}  train_end={train_end}")
    print(f"  agg dir       {agg_root}")
    print(f"{'='*78}")

    for i, combo in enumerate(combos, start=1):
        print(f"\n  ─── combo {i}/64  {_combo_key(combo)} ───")
        spec = RunSpec(
            panel=panel, window=window,
            arch=combo["arch"], image=combo["image"],
            loss=combo["loss"], weight=combo["weight"],
            target=combo["target"], n_folds=n_folds,
            train_end=train_end, wandb_project=project, wandb_enabled=True,
        )
        run_dir = run_single(spec)
        forecasts = pd.read_parquet(run_dir / "forecasts.parquet")
        prog = pd.read_json(run_dir / "progress.jsonl", lines=True)
        fits = prog[prog["event"] == "fit"]
        ic_w = float(fits["ic_train"].mean()) if not fits.empty else float("nan")
        combo_results.append({
            "combo": combo, "key": _combo_key(combo), "ic_train_mean": ic_w,
            "forecasts": forecasts[["factor", "end_date", "f_hat"]].copy(),
            "run_dir": str(run_dir),
        })

    # Aggregate to MXX / JKX / Combined
    mxx = [r for r in combo_results if r["combo"]["image"] == "mxx"]
    jkx = [r for r in combo_results if r["combo"]["image"] == "jkx"]

    idx_lbl = (pd.read_parquet(DATA_ROOT / f"cache_{panel}_w{window}" / "index.parquet")
                 [["factor", "end_date", "label_return"]])

    summary = {}
    for name, subset in [("mxx", mxx), ("jkx", jkx), ("combined", combo_results)]:
        if not subset: continue
        fc = _ic_weighted([r["forecasts"] for r in subset],
                          [r["ic_train_mean"] for r in subset])
        if fc.empty: continue
        om = _median_omega(fc)
        out = om.merge(idx_lbl, on=["factor","end_date"], how="left")
        out["timed_return"] = out["omega"] * out["label_return"]
        out.to_parquet(agg_root / f"ensemble_{name}.parquet", index=False)
        _write_excel(out, agg_root / f"{panel.upper()}_w{window}_{train_end}_{name.upper()}.xlsx", panel)
        summary[name] = _strategy_metrics(out)
        print(f"\n  [{name}]  " + "  ".join(f"{k}={v}" for k,v in summary[name].items()
                                             if k in ("n_months","HML_Sharpe","Top3_long_Sharpe","Top15_long_Sharpe","avg_month_IC")))

    # Write manifest
    import json
    manifest = {
        "panel": panel, "window": window, "train_end": train_end,
        "n_folds": n_folds, "group": group,
        "summary": summary, "timestamp": time.time(),
    }
    (agg_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return agg_root


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="gdelt")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--project", default="visual-factor-timing")
    ap.add_argument("--points", nargs="+", required=True,
                    help="one or more window:train_end specs, e.g. w12:2023-03-31")
    ap.add_argument("--sweep-id", default=None)
    args = ap.parse_args()

    sweep_id = args.sweep_id or f"{args.panel}_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = Path("factor_timing/outputs/sweeps") / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in args.points:
        w_str, te = p.split(":")
        window = int(w_str.lstrip("w"))
        agg_dir = run_ensemble_for_point(args.panel, window, te, args.n_folds, args.project)
        import json
        m = json.load(open(agg_dir / "manifest.json"))
        for ens, metrics in m.get("summary", {}).items():
            row = {"point": p, "window": window, "train_end": te, "ensemble": ens}
            row.update(metrics)
            rows.append(row)
        # Write partial summary after each run (so the user can watch mid-sweep)
        pd.DataFrame(rows).to_csv(sweep_dir / "summary.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(sweep_dir / "summary.csv", index=False)
    with pd.ExcelWriter(sweep_dir / "summary.xlsx", engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="sweep_summary", index=False)
        # Pivots by ensemble
        for e in ("mxx","jkx","combined"):
            sub = df[df["ensemble"] == e]
            if not sub.empty:
                sub.to_excel(xw, sheet_name=f"by_{e}", index=False)

    print(f"\n\n{'='*78}")
    print(f"  ✔ sweep done  →  {sweep_dir}/summary.csv")
    print(f"{'='*78}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
