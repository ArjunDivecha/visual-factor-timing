"""
=============================================================================
SCRIPT NAME: factor_timing/dashboard/build.py
=============================================================================

DESCRIPTION:
Builds a self-contained HTML dashboard for a single training run by tailing
its progress.jsonl.  Re-run this script (or let it auto-refresh via a
<meta http-equiv="refresh"> tag) to see updates during training.

USAGE:
    # watch-mode: rebuilds every 5 seconds
    python -m factor_timing.dashboard.build --run-id <run_id> --watch

    # one-shot:
    python -m factor_timing.dashboard.build --run-id <run_id>

The dashboard displays:
  - progress bar (fits done / total from summary.json)
  - ETA based on median wall_s from the log
  - live bar chart of training IC per factor (mean across completed folds)
  - histogram of val_loss
  - recent-fits table

OUTPUT:
    factor_timing/outputs/runs/<run_id>/dashboard.html

VERSION: 1.0
LAST UPDATED: 2026-04-22

DEPENDENCIES:
- pandas, plotly
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"

RUNS_ROOT = Path("factor_timing/outputs/runs")
REFRESH_SECONDS = 5


def _read_log(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "progress.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _read_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def build_dashboard(run_id: str, auto_refresh: bool = True) -> Path:
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    summary = _read_summary(run_dir)
    log_df  = _read_log(run_dir)

    total   = int(summary.get("total_fits", 0) or 0)
    started = float(summary.get("started",   0) or 0)
    done    = int((log_df["event"] == "fit").sum()) if "event" in log_df.columns else 0
    elapsed = (time.time() - started) if started else 0
    if done > 0 and elapsed > 0 and total > 0:
        eta_s = elapsed / done * (total - done)
    else:
        eta_s = float("nan")
    pct = (done / total * 100) if total else 0.0

    # Charts
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None],
               [{}, {}]],
        subplot_titles=(
            "Training IC per factor (mean over completed folds)",
            "Validation-loss distribution",
            "Wall time per fit (seconds)",
        ),
        vertical_spacing=0.15, horizontal_spacing=0.10,
    )

    fits = log_df[log_df.get("event") == "fit"] if "event" in log_df.columns else pd.DataFrame()
    if not fits.empty:
        by_factor = (fits.groupby("factor")["ic_train"]
                         .mean()
                         .sort_values(ascending=False))
        colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in by_factor.values]
        fig.add_trace(go.Bar(x=by_factor.index, y=by_factor.values,
                             marker_color=colors, name="mean IC"),
                      row=1, col=1)
        fig.update_yaxes(title_text="IC",   row=1, col=1)
        fig.update_xaxes(tickangle=-45,     row=1, col=1)

        fig.add_trace(go.Histogram(x=fits["val_loss"], nbinsx=30,
                                   marker_color="#264653", name="val_loss"),
                      row=2, col=1)
        fig.update_xaxes(title_text="val_loss",  row=2, col=1)
        fig.update_yaxes(title_text="count",     row=2, col=1)

        fig.add_trace(go.Histogram(x=fits["wall_s"], nbinsx=30,
                                   marker_color="#e9c46a", name="wall_s"),
                      row=2, col=2)
        fig.update_xaxes(title_text="seconds",   row=2, col=2)
        fig.update_yaxes(title_text="count",     row=2, col=2)

    fig.update_layout(
        height=720, showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
        title=f"{run_id} — {done}/{total} fits ({pct:.1f}%)",
    )

    # Recent-fits table
    if not fits.empty:
        recent = (fits.sort_values("ts", ascending=False)
                      .head(20)[["ts", "factor", "fold", "ic_train", "val_loss",
                                  "epochs", "wall_s"]])
        recent["ts"] = pd.to_datetime(recent["ts"], unit="s").dt.strftime("%H:%M:%S")
        recent_html = recent.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    else:
        recent_html = "<i>no fits logged yet</i>"

    # HTML page
    meta_refresh = f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">' if auto_refresh else ""
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  {meta_refresh}
  <title>{run_id}</title>
  <style>
    body {{ font-family: -apple-system, Helvetica, Arial, sans-serif;
            max-width: 1300px; margin: 20px auto; padding: 0 20px; color: #222; background:#fff; }}
    h1 {{ margin-bottom: 0; }}
    .meta {{ color:#666; font-size:13px; margin-bottom:20px; }}
    .progress {{ background:#eee; height:24px; border-radius:4px; overflow:hidden; margin:10px 0; }}
    .progress-bar {{ height:100%; background:#2a9d8f; text-align:center; color:white; line-height:24px; font-size:13px; }}
    table {{ border-collapse: collapse; font-size:13px; width: 100%; }}
    th, td {{ padding:4px 8px; border-bottom:1px solid #eee; text-align:left; }}
    th {{ background:#f8f8f8; }}
    .kpis {{ display:flex; gap:30px; margin:12px 0; }}
    .kpi {{ background:#f4f6f8; padding:10px 20px; border-radius:6px; }}
    .kpi .v {{ font-size:20px; font-weight:600; color:#264653; }}
    .kpi .l {{ font-size:11px; text-transform:uppercase; color:#888; letter-spacing:0.5px; }}
  </style>
</head>
<body>
<h1>{run_id}</h1>
<div class="meta">
  panel={summary.get("panel","?")} · window={summary.get("window","?")}
  · arch={summary.get("arch","?")} · image={summary.get("image","?")}
  · loss={summary.get("loss","?")} · weight={summary.get("weight","?")}
  · target={summary.get("target","?")} · folds={summary.get("n_folds","?")}
  · device={summary.get("device","?")}
  · train ≤ {summary.get("train_cutoff","?")} · val ≤ {summary.get("val_cutoff","?")}
</div>

<div class="progress">
  <div class="progress-bar" style="width:{pct:.2f}%">{pct:.1f}%</div>
</div>

<div class="kpis">
  <div class="kpi"><div class="v">{done}/{total}</div><div class="l">fits</div></div>
  <div class="kpi"><div class="v">{elapsed/60:.1f} min</div><div class="l">elapsed</div></div>
  <div class="kpi"><div class="v">{(eta_s/60 if np.isfinite(eta_s) else 0):.1f} min</div><div class="l">ETA</div></div>
  <div class="kpi"><div class="v">{(fits['ic_train'].mean() if not fits.empty else float('nan')):.3f}</div><div class="l">mean IC</div></div>
  <div class="kpi"><div class="v">{(fits['wall_s'].median() if not fits.empty else 0):.1f}s</div><div class="l">median fit</div></div>
</div>

{pio.to_html(fig, include_plotlyjs="cdn", full_html=False)}

<h3 style="margin-top:30px;">Recent fits</h3>
{recent_html}

<div style="margin-top:30px; color:#888; font-size:11px;">
  Dashboard auto-refresh every {REFRESH_SECONDS}s. Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.
</div>
</body>
</html>"""
    out = run_dir / "dashboard.html"
    out.write_text(html)
    return out


def _find_latest_run() -> Optional[str]:
    if not RUNS_ROOT.exists():
        return None
    runs = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime).name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="run id under factor_timing/outputs/runs")
    ap.add_argument("--watch",  action="store_true", help="rebuild every 5s")
    ap.add_argument("--no-refresh", action="store_true", help="disable meta-refresh tag")
    args = ap.parse_args()

    run_id = args.run_id or _find_latest_run()
    if run_id is None:
        raise SystemExit("No runs found in factor_timing/outputs/runs/")

    if args.watch:
        print(f"watching {run_id}  (Ctrl-C to stop)")
        try:
            while True:
                out = build_dashboard(run_id, auto_refresh=not args.no_refresh)
                print(f"  rebuilt {out}  @ {time.strftime('%H:%M:%S')}")
                time.sleep(REFRESH_SECONDS)
        except KeyboardInterrupt:
            pass
    else:
        out = build_dashboard(run_id, auto_refresh=not args.no_refresh)
        print(f"wrote {out}")
