"""
=============================================================================
SCRIPT NAME: clustering.py
=============================================================================

INPUT:
- Monthly factor-return DataFrame (wide: Date × Factor, first col "Date")

OUTPUT:
- DataFrame[factor, epoch_date, cluster_id]
    one row per (factor, retrain epoch) giving the Ward-k cluster assignment
    for that factor to be used at that retrain epoch.  Cluster assignments
    are refit at every retrain date using only data available at that time.

VERSION: 1.0
LAST UPDATED: 2026-04-22

DESCRIPTION:
Expanding-window Ward linkage clustering for the GDELT panel.  At each
retrain date t, the function refits Ward linkage on the factor-return
correlation matrix computed from monthly data strictly before t.  No
forward leakage.

Distance metric: D_ij = sqrt(2 * (1 - |rho_ij|))   (angular, sign-insensitive)
Linkage:        Ward
Cut:            fcluster with t=k, criterion='maxclust'

For the T2 panel this module is unused — T2 trains per-factor.

DEPENDENCIES:
- numpy, pandas, scipy

USAGE:
    from factor_timing.train.clustering import expanding_window_clusters
    df = expanding_window_clusters(monthly_returns, retrain_dates, k=8)
=============================================================================
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

log = logging.getLogger(__name__)


def _corr_distance(corr: pd.DataFrame) -> np.ndarray:
    """Angular distance from |correlation|; returns condensed vector."""
    a = corr.abs().values.astype(float)
    np.fill_diagonal(a, 1.0)
    D = np.sqrt(np.clip(2.0 * (1.0 - a), 0.0, 2.0))
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return squareform(D, checks=False)


def cluster_at(
    returns_wide: pd.DataFrame,
    as_of: pd.Timestamp,
    k: int = 8,
    min_obs: int = 24,
    min_factor_coverage: float = 0.8,
) -> pd.Series:
    """Cluster factors using data strictly before `as_of`.

    Args:
        returns_wide: DataFrame with a 'Date' column and one column per factor
        as_of:        clustering cutoff — only data with Date < as_of is used
        k:            number of clusters
        min_obs:      minimum pairwise observations to trust a correlation
        min_factor_coverage: drop factors with < this fraction non-NaN in-sample

    Returns:
        Series indexed by factor name with integer cluster ids in [1..k].
    """
    hist = returns_wide[returns_wide["Date"] < as_of].drop(columns=["Date"])
    if len(hist) < min_obs:
        raise ValueError(f"Too few history rows at {as_of}: {len(hist)} < {min_obs}")

    coverage = hist.notna().mean()
    keep = coverage[coverage >= min_factor_coverage].index.tolist()
    if len(keep) < k + 1:
        raise ValueError(
            f"Insufficient factors with {min_factor_coverage*100:.0f}% coverage "
            f"at {as_of}: {len(keep)} available, need >= {k+1}"
        )
    hist = hist[keep]
    corr = hist.corr(min_periods=min_obs)
    # Fill any remaining NaN correlations with 0 (weak dependence fallback)
    corr = corr.fillna(0.0)

    d_cond = _corr_distance(corr)
    Z = linkage(d_cond, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return pd.Series(labels, index=corr.index, name="cluster").astype(int)


def expanding_window_clusters(
    returns_wide: pd.DataFrame,
    retrain_dates: Iterable[pd.Timestamp],
    k: int = 8,
    min_obs: int = 24,
    min_factor_coverage: float = 0.8,
) -> pd.DataFrame:
    """Cluster factors at each retrain date using only prior information.

    Returns a long DataFrame with columns [epoch_date, factor, cluster].
    Factors that fail the coverage filter at an epoch are omitted for that
    epoch (they'll pick up an assignment at a later epoch once sufficient
    history accumulates).
    """
    rows = []
    for dt in retrain_dates:
        try:
            lbl = cluster_at(returns_wide, dt, k=k,
                             min_obs=min_obs,
                             min_factor_coverage=min_factor_coverage)
        except ValueError as e:
            log.warning("Skipping epoch %s: %s", dt, e)
            continue
        for fac, cid in lbl.items():
            rows.append({"epoch_date": pd.Timestamp(dt), "factor": fac, "cluster": int(cid)})
    return pd.DataFrame(rows)
