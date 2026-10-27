"""Temporal / personal-baseline features for the primary dataset.

WHAT THE DATA SUPPORTS
    * The row index is <participant>_<period>, periods run 1..12 and every participant has
      several rows (median 8), so an earlier-rows baseline per participant CAN be built.
    * There are NO timestamps and NO daily records. A 2-4 week daily baseline, daily/weekly
      series, or day-level trends CANNOT be reconstructed and are not attempted.
    * The period index is assumed to be an ordered, roughly monthly step (inferred from the
      labels appearing only on periods 3, 6, 9, 12). This is an inference, not documented.

RULE: every feature for period k uses only periods < k (plus the current row's own value
for the 'delta' columns). No future period is ever read.

These features are EXPERIMENTAL and are written to outputs/temporal/, not mixed into X_*.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_history_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = cfg["temporal"]
    min_hist = int(t["min_history"])
    cols = [c for c in t["features"] if c in df.columns]
    d = df[["participant_id", "period"] + cols].sort_values(["participant_id", "period"]).reset_index(drop=True)
    g = d.groupby("participant_id", sort=False)

    consecutive = (d["period"] - g["period"].shift(1)) == 1
    out = d[["participant_id", "period"]].copy()
    out["n_prior_periods"] = g.cumcount()                      # rows strictly before this one
    for f in cols:
        cur = d[f]
        notna = cur.notna().astype(int)
        prior_cnt = notna.groupby(d["participant_id"]).cumsum() - notna
        prior_sum = cur.fillna(0).groupby(d["participant_id"]).cumsum() - cur.fillna(0)
        prior_mean = (prior_sum / prior_cnt).where(prior_cnt >= min_hist)
        prev = g[f].shift(1).where(consecutive)
        out[f"{f}__prev"] = prev
        out[f"{f}__delta_prev"] = cur - prev
        out[f"{f}__prior_mean"] = prior_mean
        out[f"{f}__delta_from_own_baseline"] = cur - prior_mean
    return out


def history_coverage(hist: pd.DataFrame, labelled_ids: pd.DataFrame, min_history: int) -> dict:
    """How many labelled rows actually have >= min_history earlier periods?"""
    m = labelled_ids.merge(hist[["participant_id", "period", "n_prior_periods"]], on=["participant_id", "period"], how="left")
    return {"labelled_rows": int(len(m)),
            f"rows_with_>={min_history}_prior_periods": int((m["n_prior_periods"] >= min_history).sum()),
            "share": float((m["n_prior_periods"] >= min_history).mean())}
