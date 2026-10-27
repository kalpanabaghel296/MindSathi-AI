"""Audit of the primary dataset (Parquet + feature-explanation TSV).

Everything here is READ-ONLY with respect to the raw inputs. It writes reports:
    reports/dataset_inventory.csv      (primary rows; StudentLife/WESAD rows are appended by their own scripts)
    reports/feature_data_dictionary.csv
    reports/missing_values_report.csv
    reports/duplicates_report.csv
    reports/invalid_values_report.csv
    reports/redundant_features.csv
    reports/audit_summary_primary.json

Run:  python -m src.audit
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .utils import load_config, raw_path, reports_dir, save_json


def load_primary_raw(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(raw_path(cfg, "parquet_file"))
    sep = cfg["primary"]["id_separator"]
    parts = df.index.to_series().astype(str).str.split(sep, n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("Row index does not follow '<participant><sep><period>' - check config.primary.id_separator")
    df = df.copy()
    df.insert(0, "participant_id", parts[0].values)
    df.insert(1, "period", pd.to_numeric(parts[1], errors="coerce").astype("Int64").values)
    return df


def load_tsv(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(raw_path(cfg, "tsv_file"), sep="\t")


def band_of(score: float, edges: list[int]) -> float:
    """Standard PHQ-9 bands: 0-4, 5-9, 10-14, 15-19, 20-27 -> 0..4."""
    if pd.isna(score):
        return np.nan
    return float(np.searchsorted(edges, score, side="right"))


def audit_primary(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    p = cfg["primary"]
    out = reports_dir(cfg)
    df = load_primary_raw(cfg)
    tsv = load_tsv(cfg)
    feat_cols = [c for c in df.columns if c not in ("participant_id", "period")]

    # ---------------- schema / dictionary mapping ----------------
    tsv_names = set(tsv["Feature name"])
    in_parquet_not_tsv = sorted(set(feat_cols) - tsv_names)
    in_tsv_not_parquet = sorted(tsv_names - set(feat_cols))

    # ---------------- missing values ----------------
    miss = pd.DataFrame({
        "column": feat_cols,
        "dtype": [str(df[c].dtype) for c in feat_cols],
        "n_missing": [int(df[c].isna().sum()) for c in feat_cols],
    })
    miss["pct_missing"] = (100 * miss.n_missing / len(df)).round(3)
    miss["n_unique"] = [int(df[c].nunique(dropna=True)) for c in feat_cols]
    miss.sort_values("pct_missing", ascending=False).to_csv(out / "missing_values_report.csv", index=False)

    # ---------------- duplicates ----------------
    non_label = [c for c in feat_cols if not c.startswith("phq9")]
    dup_rows = int(df.duplicated(subset=feat_cols).sum())
    dup_feat = int(df.duplicated(subset=non_label).sum())
    dup_index = int(df.index.duplicated().sum())
    dup_pid_period = int(df.duplicated(subset=["participant_id", "period"]).sum())
    pd.DataFrame([
        ("exact duplicate rows (all columns)", dup_rows),
        ("duplicate feature vectors (excluding PHQ-9 columns)", dup_feat),
        ("duplicate row index values", dup_index),
        ("duplicate (participant_id, period) pairs", dup_pid_period),
    ], columns=["check", "count"]).to_csv(out / "duplicates_report.csv", index=False)

    # ---------------- participants / periods / labels ----------------
    per_pid = df.groupby("participant_id").size()
    lab = df.dropna(subset=[p["baseline_score"], p["target_score"]])
    lab_per_pid = lab.groupby("participant_id").size()
    n_with_zero = int(df["participant_id"].nunique() - lab["participant_id"].nunique())
    cat_ok = {}
    for score, cat in (("phq9_score_start", "phq9_cat_start"), ("phq9_score_end", "phq9_cat_end")):
        derived = lab[score].map(lambda s: band_of(s, p["phq9_bands"]))
        cat_ok[cat] = float((derived == lab[cat]).mean())

    # ---------------- invalid / suspicious values ----------------
    rules = p["invalid_rules"]
    inv = []
    def add(name, mask, action):
        inv.append({"check": name, "n_rows": int(mask.sum()), "action": action})
    add("weight < min adult weight", df.weight < rules["min_adult_weight_lb"], "set weight & bmi to missing")
    add("bmi > 60 (plausible severe obesity)", df.bmi > 60, "kept, flagged only")
    add("bmi < 14", df.bmi < 14, "same rows as low weight")
    recomputed = 703 * df.weight / df.height ** 2
    add("bmi differs from 703*weight/height^2 by > 1", (recomputed - df.bmi).abs() > 1, "none found: bmi is derived, redundant")
    both0 = (df.sleep_asleep_weekday_mean == 0) & (df.sleep_in_bed_weekday_mean == 0)
    add("sleep asleep == 0 and in-bed == 0 (weekday)", both0, "treated as device non-wear -> missing")
    add("steps_awake_mean < 500", df.steps_awake_mean < 500, "kept, flagged (may be non-wear)")
    add("steps_awake_mean > 40000", df.steps_awake_mean > 40000, "kept, flagged (extreme but possible)")
    add("asleep minutes > in-bed minutes (weekday)", df.sleep_asleep_weekday_mean > df.sleep_in_bed_weekday_mean + 1e-6, "none found")
    add("sleep ratio > 1", df.sleep_ratio_asleep_in_bed_weekday_mean > 1, "none found")
    add("negative steps", df.steps_awake_mean < 0, "none found")
    add("sleep start hour > 32 or < 15 (adjusted scale)", (df.sleep_main_start_hour_adj_median > 32) | (df.sleep_main_start_hour_adj_median < 15), "kept, flagged")
    pd.DataFrame(inv).to_csv(out / "invalid_values_report.csv", index=False)

    # ---------------- constant / redundant ----------------
    constant = [c for c in feat_cols if df[c].nunique(dropna=True) <= 1]
    num = df[feat_cols].select_dtypes(include=[np.number, bool]).astype(float)
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (upper.stack().reset_index().rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_corr"}))
    pairs = pairs[pairs.abs_corr >= cfg["primary"]["preprocess"]["corr_threshold"]].sort_values("abs_corr", ascending=False)
    pairs.to_csv(out / "redundant_features.csv", index=False)

    # ---------------- data dictionary ----------------
    d = tsv.rename(columns={"Feature name": "feature", "Category": "tsv_category", "Subcategory": "tsv_subcategory",
                            "Description": "tsv_description", "Notes": "tsv_notes"}).copy()
    d["in_parquet"] = d.feature.isin(feat_cols)
    d = d.merge(miss[["column", "dtype", "pct_missing", "n_unique"]], left_on="feature", right_on="column", how="left").drop(columns="column")
    def role(f):
        if f in p["drop_always"]:
            return "excluded_outcome_period"
        if f == p["baseline_score"]:
            return "baseline_input_stored_with_labels"
        if f in p["sensitive"]:
            return "sensitive_fairness_only"
        if f in p["timing_uncertain"]:
            return "timing_uncertain_sensitivity_set"
        return "model_candidate"
    d["role"] = d.feature.map(role)
    d["has_description"] = d.tsv_description.notna()
    d["is_constant"] = d.feature.isin(constant)
    d["raw_or_engineered"] = np.where(d.tsv_subcategory.eq("Linear regression"), "engineered_regression_aggregate",
                              np.where(d.tsv_category.eq("PHQ-9"), "questionnaire_outcome", "engineered_aggregate_or_static"))
    d.to_csv(out / "feature_data_dictionary.csv", index=False)

    # ---------------- distributions snapshot ----------------
    desc = df[feat_cols].describe(percentiles=[.01, .5, .99]).T
    desc.to_csv(out / "numeric_distribution_summary.csv")

    # ---------------- primary inventory rows ----------------
    inv_rows = pd.DataFrame([
        {"dataset": "primary", "item": raw_path(cfg, "parquet_file").name, "kind": "parquet",
         "rows": len(df), "columns": len(feat_cols),
         "notes": f"{df.participant_id.nunique()} participants; index = participant_period; aggregates, not raw sensors"},
        {"dataset": "primary", "item": raw_path(cfg, "tsv_file").name, "kind": "tsv",
         "rows": len(tsv), "columns": tsv.shape[1],
         "notes": f"feature dictionary; {int(tsv.Description.isna().sum())} of {len(tsv)} rows have no description (regression-feature blocks)"},
    ])
    inv_rows.to_csv(out / "dataset_inventory_primary.csv", index=False)

    summary = {
        "n_rows": int(len(df)), "n_feature_columns": len(feat_cols),
        "n_participants": int(df.participant_id.nunique()),
        "periods_per_participant": per_pid.describe().to_dict(),
        "period_values": sorted(df.period.dropna().astype(int).unique().tolist()),
        "labelled_rows": int(len(lab)), "labelled_participants": int(lab.participant_id.nunique()),
        "participants_with_>=2_labelled_rows": int((lab_per_pid >= 2).sum()),
        "participants_without_any_label": n_with_zero,
        "label_period_distribution": lab.period.value_counts().sort_index().to_dict(),
        "category_matches_standard_bands": cat_ok,
        "tsv_names_missing_in_parquet": in_tsv_not_parquet,
        "parquet_columns_missing_in_tsv": in_parquet_not_tsv,
        "tsv_rows_without_description": int(tsv.Description.isna().sum()),
        "exact_duplicate_rows": dup_rows, "duplicate_feature_vectors": dup_feat,
        "constant_columns": constant,
        "n_corr_pairs_over_threshold": int(len(pairs)),
        "columns_over_50pct_missing": int((miss.pct_missing > 50).sum()),
    }
    save_json(summary, out / "audit_summary_primary.json")
    return summary


def assemble_inventory(cfg: dict | None = None) -> pd.DataFrame:
    """Combine per-dataset inventory rows into reports/dataset_inventory.csv."""
    cfg = cfg or load_config()
    rep = reports_dir(cfg)
    parts = []
    for name in ("primary", "studentlife"):
        f = rep / f"dataset_inventory_{name}.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    parts.append(pd.DataFrame([{"dataset": "wesad", "item": "local WESAD folder (~2.5 GB compressed)", "kind": "folder of S*/S*.pkl",
                                "rows": None, "columns": None,
                                "notes": "NOT accessible in the execution environment - not inspected. Run: python -m src.wesad_preprocessing --audit-only"}]))
    inv = pd.concat(parts, ignore_index=True)
    inv.to_csv(rep / "dataset_inventory.csv", index=False)
    return inv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    s = audit_primary(load_config(args.config))
    assemble_inventory(load_config(args.config))
    for k, v in s.items():
        print(f"{k}: {v}")
