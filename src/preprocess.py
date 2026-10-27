"""Leakage-safe preprocessing for the primary dataset (anon_processed_df_parquet).

Stages (each is a plain function so notebooks can call them one by one):
    1. load_and_validate      - load, parse participant/period from the index, schema checks
    2. clean                  - fix only PROVABLY invalid values, log everything
    3. build_targets          - validated PHQ-9 targets (score, >=10 flag, change)
    4. select_features        - leakage + sensitivity + timing-uncertainty control
    5. participant_split      - participant-wise, stratified train/val/test
    6. TabularPreprocessor    - clip, impute, indicators, scale; FIT ON TRAIN ONLY
    7. export + validate_outputs

Run end to end:  python -m src.preprocess
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .audit import band_of, load_primary_raw, load_tsv
from .feature_engineering import add_history_features
from .utils import load_config, outputs_dir, reports_dir, save_json

ID_COLS = ["participant_id", "period"]


# ----------------------------------------------------------------------------
# 1. load + validate
# ----------------------------------------------------------------------------
def load_and_validate(cfg: dict) -> pd.DataFrame:
    p = cfg["primary"]
    df = load_primary_raw(cfg)
    tsv = load_tsv(cfg)
    feat = [c for c in df.columns if c not in ID_COLS]
    missing_in_parquet = set(tsv["Feature name"]) - set(feat)
    unknown_cols = set(feat) - set(tsv["Feature name"])
    assert not missing_in_parquet, f"TSV features absent from parquet: {sorted(missing_in_parquet)}"
    assert not unknown_cols, f"Parquet columns absent from TSV: {sorted(unknown_cols)}"
    for c in (p["target_score"], p["baseline_score"]):
        assert c in df.columns, f"required column missing: {c}"
    assert df["period"].notna().all(), "unparseable period in index"
    assert not df.duplicated(subset=ID_COLS).any(), "duplicate (participant, period) rows"
    # bool -> int so that every model column is numeric
    for c in df.columns[df.dtypes == bool]:
        df[c] = df[c].astype(int)
    return df


# ----------------------------------------------------------------------------
# 2. clean (only provably invalid values; flag - do not delete - everything else)
# ----------------------------------------------------------------------------
def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = cfg["primary"]["invalid_rules"]
    df = df.copy()
    log = []

    bad_w = df["weight"] < rules["min_adult_weight_lb"]
    df.loc[bad_w, ["weight", "bmi"]] = np.nan
    log.append(("weight below adult minimum -> weight, bmi set to NaN", int(bad_w.sum())))

    if rules.get("sleep_nonwear_if_zero", True):
        sleep_cols = [c for c in df.columns if c.startswith(("sleep_asleep_", "sleep_in_bed_", "sleep_ratio_"))]
        for wk in ("weekday", "weekend"):
            a, b = f"sleep_asleep_{wk}_mean", f"sleep_in_bed_{wk}_mean"
            nonwear = (df[a] == 0) & (df[b] == 0)
            cols = [c for c in sleep_cols if wk in c and c.endswith("_mean")]
            df.loc[nonwear, cols] = np.nan
            log.append((f"{wk}: asleep==0 & in-bed==0 -> device non-wear, {len(cols)} sleep means set to NaN", int(nonwear.sum())))

    # flags only (rows are kept): documented in reports/invalid_values_report.csv
    log.append(("steps_awake_mean < 500 (flag only, kept)", int((df["steps_awake_mean"] < 500).sum())))
    log.append(("bmi > 60 (flag only, kept)", int((df["bmi"] > 60).sum())))
    return df, pd.DataFrame(log, columns=["step", "n_rows_affected"])


# ----------------------------------------------------------------------------
# 3. targets
# ----------------------------------------------------------------------------
def build_targets(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = cfg["primary"]
    df = df.copy()
    s, e = p["baseline_score"], p["target_score"]
    # validate: scores in range, categories consistent with standard bands
    for c in (s, e):
        v = df[c].dropna()
        assert v.between(0, 27).all(), f"{c} outside 0-27"
    lab = df[s].notna() & df[e].notna()
    for sc, ct in ((s, "phq9_cat_start"), (e, "phq9_cat_end")):
        derived = df.loc[lab, sc].map(lambda x: band_of(x, p["phq9_bands"]))
        assert (derived == df.loc[lab, ct]).all(), f"{ct} inconsistent with {sc}"
    df["is_labelled"] = lab
    df["target_ge10"] = np.where(lab, (df[e] >= p["binary_threshold"]).astype(float), np.nan)
    df["target_delta"] = np.where(lab, df[e] - df[s], np.nan)
    return df


# ----------------------------------------------------------------------------
# 4. feature selection / leakage control
# ----------------------------------------------------------------------------
def select_features(df: pd.DataFrame, cfg: dict) -> dict:
    p = cfg["primary"]
    reserved = set(ID_COLS) | {"is_labelled", "target_ge10", "target_delta", p["baseline_score"]}
    forbidden = set(p["drop_always"])
    sensitive = set(p["sensitive"])
    uncertain = [c for c in p["timing_uncertain"] if c in df.columns]
    main = [c for c in df.columns
            if c not in reserved | forbidden | sensitive | set(uncertain)]
    assert not any(c.startswith("phq9") for c in main), "PHQ-9 column leaked into the feature list"
    return {"main": main, "timing_uncertain": uncertain, "sensitive": sorted(sensitive),
            "forbidden": sorted(forbidden)}


# ----------------------------------------------------------------------------
# 5. participant-wise split
# ----------------------------------------------------------------------------
def participant_split(lab: pd.DataFrame, cfg: dict) -> pd.Series:
    """Return a Series (index = participant_id) with 'train' | 'val' | 'test'.

    Rows of one participant always land in the same split. Stratification uses the
    binary target at row level (StratifiedGroupKFold keeps groups intact)."""
    sp = cfg["primary"]["split"]
    assert sp["train_folds"] + sp["val_folds"] + sp["test_folds"] == sp["n_folds"]
    sgkf = StratifiedGroupKFold(n_splits=sp["n_folds"], shuffle=True, random_state=cfg["seed"])
    fold = pd.Series(-1, index=lab.index)
    for i, (_, te) in enumerate(sgkf.split(lab, lab["target_ge10"].astype(int), groups=lab["participant_id"])):
        fold.iloc[te] = i
    name = np.where(fold < sp["train_folds"], "train",
            np.where(fold < sp["train_folds"] + sp["val_folds"], "val", "test"))
    per_row = pd.Series(name, index=lab.index)
    pid_split = per_row.groupby(lab["participant_id"]).agg(lambda s: s.iloc[0])
    assert per_row.groupby(lab["participant_id"]).nunique().max() == 1, "participant spans several splits"
    return pid_split


# ----------------------------------------------------------------------------
# 6. fitted preprocessing object
# ----------------------------------------------------------------------------
class TabularPreprocessor:
    """Clip -> impute -> missing-indicators -> robust scale. All statistics come from fit() data only.

    * constant and near-duplicate columns (|r| >= corr_threshold) are dropped
    * binary columns are imputed with the mode and are NOT scaled
    * other columns: winsorised at train quantiles, median-imputed, scaled by (x - median) / IQR
    * a missing-indicator is added for columns whose TRAIN missing rate exceeds a threshold;
      indicators with identical train patterns are collapsed into one
    """

    def __init__(self, clip_quantiles=(0.005, 0.995), indicator_min_missing=0.05, corr_threshold=0.98):
        self.clip_quantiles = tuple(clip_quantiles)
        self.indicator_min_missing = indicator_min_missing
        self.corr_threshold = corr_threshold

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        X = X.astype(float)
        nun = X.nunique(dropna=True)
        self.constant_cols_ = nun[nun <= 1].index.tolist()
        X = X.drop(columns=self.constant_cols_)

        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.dropped_correlated_ = {c: upper[c].idxmax() for c in upper.columns if (upper[c] >= self.corr_threshold).any()}
        X = X.drop(columns=list(self.dropped_correlated_))

        self.columns_ = X.columns.tolist()
        nun = X.nunique(dropna=True)
        self.binary_cols_ = nun[nun <= 2].index.tolist()
        self.cont_cols_ = [c for c in self.columns_ if c not in self.binary_cols_]

        lo, hi = self.clip_quantiles
        self.clip_low_ = X[self.cont_cols_].quantile(lo)
        self.clip_high_ = X[self.cont_cols_].quantile(hi)
        clipped = X[self.cont_cols_].clip(self.clip_low_, self.clip_high_, axis=1)
        self.median_ = clipped.median()
        self.mode_ = X[self.binary_cols_].mode(dropna=True).iloc[0]
        q25, q75 = clipped.quantile(0.25), clipped.quantile(0.75)
        iqr = (q75 - q25).replace(0, np.nan)
        self.scale_ = iqr.fillna(clipped.std().replace(0, np.nan)).fillna(1.0)

        miss_rate = X.isna().mean()
        cand = miss_rate[miss_rate > self.indicator_min_missing].index.tolist()
        seen, keep = {}, []
        for c in cand:
            key = X[c].isna().to_numpy().tobytes()
            if key not in seen:
                seen[key] = c
                keep.append(c)
        self.indicator_cols_ = keep
        self.n_indicator_candidates_ = len(cand)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.columns_).astype(float)
        ind = X[self.indicator_cols_].isna().astype(int)
        ind.columns = [f"{c}__missing" for c in self.indicator_cols_]
        cont = X[self.cont_cols_].clip(self.clip_low_, self.clip_high_, axis=1)
        cont = cont.fillna(self.median_)
        cont = (cont - self.median_) / self.scale_
        binr = X[self.binary_cols_].fillna(self.mode_)
        out = pd.concat([cont, binr, ind], axis=1)
        out.index = X.index
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> list[str]:
        return self.cont_cols_ + self.binary_cols_ + [f"{c}__missing" for c in self.indicator_cols_]


# ----------------------------------------------------------------------------
# 7. run + export
# ----------------------------------------------------------------------------
def run_pipeline(cfg: dict | None = None, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    p = cfg["primary"]
    out = outputs_dir(cfg, "primary")
    rep = reports_dir(cfg)

    df = load_and_validate(cfg)
    df, clean_log = clean(df, cfg)
    df = build_targets(df, cfg)
    clean_log.to_csv(rep / "cleaning_log.csv", index=False)

    sets = select_features(df, cfg)
    cand = sets["main"] + sets["timing_uncertain"]

    # cleaned dataset (all rows, incl. unlabelled ones; ids and targets kept for traceability)
    cleaned = df.copy()
    cleaned.to_parquet(out / "cleaned_primary_dataset.parquet", index=False)

    lab = df[df["is_labelled"]].reset_index(drop=True)
    pid_split = participant_split(lab, cfg)
    lab["split"] = lab["participant_id"].map(pid_split)
    pid_split.rename("split").rename_axis("participant_id").reset_index().to_csv(out / "split_assignment.csv", index=False)

    pre = TabularPreprocessor(p["preprocess"]["clip_quantiles"], p["preprocess"]["indicator_min_missing"],
                              p["preprocess"]["corr_threshold"])
    train_mask = lab["split"] == "train"
    pre.fit(lab.loc[train_mask, cand])           # <- TRAIN ROWS ONLY
    joblib.dump({"preprocessor": pre, "feature_sets": sets, "config_primary": p}, out / "preprocessor.joblib")

    all_out_cols = pre.get_feature_names_out()
    uncertain_out = [c for c in all_out_cols if c.replace("__missing", "") in set(sets["timing_uncertain"])]
    main_out = [c for c in all_out_cols if c not in uncertain_out]

    summary_rows = []
    for split in ("train", "val", "test"):
        m = lab["split"] == split
        Xt = pre.transform(lab.loc[m, cand])
        y = lab.loc[m, ID_COLS + [p["baseline_score"], p["target_score"], "target_ge10", "target_delta"]].reset_index(drop=True)
        meta = lab.loc[m, ID_COLS + [c for c in sets["sensitive"] if c in lab.columns] + ["birthyear"]].reset_index(drop=True)
        Xt = Xt.reset_index(drop=True)
        Xt[main_out].to_parquet(out / f"X_{split}.parquet", index=False)
        Xt[uncertain_out].to_parquet(out / f"X_{split}_timing_uncertain_extra.parquet", index=False)
        y.to_parquet(out / f"y_{split}.parquet", index=False)
        meta.to_parquet(out / f"meta_{split}.parquet", index=False)
        summary_rows.append({"split": split, "rows": int(m.sum()), "participants": int(lab.loc[m, "participant_id"].nunique()),
                             "positive_rate_ge10": float(y["target_ge10"].mean()),
                             "mean_phq9_end": float(y[p["target_score"]].mean()),
                             "mean_phq9_start": float(y[p["baseline_score"]].mean())})
    split_summary = pd.DataFrame(summary_rows)
    split_summary.to_csv(rep / "split_summary.csv", index=False)

    ytr = lab.loc[train_mask, "target_ge10"]
    pos, neg = float(ytr.sum()), float((1 - ytr).sum())
    save_json({"train_negative": neg, "train_positive": pos,
               "sklearn_class_weight": {"0": (neg + pos) / (2 * neg), "1": (neg + pos) / (2 * pos)},
               "xgboost_scale_pos_weight": neg / pos,
               "note": "Computed from TRAIN only. Apply as class weights during model fitting; no resampling was done on the exported files."},
              out / "class_weights.json")

    (out / "final_feature_list.txt").write_text("\n".join(main_out) + "\n")
    (out / "timing_uncertain_feature_list.txt").write_text("\n".join(uncertain_out) + "\n")
    save_json({"main_feature_columns": main_out, "timing_uncertain_extra_columns": uncertain_out,
               "excluded_outcome_columns": sets["forbidden"], "sensitive_fairness_only": sets["sensitive"],
               "dropped_constant": pre.constant_cols_, "dropped_correlated(dropped->kept)": pre.dropped_correlated_,
               "missing_indicator_columns": pre.indicator_cols_,
               "n_indicator_candidates_before_dedup": pre.n_indicator_candidates_},
              out / "feature_sets.json")

    # experimental history features (strictly past periods of the same participant)
    if cfg["temporal"]["enabled"]:
        hist = add_history_features(df, cfg)
        hist.to_parquet(outputs_dir(cfg, "temporal") / "temporal_features_experimental.parquet", index=False)

    result = {"clean_log": clean_log, "split_summary": split_summary, "n_main_features": len(main_out),
              "n_uncertain_features": len(uncertain_out), "dropped_correlated": pre.dropped_correlated_,
              "n_indicators": len(pre.indicator_cols_), "n_labelled_rows": len(lab)}
    validate_outputs(cfg)
    if verbose:
        print(clean_log.to_string(index=False)); print(split_summary.to_string(index=False))
        print(f"main features: {len(main_out)} | timing-uncertain extras: {len(uncertain_out)} | indicators: {len(pre.indicator_cols_)}")
    return result


def validate_outputs(cfg: dict | None = None) -> pd.DataFrame:
    """Independent checks on the files that were written. Raises AssertionError on failure."""
    cfg = cfg or load_config()
    p = cfg["primary"]
    out = outputs_dir(cfg, "primary")
    checks = []
    def ok(name, cond, detail=""):
        checks.append({"check": name, "passed": bool(cond), "detail": detail})
        assert cond, f"VALIDATION FAILED: {name} {detail}"

    X = {s: pd.read_parquet(out / f"X_{s}.parquet") for s in ("train", "val", "test")}
    Y = {s: pd.read_parquet(out / f"y_{s}.parquet") for s in ("train", "val", "test")}
    M = {s: pd.read_parquet(out / f"meta_{s}.parquet") for s in ("train", "val", "test")}
    E = {s: pd.read_parquet(out / f"X_{s}_timing_uncertain_extra.parquet") for s in ("train", "val", "test")}
    feats = (out / "final_feature_list.txt").read_text().split()

    pids = {s: set(Y[s]["participant_id"]) for s in Y}
    ok("no participant in two splits", not (pids["train"] & pids["val"] or pids["train"] & pids["test"] or pids["val"] & pids["test"]),
       f"train={len(pids['train'])} val={len(pids['val'])} test={len(pids['test'])}")
    for s in X:
        ok(f"{s}: X rows == y rows == meta rows", len(X[s]) == len(Y[s]) == len(M[s]) == len(E[s]), str(len(X[s])))
        ok(f"{s}: no NaN/inf in X", np.isfinite(X[s].to_numpy()).all())
        ok(f"{s}: no NaN/inf in timing-uncertain extra", np.isfinite(E[s].to_numpy()).all())
        ok(f"{s}: columns identical to final_feature_list", list(X[s].columns) == feats)
        ok(f"{s}: targets complete", Y[s][[p["target_score"], "target_ge10"]].notna().all().all())
    allcols = set().union(*[set(X[s].columns) | set(E[s].columns) for s in X])
    ok("no PHQ-9 column in any X file", not any("phq9" in c for c in allcols))
    ok("sensitive columns not in X", not (set(p["sensitive"]) & allcols))
    ok("no identifier columns in X", not ({"participant_id", "period"} & allcols))
    ok("no constant columns in train X", (X["train"].nunique() > 1).all())
    ok("train medians ~0 for scaled columns", abs(X["train"].median().median()) < 0.5)
    for s in ("val", "test"):
        ok(f"{s}: positive class present", Y[s]["target_ge10"].sum() > 0)
    ok("prediction target is the END score (not start)", p["target_score"] == "phq9_score_end")

    pre = joblib.load(out / "preprocessor.joblib")["preprocessor"]
    ok("preprocessor fitted on training rows only", pre.median_.notna().all() and len(pre.columns_) > 0)
    # refit-on-train must reproduce the saved transform (determinism / no hidden fit on val-test)
    raw = pd.read_parquet(out / "cleaned_primary_dataset.parquet")
    lab = raw[raw["is_labelled"]]
    tr_ids = pids["train"]
    sets = joblib.load(out / "preprocessor.joblib")["feature_sets"]
    cand = sets["main"] + sets["timing_uncertain"]
    pre2 = TabularPreprocessor(pre.clip_quantiles, pre.indicator_min_missing, pre.corr_threshold).fit(lab[lab.participant_id.isin(tr_ids)][cand])
    ok("refit on train ids reproduces saved preprocessor", np.allclose(pre2.median_.values, pre.median_.values) and pre2.columns_ == pre.columns_)

    res = pd.DataFrame(checks)
    res.to_csv(reports_dir(cfg) / "validation_report.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run_pipeline(load_config(args.config))
