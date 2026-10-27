"""Train the risk model on the prepared splits and write what the dashboard needs.

Run:  python -m src.train_models
Needs outputs/primary/X_*.parquet, y_*.parquet, meta_*.parquet (made by src.preprocess).
Steps: baselines -> pick best on VALIDATION -> isotonic calibration on validation -> thresholds on validation
       -> ONE evaluation on test -> fairness gaps -> demo users (held-out test participants) with top factors.
Writes outputs/models/{risk_model.joblib, metrics.json, thresholds.json, demo_users.json}.
Output is a screening risk indicator, never a diagnosis.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score

from app.risk import level_from_prob
from .utils import load_config, outputs_dir

SEED = 42
FRIENDLY = {  # plain-language names for the most common features; others are cleaned automatically
    "steps_awake_mean": "average daily steps", "sleep_asleep_weekday_mean": "sleep time on weekdays", "sleep_asleep_weekend_mean": "sleep time on weekends",
    "sleep_in_bed_weekday_mean": "time in bed on weekdays", "sleep_main_start_hour_adj_median": "usual bedtime", "sleep_main_start_hour_adj_iqr": "how much bedtime varies",
    "money": "money worries", "comorbid_migraines": "migraine history", "phq9_score_start": "starting questionnaire score", "bmi": "BMI",
}


def friendly(name: str) -> str:
    return FRIENDLY.get(name, name.replace("__", " ").replace("_", " ").strip())


def pick_thresholds(y, p, recall_target=0.85, precision_target=0.60) -> dict:
    """Moderate = highest cut-off that still catches >= 85% of >=10 cases; high = lowest cut-off with precision >= 60%. Validation data only."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok_m = np.where(rec >= recall_target)[0]
    t_m = float(thr[ok_m.max()]) if len(ok_m) else float(np.quantile(p, 0.5))
    ok_h = np.where(prec >= precision_target)[0]
    t_h = float(thr[ok_h.min()]) if len(ok_h) else float(np.quantile(p, 0.9))
    if t_h <= t_m:
        t_h = min(0.99, t_m + 0.15)
    return {"moderate": round(t_m, 4), "high": round(t_h, 4)}


def candidates(cw: dict) -> dict:
    w = cw["sklearn_class_weight"]
    out = {"logistic": LogisticRegression(max_iter=2000, class_weight={0: w["0"], 1: w["1"]}, C=0.3),
           "hist_gb": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=SEED, class_weight={0: w["0"], 1: w["1"]})}
    try:
        from xgboost import XGBClassifier
        out["xgboost"] = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                       scale_pos_weight=cw["xgboost_scale_pos_weight"], eval_metric="logloss", random_state=SEED)
    except ImportError:
        pass
    return out


def explain(model_name, model, X: pd.DataFrame, cols: list[str], k=5):
    """Top factors per row. xgboost+shap -> SHAP values; logistic -> exact coef*value contributions; else None."""
    if model_name == "xgboost":
        try:
            import shap
            return "SHAP TreeExplainer", np.asarray(shap.TreeExplainer(model).shap_values(X))
        except ImportError:
            pass
    if model_name == "logistic":
        return "logistic contribution (coefficient x scaled value)", X.to_numpy() * model.coef_[0]
    return None, None


def run(data: dict, cw: dict, out_dir, n_demo=8) -> dict:
    """data[split] = (X, y, meta) for split in train/val/test. Pure function of DataFrames, so it can be tested without parquet."""
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, mte) = data["train"], data["val"], data["test"]
    ttr, tva, tte = ytr["target_ge10"].astype(int), yva["target_ge10"].astype(int), yte["target_ge10"].astype(int)
    metrics = {"validation_auc": {}, "note": "Screening risk indicator, not a diagnosis."}

    # mandatory baseline: the person's own starting questionnaire score only
    base = LogisticRegression(max_iter=1000).fit(ytr[["phq9_score_start"]], ttr)
    metrics["baseline_start_score_only"] = {"validation_auc": round(roc_auc_score(tva, base.predict_proba(yva[["phq9_score_start"]])[:, 1]), 4)}

    fitted = {}
    for name, m in candidates(cw).items():
        m.fit(Xtr, ttr)
        fitted[name] = m
        metrics["validation_auc"][name] = round(roc_auc_score(tva, m.predict_proba(Xva)[:, 1]), 4)
    va = metrics["validation_auc"]
    top = max(va.values())
    try:
        import shap
        pref = ("xgboost", "logistic")
    except ImportError:
        pref = ("logistic",)
    explainable = [n for n in pref if n in va and va[n] >= top - 0.005]
    best = explainable[0] if explainable else max(va, key=va.get)
    model = fitted[best]
    metrics["chosen_model"] = best

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(model.predict_proba(Xva)[:, 1], tva)  # calibration: validation only
    thr = pick_thresholds(tva, iso.predict(model.predict_proba(Xva)[:, 1]))

    # ---- single evaluation on test ----
    p_te = iso.predict(model.predict_proba(Xte)[:, 1])
    pred = (p_te >= thr["moderate"]).astype(int)
    metrics["test"] = {"roc_auc": round(roc_auc_score(tte, p_te), 4), "pr_auc": round(average_precision_score(tte, p_te), 4),
                       "brier": round(brier_score_loss(tte, p_te), 4), "macro_f1_at_moderate_threshold": round(f1_score(tte, pred, average="macro"), 4),
                       "baseline_start_score_only_auc": round(roc_auc_score(tte, base.predict_proba(yte[["phq9_score_start"]])[:, 1]), 4),
                       "n_rows": int(len(tte)), "positive_rate": round(float(tte.mean()), 4)}
    metrics["test"]["beats_baseline"] = metrics["test"]["roc_auc"] > metrics["test"]["baseline_start_score_only_auc"]
    levels = [level_from_prob(p, thr["moderate"], thr["high"]) for p in p_te]
    metrics["test"]["share_by_level"] = {l: round(levels.count(l) / len(levels), 3) for l in ("low", "moderate", "high")}

    fair = {}
    for col in mte.columns:
        if col in ("sex",) or col.startswith("race_"):
            for g in mte[col].dropna().unique():
                idx = (mte[col] == g).to_numpy()
                if idx.sum() >= 30 and tte[idx].nunique() == 2:
                    fair[f"{col}={g}"] = {"n": int(idx.sum()), "auc": round(roc_auc_score(tte[idx], p_te[idx]), 4)}
    metrics["fairness_auc_by_group"] = fair

    # ---- demo users: chosen by predicted risk quantile (NOT by true outcome), one row per participant ----
    method, contrib = explain(best, model, Xte, list(Xte.columns))
    ids = yte["participant_id"].to_numpy() if "participant_id" in yte else np.arange(len(yte))
    order = np.argsort(p_te)
    picks, seen = [], set()
    for q in np.linspace(0.03, 0.97, n_demo):
        for j in list(order[int(q * (len(order) - 1)):]) + list(order[:int(q * (len(order) - 1))][::-1]):
            if ids[j] not in seen:
                seen.add(ids[j]); picks.append(int(j)); break
    users = []
    for n, j in enumerate(picks):
        u = {"label": f"Demo user {chr(65 + n)}", "probability": round(float(p_te[j]), 3), "level": levels[j],
             "questionnaire_start": None if pd.isna(yte["phq9_score_start"].iloc[j]) else float(yte["phq9_score_start"].iloc[j]),
             "questionnaire_end_for_evaluation_only": float(yte["phq9_score_end"].iloc[j]), "factors": []}
        if contrib is not None:
            for i in np.argsort(-np.abs(contrib[j]))[:5]:
                u["factors"].append({"name": friendly(Xte.columns[i]), "direction": "raises" if contrib[j][i] > 0 else "lowers", "size": round(float(abs(contrib[j][i])), 3)})
        users.append(u)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": iso, "columns": list(Xtr.columns), "thresholds": thr, "name": best}, out_dir / "risk_model.joblib")
    (out_dir / "thresholds.json").write_text(json.dumps(thr, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "demo_users.json").write_text(json.dumps({"method": method, "thresholds": thr, "users": users}, indent=2, ensure_ascii=False))
    return metrics


def main():
    cfg = load_config()
    out = outputs_dir(cfg, "primary")
    data = {s: (pd.read_parquet(out / f"X_{s}.parquet"), pd.read_parquet(out / f"y_{s}.parquet"), pd.read_parquet(out / f"meta_{s}.parquet")) for s in ("train", "val", "test")}
    cw = json.loads((out / "class_weights.json").read_text())
    m = run(data, cw, outputs_dir(cfg, "models"))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
