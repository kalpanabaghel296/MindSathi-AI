"""Advanced training for the PHQ-9 >= 10 risk model (GPU used when available).

Run:   python -m src.train_advanced              # full run
       python -m src.train_advanced --quick      # fast check (few trials)
       python -m src.train_advanced --cpu        # force CPU

What it does (all tuning uses participant-wise cross-validation on TRAIN+VAL only; test is touched ONCE at the end):
 1. Two input variants:  behaviour_only  |  behaviour_plus_start (adds the person's starting PHQ-9 score).
 2. Models: start-score-only baseline, Logistic Regression, HistGradientBoosting, XGBoost (GPU), MLP (PyTorch, GPU, epochs + early stopping).
 3. Random search for hyper-parameters, early stopping (XGBoost trees / MLP epochs) on an inner participant-wise split.
 4. Pick the best by CV ROC-AUC (an explainable model wins if within 0.005 of the best), refit on train, calibrate + set thresholds on val,
    evaluate on test once, fairness gaps, demo users -> outputs/models/ (the dashboard reads these).
 5. Figures in reports/figures/ and tables in reports/advanced_*.csv|json.
Output is a screening risk indicator, never a diagnosis.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from app.risk import level_from_prob
from .train_models import explain, friendly, pick_thresholds
from .utils import ROOT, load_config, outputs_dir

warnings.filterwarnings("ignore")
SEED = 42
START = "phq9_score_start"
EXPLAINABLE = ("xgboost", "logistic")


# ------------------------------------------------------------------ hardware
def pick_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    try:  # XGBoost can use the GPU even without torch
        from xgboost import XGBClassifier
        XGBClassifier(n_estimators=2, device="cuda", tree_method="hist").fit(np.random.rand(40, 3), np.random.randint(0, 2, 40))
        return "cuda"
    except Exception:
        return "cpu"


# ------------------------------------------------------------------ MLP (PyTorch)
class MLPModel:
    """Small sklearn-like wrapper: predict_proba(X) -> (n, 2). Keeps the loss history for plotting."""

    def __init__(self, hidden=(256, 128), dropout=0.3, lr=1e-3, weight_decay=1e-3, max_epochs=300, patience=20, batch=256, pos_weight=2.0, device="cpu", seed=SEED):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.history_, self.net_ = [], None

    def fit(self, X, y, Xv, yv):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        dev = self.device
        t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=dev)
        xt, yt, xv, yv_ = t(X), t(y), t(Xv), t(yv)
        layers, d = [], xt.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]; d = h
        net = nn.Sequential(*layers, nn.Linear(d, 1)).to(dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(self.pos_weight), device=dev))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, best_ep = np.inf, None, 0
        for ep in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(xt), device=dev)
            for i in range(0, len(xt), self.batch):
                idx = perm[i:i + self.batch]
                if len(idx) < 2:
                    continue  # BatchNorm needs more than one row
                opt.zero_grad(); lossf(net(xt[idx]).squeeze(1), yt[idx]).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                tl, vl = lossf(net(xt).squeeze(1), yt).item(), lossf(net(xv).squeeze(1), yv_).item()
            self.history_.append((ep + 1, tl, vl))
            if vl < best - 1e-4:
                best, best_ep, best_state = vl, ep, {k: v.detach().clone() for k, v in net.state_dict().items()}
            elif ep - best_ep >= self.patience:
                break
        net.load_state_dict(best_state)
        self.net_ = net.cpu().eval(); self.best_epoch_ = best_ep + 1
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(np.asarray(X, dtype=np.float32))).squeeze(1)).numpy()
        return np.c_[1 - p, p]


# ------------------------------------------------------------------ model factory + search spaces
def sample_params(kind: str, rng: np.random.Generator) -> dict:
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    if kind == "logistic":
        return {"C": pick([0.01, 0.03, 0.1, 0.3, 1.0])}
    if kind == "hist_gb":
        return {"max_depth": pick([2, 3, 4, 5]), "learning_rate": float(10 ** rng.uniform(-2, -0.9)), "max_iter": pick([100, 200, 300, 500]),
                "l2_regularization": pick([0.0, 1.0, 5.0, 20.0]), "min_samples_leaf": pick([20, 40, 80])}
    if kind == "xgboost":
        return {"max_depth": pick([2, 3, 4, 5, 6]), "learning_rate": float(10 ** rng.uniform(-2, -1)), "subsample": float(rng.uniform(0.6, 1)),
                "colsample_bytree": float(rng.uniform(0.4, 1)), "min_child_weight": pick([1, 3, 5, 10]), "reg_lambda": pick([1, 3, 10, 30]), "gamma": pick([0, 0.5, 2])}
    if kind == "mlp":
        return {"hidden": pick([(128,), (256, 128), (256, 128, 64)]), "dropout": pick([0.2, 0.4]), "weight_decay": pick([1e-4, 1e-3, 1e-2]), "lr": pick([1e-3, 3e-4])}
    raise ValueError(kind)


def inner_split(y, groups, seed=SEED):
    """About 14% of participants for early stopping (participant-wise, stratified)."""
    sg = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    tr, es = next(sg.split(np.zeros(len(y)), y, groups))
    return tr, es


def fit_model(kind, params, X, y, groups, cw, device):
    """Fit one model on (X, y). Early stopping uses an inner participant-wise split of X. Returns object with predict_proba."""
    y = np.asarray(y); groups = np.asarray(groups)
    w = cw["sklearn_class_weight"]
    if kind == "logistic":
        return LogisticRegression(max_iter=3000, class_weight={0: w["0"], 1: w["1"]}, **params).fit(X, y)
    if kind == "hist_gb":
        return HistGradientBoostingClassifier(class_weight={0: w["0"], 1: w["1"]}, early_stopping=False, random_state=SEED, **params).fit(X, y)
    tr, es = inner_split(y, groups)
    Xa, Xe = X.iloc[tr], X.iloc[es]
    if kind == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=3000, early_stopping_rounds=50, eval_metric="logloss", tree_method="hist", device=device,
                          scale_pos_weight=cw["xgboost_scale_pos_weight"], random_state=SEED, **params)
        m.fit(Xa, y[tr], eval_set=[(Xe, y[es])], verbose=False)
        return m
    if kind == "mlp":
        return MLPModel(pos_weight=cw["xgboost_scale_pos_weight"], device=device, **params).fit(Xa.to_numpy(), y[tr], Xe.to_numpy(), y[es])
    raise ValueError(kind)


def cv_score(kind, params, X, y, groups, cw, device, folds=5):
    aucs, aps = [], []
    sg = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, te in sg.split(X, y, groups):
        m = fit_model(kind, params, X.iloc[tr], y[tr], groups[tr], cw, device)
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(aps))


def available_kinds():
    kinds = ["logistic", "hist_gb"]
    for mod, k in (("xgboost", "xgboost"), ("torch", "mlp")):
        try:
            __import__(mod); kinds.append(k)
        except ImportError:
            print(f"[skip] {k}: package '{mod}' not installed")
    return kinds


# ------------------------------------------------------------------ main experiment
def build_variants(data):
    out = {"behaviour_only": {s: data[s][0] for s in data}}
    mu, sd = data["train"][1][START].mean(), data["train"][1][START].std() or 1.0
    out["behaviour_plus_start"] = {s: data[s][0].assign(**{START: ((data[s][1][START].fillna(mu) - mu) / sd).to_numpy()}) for s in data}
    return out, {"mean": float(mu), "std": float(sd)}


def run(data, cw, out_dir, fig_dir, rep_dir, trials=30, folds=5, device="cpu", kinds=None, n_demo=8):
    rng = np.random.default_rng(SEED)
    kinds = kinds or available_kinds()
    variants, start_stats = build_variants(data)
    t = {s: data[s][1]["target_ge10"].astype(int).to_numpy() for s in data}
    g = {s: data[s][1]["participant_id"].to_numpy() for s in data}
    n_trials = {"logistic": 5, "hist_gb": max(4, trials // 3), "xgboost": trials, "mlp": max(4, trials // 4)}

    # pooled train+val for cross-validation (test never included)
    rows, best_cfg = [], {}
    yall, gall = np.r_[t["train"], t["val"]], np.r_[g["train"], g["val"]]
    sc = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    base_auc = []
    S = np.r_[data["train"][1][START].fillna(start_stats["mean"]), data["val"][1][START].fillna(start_stats["mean"])].reshape(-1, 1)
    for tr, te in sc.split(S, yall, gall):
        base_auc.append(roc_auc_score(yall[te], LogisticRegression().fit(S[tr], yall[tr]).predict_proba(S[te])[:, 1]))
    baseline_cv = float(np.mean(base_auc))
    print(f"baseline (start score only) CV AUC = {baseline_cv:.4f}")

    for vname, V in variants.items():
        Xall = pd.concat([V["train"], V["val"]], ignore_index=True)
        for kind in kinds:
            t0, best = time.time(), None
            for i in range(n_trials[kind]):
                params = sample_params(kind, rng)
                try:
                    m, s, ap = cv_score(kind, params, Xall, yall, gall, cw, device, folds)
                except Exception as e:  # one bad config must not kill the run
                    print(f"  [{vname}/{kind}] trial {i} failed: {e}"); continue
                if best is None or m > best[0]:
                    best = (m, s, ap, params)
            if best is None:
                continue
            print(f"[{vname:22s}] {kind:9s} CV AUC {best[0]:.4f} ± {best[1]:.4f}  ({time.time() - t0:.0f}s)")
            rows.append({"variant": vname, "model": kind, "cv_auc_mean": round(best[0], 4), "cv_auc_std": round(best[1], 4), "cv_pr_auc": round(best[2], 4),
                         "best_params": json.dumps(best[3], default=str)})
            best_cfg[(vname, kind)] = best[3]
    res = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
    rep_dir.mkdir(parents=True, exist_ok=True); res.to_csv(rep_dir / "advanced_cv_results.csv", index=False)

    # ---- choose: best CV AUC; an explainable model wins if within 0.005 ----
    try:
        import shap  # noqa: F401
        explainable = EXPLAINABLE
    except ImportError:
        explainable = ("logistic",)
    top = res.iloc[0]
    close = res[(res.cv_auc_mean >= top.cv_auc_mean - 0.005) & (res.variant == top.variant) & (res.model.isin(explainable))]
    pick = close.iloc[0] if len(close) else top
    vname, kind = pick.variant, pick.model
    params = best_cfg[(vname, kind)]
    print(f"\nChosen: {vname} / {kind}  (CV AUC {pick.cv_auc_mean})")

    # ---- refit on TRAIN, calibrate + thresholds on VAL, evaluate on TEST once ----
    V = variants[vname]
    model = fit_model(kind, params, V["train"], t["train"], g["train"], cw, device)
    pv = model.predict_proba(V["val"])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, t["val"])
    thr = pick_thresholds(t["val"], iso.predict(pv))
    p_te = iso.predict(model.predict_proba(V["test"])[:, 1])
    yte = t["test"]
    base = LogisticRegression().fit(data["train"][1][[START]].fillna(start_stats["mean"]), t["train"])
    base_te = roc_auc_score(yte, base.predict_proba(data["test"][1][[START]].fillna(start_stats["mean"]))[:, 1])
    levels = [level_from_prob(p, thr["moderate"], thr["high"]) for p in p_te]
    metrics = {"note": "Screening risk indicator, not a diagnosis.", "chosen_model": kind, "variant": vname,
               "variant_description": "behaviour + context + the person's starting PHQ-9 score" if vname == "behaviour_plus_start" else "behaviour + context only (no questionnaire needed)",
               "device": device, "baseline_start_score_only_cv_auc": round(baseline_cv, 4), "cv_results": res.drop(columns="best_params").to_dict("records"),
               "best_params": params,
               "test": {"roc_auc": round(roc_auc_score(yte, p_te), 4), "pr_auc": round(average_precision_score(yte, p_te), 4), "brier": round(brier_score_loss(yte, p_te), 4),
                        "macro_f1_at_moderate_threshold": round(f1_score(yte, (p_te >= thr["moderate"]).astype(int), average="macro"), 4),
                        "baseline_start_score_only_auc": round(base_te, 4), "n_rows": int(len(yte)), "positive_rate": round(float(yte.mean()), 4),
                        "share_by_level": {l: round(levels.count(l) / len(levels), 3) for l in ("low", "moderate", "high")}}}
    metrics["test"]["beats_baseline"] = metrics["test"]["roc_auc"] > base_te
    fair, mte = {}, data["test"][2]
    for col in mte.columns:
        if col == "sex" or col.startswith("race_"):
            for gv in mte[col].dropna().unique():
                idx = (mte[col] == gv).to_numpy()
                if idx.sum() >= 30 and len(np.unique(yte[idx])) == 2:
                    fair[f"{col}={gv}"] = {"n": int(idx.sum()), "auc": round(roc_auc_score(yte[idx], p_te[idx]), 4)}
    metrics["fairness_auc_by_group"] = fair

    # ---- demo users (chosen by predicted risk quantile, not by outcome) ----
    method, contrib = explain(kind, model, V["test"], list(V["test"].columns))
    yt = data["test"][1]; ids = yt["participant_id"].to_numpy(); order = np.argsort(p_te); picks, seen = [], set()
    for q in np.linspace(0.03, 0.97, n_demo):
        k0 = int(q * (len(order) - 1))
        for j in list(order[k0:]) + list(order[:k0][::-1]):
            if ids[j] not in seen:
                seen.add(ids[j]); picks.append(int(j)); break
    users = []
    for n, j in enumerate(picks):
        u = {"label": f"Demo user {chr(65 + n)}", "probability": round(float(p_te[j]), 3), "level": levels[j],
             "questionnaire_start": None if pd.isna(yt[START].iloc[j]) else float(yt[START].iloc[j]),
             "questionnaire_end_for_evaluation_only": float(yt["phq9_score_end"].iloc[j]), "factors": []}
        if contrib is not None:
            for i in np.argsort(-np.abs(contrib[j]))[:5]:
                u["factors"].append({"name": friendly(V["test"].columns[i]), "direction": "raises" if contrib[j][i] > 0 else "lowers", "size": round(float(abs(contrib[j][i])), 3)})
        users.append(u)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": iso, "columns": list(V["train"].columns), "thresholds": thr, "name": kind, "variant": vname, "start_stats": start_stats}, out_dir / "risk_model.joblib")
    (out_dir / "thresholds.json").write_text(json.dumps(thr, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "demo_users.json").write_text(json.dumps({"method": method, "thresholds": thr, "users": users}, indent=2, ensure_ascii=False))

    make_figures(res, baseline_cv, model, kind, best_cfg, variants[vname], t, g, cw, device, p_te, yte, fig_dir)
    return metrics


# ------------------------------------------------------------------ figures
def make_figures(res, baseline_cv, model, kind, best_cfg, V, t, g, cw, device, p_te, yte, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. CV comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    lab = res.variant.str.replace("_", " ") + "\n" + res.model
    ax.barh(range(len(res)), res.cv_auc_mean, xerr=res.cv_auc_std, color="#1f7a7a")
    ax.set_yticks(range(len(res))); ax.set_yticklabels(lab, fontsize=8); ax.invert_yaxis()
    ax.axvline(baseline_cv, color="#bf4b45", ls="--", label=f"start score only ({baseline_cv:.3f})"); ax.axvline(0.5, color="grey", lw=.5)
    ax.set_xlim(0.5, max(0.95, res.cv_auc_mean.max() + 0.03)); ax.set_xlabel("Cross-validated ROC-AUC (participant-wise)"); ax.legend(); fig.tight_layout()
    fig.savefig(fig_dir / "cv_comparison.png", dpi=150); plt.close(fig)

    # 2. loss curve (XGBoost: validation logloss per tree; MLP: train and validation loss per epoch)
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        if kind == "mlp":
            h = np.array(model.history_); ax.plot(h[:, 0], h[:, 1], label="train loss"); ax.plot(h[:, 0], h[:, 2], label="validation loss")
            ax.axvline(model.best_epoch_, color="grey", ls=":", label=f"early stop (epoch {model.best_epoch_})"); ax.set_xlabel("epoch")
        elif kind == "xgboost":
            ev = model.evals_result()["validation_0"]["logloss"]; ax.plot(ev, label="validation logloss"); ax.axvline(model.best_iteration, color="grey", ls=":", label=f"early stop (tree {model.best_iteration})"); ax.set_xlabel("trees")
        else:
            raise RuntimeError
        ax.set_ylabel("loss"); ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "loss_curve.png", dpi=150)
    except Exception:
        pass
    plt.close("all")

    # 3. learning curve: more training participants -> validation AUC
    pids = np.unique(g["train"]); rng = np.random.default_rng(SEED); xs, ys, es = [], [], []
    params = best_cfg_for(res, best_cfg, kind)
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        a = []
        for r in range(2 if frac < 1 else 1):
            keep = np.isin(g["train"], rng.choice(pids, int(len(pids) * frac), replace=False))
            if len(np.unique(t["train"][keep])) < 2:
                continue
            try:
                m = fit_model(kind, params, V["train"][keep], t["train"][keep], g["train"][keep], cw, device)
                a.append(roc_auc_score(t["val"], m.predict_proba(V["val"])[:, 1]))
            except Exception:
                pass
        if a:
            xs.append(int(frac * len(V["train"]))); ys.append(np.mean(a))
    if xs:
        fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(xs, ys, "o-", color="#1f7a7a"); ax.set_xlabel("training rows"); ax.set_ylabel("validation ROC-AUC"); ax.set_title(f"Learning curve ({kind})"); fig.tight_layout()
        fig.savefig(fig_dir / "learning_curve.png", dpi=150); plt.close(fig)

    # 4. calibration on test
    pt, pp = calibration_curve(yte, p_te, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5)); ax.plot([0, 1], [0, 1], "k:", label="perfect"); ax.plot(pp, pt, "o-", color="#1f7a7a", label="model")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed share with PHQ-9 >= 10"); ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "calibration_test.png", dpi=150); plt.close(fig)


def best_cfg_for(res, best_cfg, kind):
    for (v, k), p in best_cfg.items():
        if k == kind:
            return p
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config(); out = outputs_dir(cfg, "primary"); dev = pick_device(a.cpu)
    print(f"device: {dev}")
    data = {s: (pd.read_parquet(out / f"X_{s}.parquet"), pd.read_parquet(out / f"y_{s}.parquet"), pd.read_parquet(out / f"meta_{s}.parquet")) for s in ("train", "val", "test")}
    cw = json.loads((out / "class_weights.json").read_text())
    m = run(data, cw, outputs_dir(cfg, "models"), ROOT / "reports" / "figures", ROOT / "reports", trials=6 if a.quick else a.trials, folds=3 if a.quick else 5, device=dev)
    print(json.dumps({k: m[k] for k in ("chosen_model", "variant", "test")}, indent=2))


if __name__ == "__main__":
    main()
