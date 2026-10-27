"""Audit + minimal cleaning of the archive dataset (identified from its files as StudentLife).

The ZIP is read in place (nothing is extracted). Sensing streams (~3 GB) are only inventoried
and their headers inspected; they are NOT parsed or aligned here, because the PHQ-9 labels
(46 participants, pre/post only) are far too few to train on and the feature space shares
nothing with the primary dataset. The one cleaned artefact is a scored PHQ-9 table.

Outputs:
    reports/studentlife_inventory.csv
    reports/studentlife_audit.json
    reports/dataset_inventory_studentlife.csv
    outputs/studentlife/phq9_scored.csv

Run:  python -m src.studentlife_audit
"""
from __future__ import annotations

import argparse
import io
import re
import zipfile
from collections import defaultdict

import pandas as pd

from .utils import load_config, outputs_dir, raw_path, reports_dir, save_json

PHQ_MAP = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
BANDS = [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"), (15, 19, "moderately severe"), (20, 27, "severe")]


def _modality(name: str) -> str:
    parts = name.split("/")
    if len(parts) >= 3 and parts[1] in ("sensing", "EMA"):
        return "/".join(parts[1:3])
    return parts[1] if len(parts) >= 2 else parts[0]


def inventory(zf: zipfile.ZipFile) -> pd.DataFrame:
    agg = defaultdict(lambda: {"files": 0, "bytes": 0, "users": set(), "exts": set()})
    for i in zf.infolist():
        if i.is_dir():
            continue
        m = _modality(i.filename)
        a = agg[m]
        a["files"] += 1
        a["bytes"] += i.file_size
        a["exts"].add(i.filename.rsplit(".", 1)[-1] if "." in i.filename else "")
        u = re.search(r"u(\d{2})", i.filename.split("/")[-1])
        if u:
            a["users"].add(u.group(0))
    rows = [{"modality": k, "files": v["files"], "users": len(v["users"]), "uncompressed_MB": round(v["bytes"] / 1e6, 2),
             "extensions": ",".join(sorted(v["exts"]))} for k, v in sorted(agg.items())]
    return pd.DataFrame(rows)


def sensing_headers(zf: zipfile.ZipFile) -> dict:
    out = {}
    seen = set()
    for i in zf.infolist():
        m = _modality(i.filename)
        if m.startswith("sensing/") and m not in seen and i.filename.endswith(".csv"):
            seen.add(m)
            with zf.open(i) as f:
                out[m] = f.readline().decode("utf-8", "replace").strip()
    return out


def score_phq9(zf: zipfile.ZipFile) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(io.BytesIO(zf.read("dataset/survey/PHQ-9.csv")))
    item_cols = list(d.columns[2:11])                      # 9 scored items; last column = functional difficulty (not scored)
    unknown = set()
    for c in item_cols:
        unknown |= set(d[c].dropna().unique()) - set(PHQ_MAP)
    assert not unknown, f"unexpected PHQ-9 answer text: {unknown}"
    items = d[item_cols].apply(lambda s: s.map(PHQ_MAP))
    out = d[["uid", "type"]].copy()
    out["phq9_total"] = items.sum(axis=1, min_count=9)     # NaN if any of the 9 items is missing
    out["n_items_answered"] = items.notna().sum(axis=1)
    out["severity_band"] = out["phq9_total"].map(lambda s: next((b for lo, hi, b in BANDS if lo <= s <= hi), None) if pd.notna(s) else None)
    out["item9_positive"] = items.iloc[:, 8] > 0            # kept for safety review only, never a model input
    wide = out.pivot(index="uid", columns="type", values="phq9_total")
    info = {
        "responses": int(len(out)), "participants": int(out.uid.nunique()),
        "pre_responses": int((out.type == "pre").sum()), "post_responses": int((out.type == "post").sum()),
        "participants_with_pre_and_post": int(wide.dropna().shape[0]),
        "incomplete_responses": int((out.n_items_answered < 9).sum()),
        "pre_mean": float(wide["pre"].mean()), "post_mean": float(wide["post"].mean()),
        "pre_ge10": int((wide["pre"] >= 10).sum()), "post_ge10": int((wide["post"] >= 10).sum()),
    }
    return out, info


def audit_studentlife(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    rep = reports_dir(cfg)
    zpath = raw_path(cfg, "studentlife_zip")
    with zipfile.ZipFile(zpath) as zf:
        inv = inventory(zf)
        headers = sensing_headers(zf)
        scored, phq = score_phq9(zf)
        survey_files = [i.filename for i in zf.infolist() if "/survey/" in i.filename]
        n_entries = len(zf.infolist())
        total_gb = sum(i.file_size for i in zf.infolist()) / 1e9
    inv.to_csv(rep / "studentlife_inventory.csv", index=False)
    scored.to_csv(outputs_dir(cfg, "studentlife") / "phq9_scored.csv", index=False)
    summary = {
        "archive": zpath.name, "identified_as": "StudentLife (Dartmouth): per-user u00..u59 phone sensing, EMA, surveys, app usage, calls, SMS",
        "entries": n_entries, "uncompressed_GB": round(total_gb, 2), "sensing_headers": headers,
        "survey_files": survey_files, "phq9": phq,
        "merge_with_primary": "NO - different population, sensors, features, label timing and scale of labels",
        "gaps": ["no demographic file in the archive", "PHQ-9 only pre/post (46 participants) - cannot support model training",
                 "sensing streams not parsed in this pipeline"],
    }
    save_json(summary, rep / "studentlife_audit.json")
    pd.DataFrame([{"dataset": "studentlife", "item": zpath.name, "kind": "zip", "rows": n_entries, "columns": None,
                   "notes": f"{total_gb:.2f} GB uncompressed; {inv.users.max()} users in sensing; PHQ-9: {phq['participants']} participants, {phq['responses']} responses"}]
                 ).to_csv(rep / "dataset_inventory_studentlife.csv", index=False)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    s = audit_studentlife(load_config(args.config))
    print(pd.read_csv(reports_dir(load_config(args.config)) / "studentlife_inventory.csv").to_string(index=False))
    print(s["phq9"])
