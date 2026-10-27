"""WESAD (Wearable Stress and Affect Detection) preprocessing - SEPARATE from the depression pipeline.

WESAD labels baseline / stress / amusement (/ meditation). It has NO depression label. Its output is
a stress/affect feature table for a supplementary experiment and must never be merged into the
PHQ-9 targets of the primary dataset.

Expected local layout (official release):
    <WESAD_DIR>/S2/S2.pkl, S3/S3.pkl, ... S17/S17.pkl     (S1 and S12 do not exist in the release)
Each .pkl is a Python-2 pickle -> loaded with encoding="latin1". It contains
    data["signal"]["chest"]: ACC(N,3) ECG EMG EDA Resp Temp   (nominal 700 Hz)
    data["signal"]["wrist"]: ACC(N,3) 32 Hz, BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz
    data["label"]: (N,) at 700 Hz   0=transient 1=baseline 2=stress 3=amusement 4=meditation 5/6/7=ignore
    data["subject"]: "S2"
Sampling rates are NOT hard-coded: each one is inferred as len(signal) * 700 / len(label) and reported.

USAGE
    python -m src.wesad_preprocessing --wesad-dir /path/to/WESAD --audit-only
    python -m src.wesad_preprocessing --wesad-dir /path/to/WESAD
    python -m src.wesad_preprocessing --wesad-dir /path/to/WESAD --subjects S2 S3 --out-dir outputs/wesad

PROCESSING DECISIONS (documented in reports/preprocessing_decisions.md)
    * one subject in memory at a time; each subject is written to its own parquet and then freed
    * windows: window_sec=60, step_sec=30 (50% overlap) from config; a window is kept only if one label covers
      >= label_purity (80%) of it AND that label is in labels_keep (1,2,3). Transient/ignored labels are dropped.
    * label alignment: the label of a window is the majority label of the 700 Hz label samples inside it;
      all signals are cut by TIME (seconds), so different sampling rates stay aligned.
    * invalid readings: non-finite values -> NaN; EDA outside [0, 60] uS and TEMP outside [20, 45] C -> NaN;
      a signal with > max_nan_fraction invalid samples in a window gets NaN features for that window;
      short gaps (<= limit) are linearly interpolated; flat-lined ECG/BVP/Resp -> NaN features.
    * heart-rate features come from a simple band-pass + peak detector (ECG) / pulse detector (BVP):
      useful as a baseline, NOT clinical-grade R-peak detection.
    * subject id is kept in the output for splitting/evaluation and is NOT listed in feature_columns.json.
    * no per-subject normalisation that uses labels (it would leak); evaluate with leave-one-subject-out.
"""
from __future__ import annotations

import argparse
import gc
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from .utils import ROOT, load_config, save_json

LABEL_NAMES = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation", 5: "ignore", 6: "ignore", 7: "ignore"}
CHEST = ["ACC", "ECG", "EMG", "EDA", "Resp", "Temp"]
WRIST = ["ACC", "BVP", "EDA", "TEMP"]
SUBJECT_RE = re.compile(r"^S\d+$")
EXPECTED_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]


# ----------------------------------------------------------------------------
# discovery + loading
# ----------------------------------------------------------------------------
def find_subject_files(wesad_dir: Path) -> tuple[dict, dict]:
    wesad_dir = Path(wesad_dir)
    if not wesad_dir.is_dir():
        raise FileNotFoundError(f"WESAD directory not found: {wesad_dir}\nSet --wesad-dir or paths.wesad_dir in configs/config.yaml")
    found, extras = {}, {}
    for d in sorted(wesad_dir.iterdir()):
        if d.is_dir() and SUBJECT_RE.match(d.name):
            pkl = d / f"{d.name}.pkl"
            if pkl.is_file():
                found[d.name] = pkl
            extras[d.name] = sorted(p.name for p in d.iterdir() if p.name != pkl.name)
    missing = [s for s in EXPECTED_SUBJECTS if s not in found]
    return found, {"missing_expected_subjects": missing, "other_files_per_subject": extras,
                   "top_level_entries": sorted(p.name for p in wesad_dir.iterdir())}


def load_subject(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


# ----------------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------------
def inspect_subject(data: dict, label_fs: float = 700.0) -> tuple[dict, list[dict]]:
    label = np.asarray(data["label"]).ravel()
    dur = len(label) / label_fs
    counts = pd.Series(label).value_counts().sort_index()
    info = {"subject": str(data.get("subject")), "duration_sec": round(dur, 1), "n_label_samples": int(len(label)),
            "label_seconds": {f"{int(k)}:{LABEL_NAMES.get(int(k), '?')}": round(v / label_fs, 1) for k, v in counts.items()},
            "unexpected_labels": [int(k) for k in counts.index if int(k) not in LABEL_NAMES]}
    rows = []
    for dev, sigs in data["signal"].items():
        for name, arr in sigs.items():
            a = np.asarray(arr, dtype=float)
            fin = np.isfinite(a)
            rows.append({"subject": info["subject"], "device": dev, "signal": name, "shape": str(a.shape),
                         "inferred_fs_hz": round(len(a) * label_fs / len(label), 3),
                         "n_nonfinite": int((~fin).sum()),
                         "min": float(np.nanmin(a[fin])) if fin.any() else np.nan,
                         "max": float(np.nanmax(a[fin])) if fin.any() else np.nan,
                         "std": float(np.nanstd(a[fin])) if fin.any() else np.nan,
                         "flatline": bool(fin.any() and np.nanstd(a[fin]) < 1e-9)})
    return info, rows


# ----------------------------------------------------------------------------
# signal helpers
# ----------------------------------------------------------------------------
def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 2) -> np.ndarray:
    nyq = fs / 2
    hi = min(hi, nyq * 0.95)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, np.nan_to_num(x - np.nanmean(x)))


def _mask_invalid(x: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if lo is not None:
        bad |= x < lo
    if hi is not None:
        bad |= x > hi
    x[bad] = np.nan
    return x


def _fill_small_gaps(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).interpolate(limit=int(max(1, len(x) * 0.05)), limit_direction="both").to_numpy()


def _basic(x: np.ndarray, p: str, fs: float) -> dict:
    t = np.arange(len(x)) / fs
    ok = np.isfinite(x)
    slope = np.polyfit(t[ok], x[ok], 1)[0] if ok.sum() > 2 else np.nan
    return {f"{p}_mean": np.nanmean(x), f"{p}_std": np.nanstd(x), f"{p}_min": np.nanmin(x),
            f"{p}_max": np.nanmax(x), f"{p}_slope": slope}


def _nan_basic(p: str) -> dict:
    return {f"{p}_{s}": np.nan for s in ("mean", "std", "min", "max", "slope")}


def _hrv_from_peaks(peaks: np.ndarray, fs: float, p: str) -> dict:
    keys = [f"{p}_hr_mean", f"{p}_sdnn_ms", f"{p}_rmssd_ms", f"{p}_n_beats"]
    if len(peaks) < 6:
        return dict.fromkeys(keys, np.nan)
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.3) & (rr < 2.0)]                       # 30-200 bpm only
    if len(rr) < 5:
        return dict.fromkeys(keys, np.nan)
    return {keys[0]: 60.0 / rr.mean(), keys[1]: rr.std(ddof=1) * 1000,
            keys[2]: np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000, keys[3]: float(len(rr) + 1)}


def _detect_peaks_ecg(f: np.ndarray, fs: float) -> np.ndarray:
    thr = np.mean(f) + 2.0 * np.std(f)
    pk, _ = find_peaks(f, height=thr, distance=int(0.3 * fs))
    return pk


def _detect_peaks_ppg(f: np.ndarray, fs: float) -> np.ndarray:
    pk, _ = find_peaks(f, prominence=0.5 * np.std(f), distance=int(0.3 * fs))
    return pk


# ----------------------------------------------------------------------------
# per-window features
# ----------------------------------------------------------------------------
def _prepare_signals(data: dict, label_len: int, label_fs: float, devices: list[str]) -> dict:
    """Return {key: dict(x=array(N,) or (N,3), fs=float, kind=str, filtered=array|None)}. Heavy filtering done ONCE per subject."""
    out = {}
    spec = {"chest": CHEST, "wrist": WRIST}
    for dev in devices:
        for name in spec[dev]:
            if name not in data["signal"].get(dev, {}):
                continue
            arr = np.asarray(data["signal"][dev][name], dtype=float)
            fs = len(arr) * label_fs / label_len
            key = f"{dev}_{name.lower()}"
            entry = {"fs": fs, "kind": name.upper()}
            if name == "ACC":
                entry["x"] = _mask_invalid(arr.reshape(len(arr), -1))
            else:
                x = arr.ravel()
                if name.upper() == "EDA":
                    x = _mask_invalid(x, 0.0, 60.0)
                elif name.upper() == "TEMP":
                    x = _mask_invalid(x, 20.0, 45.0)
                else:
                    x = _mask_invalid(x)
                entry["x"] = x
                if name == "ECG":
                    entry["filt"] = _bandpass(x, fs, 5.0, 15.0)
                elif name == "BVP":
                    entry["filt"] = _bandpass(x, fs, 0.5, 8.0)
                elif name == "Resp":
                    entry["filt"] = _bandpass(x, fs, 0.1, 0.8)
            out[key] = entry
    return out


def _window_features(sig: dict, t0: float, t1: float, max_nan: float) -> dict:
    feats = {}
    for key, e in sig.items():
        fs = e["fs"]
        i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
        x = e["x"][i0:i1]
        kind = e["kind"]
        n = len(x)
        if n < 4:
            bad = True
        else:
            bad = (np.isnan(x).mean() > max_nan)
        if kind == "ACC":
            if bad:
                feats.update({f"{key}_mag_mean": np.nan, f"{key}_mag_std": np.nan, f"{key}_mag_max": np.nan})
            else:
                mag = np.sqrt(np.nansum(x ** 2, axis=1))
                feats.update({f"{key}_mag_mean": mag.mean(), f"{key}_mag_std": mag.std(), f"{key}_mag_max": mag.max()})
        elif kind in ("EDA", "TEMP"):
            feats.update(_nan_basic(key) if bad else _basic(_fill_small_gaps(x), key, fs))
        elif kind == "EMG":
            feats.update({f"{key}_absmean": np.nan, f"{key}_std": np.nan, f"{key}_rms": np.nan} if bad else
                         {f"{key}_absmean": np.nanmean(np.abs(x - np.nanmean(x))), f"{key}_std": np.nanstd(x),
                          f"{key}_rms": np.sqrt(np.nanmean(x ** 2))})
        elif kind in ("ECG", "BVP"):
            f = e["filt"][i0:i1]
            if bad or np.std(f) < 1e-9:
                feats.update(_hrv_from_peaks(np.array([]), fs, key))
            else:
                pk = _detect_peaks_ecg(f, fs) if kind == "ECG" else _detect_peaks_ppg(f, fs)
                feats.update(_hrv_from_peaks(pk, fs, key))
        elif kind == "RESP":
            f = e["filt"][i0:i1]
            if bad or np.std(f) < 1e-9:
                feats.update({f"{key}_rate_bpm": np.nan, f"{key}_std": np.nan})
            else:
                pk, _ = find_peaks(f, distance=int(1.5 * fs), prominence=0.3 * np.std(f))
                feats.update({f"{key}_rate_bpm": len(pk) * 60.0 / (t1 - t0), f"{key}_std": np.nanstd(x)})
    return feats


def extract_subject_features(data: dict, cfg_w: dict, devices: list[str]) -> pd.DataFrame:
    label_fs = float(cfg_w["nominal_label_fs"])
    label = np.asarray(data["label"]).ravel().astype(int)
    dur = len(label) / label_fs
    win, step = float(cfg_w["window_sec"]), float(cfg_w["step_sec"])
    keep = set(cfg_w["labels_keep"])
    sig = _prepare_signals(data, len(label), label_fs, devices)
    rows = []
    for t0 in np.arange(0.0, dur - win + 1e-9, step):
        lab = label[int(round(t0 * label_fs)): int(round((t0 + win) * label_fs))]
        vals, cnts = np.unique(lab, return_counts=True)
        maj, purity = int(vals[cnts.argmax()]), float(cnts.max() / len(lab))
        if maj not in keep or purity < cfg_w["label_purity"]:
            continue
        r = {"subject": str(data.get("subject")), "window_start_sec": t0, "window_end_sec": t0 + win,
             "label": maj, "label_name": LABEL_NAMES[maj], "label_purity": purity,
             "label_binary_stress": int(maj == 2)}
        r.update(_window_features(sig, t0, t0 + win, cfg_w["max_nan_fraction"]))
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        feat_cols = [c for c in df.columns if c not in ID_LABEL_COLS]
        df["n_nan_features"] = df[feat_cols].isna().sum(axis=1)
    return df


ID_LABEL_COLS = ["subject", "window_start_sec", "window_end_sec", "label", "label_name", "label_purity",
                 "label_binary_stress", "n_nan_features"]


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def run(wesad_dir: Path, out_dir: Path, cfg: dict, subjects: list[str] | None = None,
        audit_only: bool = False, devices: tuple[str, ...] = ("chest", "wrist")) -> dict:
    cw = cfg["wesad"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files, extra = find_subject_files(Path(wesad_dir))
    if subjects:
        files = {s: p for s, p in files.items() if s in subjects}
    if not files:
        raise FileNotFoundError(f"No <S>/<S>.pkl files found under {wesad_dir}. Found entries: {extra['top_level_entries']}")
    audit_subjects, sig_rows, per_subject = [], [], []
    for s, path in files.items():
        data = load_subject(path)                      # one subject in memory at a time
        info, rows = inspect_subject(data, float(cw["nominal_label_fs"]))
        audit_subjects.append(info)
        sig_rows.extend(rows)
        if not audit_only:
            df = extract_subject_features(data, cw, list(devices))
            df.to_parquet(out_dir / f"wesad_features_{s}.parquet", index=False)
            per_subject.append({"subject": s, "windows": int(len(df)),
                                "windows_by_label": df["label_name"].value_counts().to_dict() if len(df) else {},
                                "mean_nan_features_per_window": float(df["n_nan_features"].mean()) if len(df) else None})
        del data
        gc.collect()
        print(f"[wesad] processed {s}")
    pd.DataFrame(sig_rows).to_csv(out_dir / "wesad_signal_inventory.csv", index=False)
    report = {"wesad_dir": str(wesad_dir), "subjects_found": sorted(files), **extra,
              "per_subject_audit": audit_subjects, "window_config": cw, "audit_only": audit_only}
    if not audit_only:
        parts = [pd.read_parquet(out_dir / f"wesad_features_{s}.parquet") for s in files]
        allw = pd.concat([p for p in parts if len(p)], ignore_index=True)
        allw.to_parquet(out_dir / "wesad_windows_features.parquet", index=False)
        feat_cols = [c for c in allw.columns if c not in ID_LABEL_COLS]
        save_json({"feature_columns": feat_cols, "excluded_from_model_inputs": ID_LABEL_COLS,
                   "note": "subject is for splitting/evaluation only; use leave-one-subject-out."}, out_dir / "feature_columns.json")
        pd.DataFrame({"subject": sorted(allw.subject.unique())}).assign(fold=lambda d: range(len(d))).to_csv(out_dir / "loso_folds.csv", index=False)
        lab_dist = allw.groupby(["subject", "label_name"]).size().unstack(fill_value=0)
        lab_dist.to_csv(out_dir / "wesad_label_distribution.csv")
        report.update({"total_windows": int(len(allw)), "n_features": len(feat_cols), "per_subject": per_subject,
                       "label_distribution_total": allw["label_name"].value_counts().to_dict()})
    save_json(report, out_dir / "wesad_processing_report.json")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wesad-dir", default=None, help="folder containing S2/, S3/, ... (default: paths.wesad_dir in config)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--subjects", nargs="*", default=None, help="e.g. S2 S3 (default: all found)")
    ap.add_argument("--audit-only", action="store_true", help="inspect structure, rates and labels; no feature extraction")
    ap.add_argument("--devices", nargs="+", default=["chest", "wrist"], choices=["chest", "wrist"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    wdir = Path(a.wesad_dir or cfg["paths"]["wesad_dir"])
    odir = Path(a.out_dir or cfg["wesad"]["out_dir"])
    odir = odir if odir.is_absolute() else ROOT / odir
    rep = run(wdir, odir, cfg, a.subjects, a.audit_only, tuple(a.devices))
    print(f"subjects: {rep['subjects_found']}")
    print(f"missing expected subjects: {rep['missing_expected_subjects']}")
    if not a.audit_only:
        print(f"windows: {rep['total_windows']} | features: {rep['n_features']} | labels: {rep['label_distribution_total']}")
