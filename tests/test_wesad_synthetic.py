"""SYNTHETIC smoke test for src/wesad_preprocessing.py.

It fabricates one tiny WESAD-shaped subject (random + sinusoid signals) ONLY to prove that the code
loads the pickle layout, aligns labels, windows, extracts features and writes files. It says NOTHING
about real WESAD data or model performance. Real WESAD has not been processed in this project's
validated run - run the script locally on your own copy.
"""
import pickle

import numpy as np
import pandas as pd

from src.utils import load_config
from src.wesad_preprocessing import run


def make_subject(root, name="S2", seconds=420, seed=0):
    rng = np.random.default_rng(seed)
    n700 = seconds * 700
    t700 = np.arange(n700) / 700
    label = np.zeros(n700, dtype=int)
    label[20 * 700:140 * 700] = 1
    label[140 * 700:260 * 700] = 2
    label[260 * 700:380 * 700] = 3
    ecg = rng.normal(0, 0.05, n700)
    ecg[(np.arange(0, seconds, 1 / 1.2) * 700).astype(int).clip(0, n700 - 1)] += 1.0   # 72 bpm spikes
    chest = {"ACC": rng.normal(0, 0.1, (n700, 3)), "ECG": ecg.reshape(-1, 1), "EMG": rng.normal(0, 0.02, (n700, 1)),
             "EDA": (3 + 0.2 * np.sin(t700 / 20)).reshape(-1, 1), "Resp": np.sin(2 * np.pi * 0.25 * t700).reshape(-1, 1),
             "Temp": np.full((n700, 1), 33.0) + rng.normal(0, 0.01, (n700, 1))}
    def t(fs): return np.arange(seconds * fs) / fs
    wrist = {"ACC": rng.normal(0, 5, (seconds * 32, 3)), "BVP": np.sin(2 * np.pi * 1.2 * t(64)).reshape(-1, 1),
             "EDA": (2 + 0.1 * np.sin(t(4) / 10)).reshape(-1, 1), "TEMP": np.full((seconds * 4, 1), 32.5)}
    d = root / name
    d.mkdir(parents=True)
    with open(d / f"{name}.pkl", "wb") as f:
        pickle.dump({"signal": {"chest": chest, "wrist": wrist}, "label": label, "subject": name}, f, protocol=2)


def test_wesad_pipeline_on_synthetic_subject(tmp_path):
    make_subject(tmp_path / "WESAD")
    cfg = load_config()
    out = tmp_path / "out"
    rep = run(tmp_path / "WESAD", out, cfg)
    df = pd.read_parquet(out / "wesad_windows_features.parquet")
    assert rep["subjects_found"] == ["S2"] and "S3" in rep["missing_expected_subjects"]
    assert set(df.label.unique()) == {1, 2, 3}                        # transient (0) windows dropped
    assert (df.label_purity >= cfg["wesad"]["label_purity"]).all()
    assert df.groupby("label").size().min() >= 2
    assert abs(df["chest_ecg_hr_mean"].median() - 72) < 4              # synthetic 1.2 Hz = 72 bpm
    assert abs(df["wrist_bvp_hr_mean"].median() - 72) < 4
    assert abs(df["chest_resp_rate_bpm"].median() - 15) < 3            # 0.25 Hz = 15 breaths/min
    assert (out / "feature_columns.json").exists() and (out / "wesad_signal_inventory.csv").exists()
    inv = pd.read_csv(out / "wesad_signal_inventory.csv")
    assert abs(inv.loc[(inv.device == "wrist") & (inv.signal == "BVP"), "inferred_fs_hz"].iloc[0] - 64) < 0.01
    import json
    cols = json.load(open(out / "feature_columns.json"))["feature_columns"]
    assert "subject" not in cols and "label" not in cols


def test_audit_only_and_missing_dir(tmp_path):
    make_subject(tmp_path / "WESAD")
    cfg = load_config()
    rep = run(tmp_path / "WESAD", tmp_path / "o2", cfg, audit_only=True)
    assert rep["audit_only"] and "total_windows" not in rep
    import pytest
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "nope", tmp_path / "o3", cfg)
