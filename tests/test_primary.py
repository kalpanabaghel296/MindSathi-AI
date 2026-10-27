"""Executable checks for the primary pipeline. Run:  pytest -q tests/test_primary.py
Needs the raw files (see configs/config.yaml) for the data-dependent tests."""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_history_features
from src.preprocess import TabularPreprocessor

CFG = {"temporal": {"enabled": True, "min_history": 2, "features": ["a"]}}


def _toy():
    rows = []
    for pid in ("p1", "p2"):
        for per in range(1, 7):
            rows.append({"participant_id": pid, "period": per, "a": float(per * (1 if pid == "p1" else 10))})
    return pd.DataFrame(rows)


def test_history_features_never_use_future_rows():
    df = _toy()
    base = add_history_features(df, CFG)
    # corrupt every value AFTER period 3 for p1: features of periods <= 3 must not change
    df2 = df.copy()
    df2.loc[(df2.participant_id == "p1") & (df2.period > 3), "a"] = 1e9
    alt = add_history_features(df2, CFG)
    m = (base.participant_id == "p1") & (base.period <= 3)
    pd.testing.assert_frame_equal(base[m].reset_index(drop=True), alt[m].reset_index(drop=True))


def test_history_baseline_needs_min_history_and_is_per_participant():
    h = add_history_features(_toy(), CFG)
    p1 = h[h.participant_id == "p1"].set_index("period")
    assert np.isnan(p1.loc[2, "a__prior_mean"])            # only 1 earlier period < min_history=2
    assert p1.loc[3, "a__prior_mean"] == pytest.approx(1.5)  # mean(1,2)
    assert p1.loc[4, "a__delta_from_own_baseline"] == pytest.approx(4 - 2.0)
    p2 = h[h.participant_id == "p2"].set_index("period")
    assert p2.loc[3, "a__prior_mean"] == pytest.approx(15.0)  # no mixing across participants


def test_preprocessor_learns_only_from_fit_data():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(5, 2, 500), "b": rng.integers(0, 2, 500)})
    test = pd.DataFrame({"x": [1e6, -1e6], "y": [np.nan, 5.0], "b": [np.nan, 1]})
    pre = TabularPreprocessor().fit(train)
    med_before = pre.median_.copy()
    out = pre.transform(test)
    assert pre.median_.equals(med_before)                    # transform never changes fitted state
    assert np.isfinite(out.to_numpy()).all()
    assert out["x"].abs().max() < 10                         # test outliers clipped with TRAIN bounds
