# Depression-preprocessing

Leakage-safe preprocessing for **"AI-Based Early Detection and Support System for Depression Using Behavioral and Wearable Data"**.
Uses exactly three datasets: the primary Parquet (+ feature TSV), the archive ZIP (identified as StudentLife) and WESAD (local).
**No models are trained here.** Outputs are risk indicators for research, never diagnoses.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. put the raw files in data/raw/  (or edit paths.raw_dir in configs/config.yaml):
#      anon_processed_df_parquet
#      Feature_explanations_-_anonymized_data_-_Sheet1.tsv
#      archive (2).zip                      (rename in config if your file name differs)
python -m src.audit                         # primary audit + reports
python -m src.studentlife_audit             # archive audit + PHQ-9 scoring
python -m src.preprocess                    # clean -> targets -> split -> fit on train -> export -> validate
python -m pytest -q tests                   # unit + synthetic checks
```

## WESAD (kept outside this ZIP - ~2.5 GB)

```bash
# set paths.wesad_dir in configs/config.yaml, or pass it on the command line
python -m src.wesad_preprocessing --wesad-dir /path/to/WESAD --audit-only     # structure, sampling rates, labels
python -m src.wesad_preprocessing --wesad-dir /path/to/WESAD                  # window features, one subject at a time
python -m src.wesad_preprocessing --wesad-dir /path/to/WESAD --subjects S2    # try one subject first
```
Expected layout: `WESAD/S2/S2.pkl ... S17/S17.pkl` (S1 and S12 do not exist). Outputs go to `outputs/wesad/`.
WESAD labels (baseline / stress / amusement) are **never** merged with PHQ-9 targets.

## What was executed vs not

| Part | Status |
|---|---|
| `src/audit.py`, `src/preprocess.py`, `src/feature_engineering.py`, `src/studentlife_audit.py` | **Executed** on the real files; 26 output checks pass |
| Notebooks 01 and 02 | **Executed** (outputs saved inside; they print the temporary config path used for that run) |
| `tests/test_primary.py` (3) and `tests/test_wesad_synthetic.py` (2) | **Executed**, 5 passed |
| `src/wesad_preprocessing.py`, notebook 03 | **Generated; smoke-tested only on one fabricated synthetic subject. Not run on real WESAD.** Run locally |
| StudentLife sensing streams (~3 GB) | Inventoried and headers read; **not parsed** |

The executed run used the uploaded file names (`archive__2_.zip`); the shipped config uses `archive (2).zip`.

## Layout

```
configs/config.yaml          all paths and decisions (timing-uncertain list, sensitive columns, windows ...)
src/audit.py                 primary audit -> reports/
src/preprocess.py            pipeline + TabularPreprocessor + validation
src/feature_engineering.py   history/baseline features (only earlier periods)
src/studentlife_audit.py     archive audit + PHQ-9 scoring (kept separate)
src/wesad_preprocessing.py   WESAD audit + window features (run locally)
notebooks/                   01 audit, 02 primary preprocessing, 03 WESAD (not executed)
reports/                     inventory, dictionary, missing/duplicate/invalid reports, comparison + decisions
outputs/primary/             X_*/y_*/meta_* (train, val, test), cleaned dataset, preprocessor.joblib, feature lists
outputs/studentlife/         phq9_scored.csv
outputs/temporal/            experimental history features
```

## Using the primary outputs

* `X_{train,val,test}.parquet` - the main model inputs (scaled). `final_feature_list.txt` lists the columns.
* `y_*.parquet` - `phq9_score_end` (target), `target_ge10`, `target_delta`, and `phq9_score_start` (optional baseline input), with `participant_id` and `period`.
* `meta_*.parquet` - sex, race, birth year for **fairness reporting only**.
* `X_*_timing_uncertain_extra.parquet` - extra features whose time window is undocumented; join only for a sensitivity test.
* `class_weights.json` - from train only; apply during model fitting.
* Load the fitted pipeline with `joblib.load("outputs/primary/preprocessor.joblib")` (run from the project root so `src` is importable); its `preprocessor` entry is a `TabularPreprocessor`.
* Evaluate with the participant-wise splits as given; for cross-validation use `GroupKFold` on `participant_id`.

## Key limitations

Time windows of "recent / last 7 days" and regression features are undocumented; the period index is assumed monthly; 84 columns have no TSV description; the cohort has chronic conditions (not students); PHQ-9 is a screening questionnaire. See `reports/preprocessing_decisions.md`.
