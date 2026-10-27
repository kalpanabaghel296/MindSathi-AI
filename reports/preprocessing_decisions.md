# Preprocessing decisions (and why, in plain language)

Everything below was **executed** on the primary dataset unless marked otherwise. Settings live in `configs/config.yaml`.

## A. The prediction task I prepared the data for

**Target:** the questionnaire score `phq9_score_end` (0-27), plus a yes/no flag `target_ge10` (score >= 10, the usual PHQ-9 screening cut-off). A change score `target_delta = end - start` is also stored.

* **Unit of analysis:** one participant-period (a 3-month window).
* **Inputs:** behaviour/context features of that period; the start-of-period score `phq9_score_start` is kept in the `y_*` files as an optional baseline input, because a person's earlier score is legitimately known when you predict a later one.
* **Prediction time:** at the end of the period, before the end questionnaire is read.
* **Not a diagnosis:** a PHQ-9 score of 10 or more is a screening result, not a confirmed clinical diagnosis. Never describe the output as "depression detected".
* **Honest label of the task:** because the time windows of the behaviour features are undocumented, this is best called *estimating* end-of-period symptoms from behaviour, not proven forecasting.
* **Metrics to use later:** ROC-AUC and PR-AUC (the positive rate is ~32%), macro-F1, balanced accuracy for the yes/no flag; MAE and R2 for the score; always compare against the "just use the start score" baseline.

## B. Step by step

| # | Decision | Why |
|---|---|---|
| 1 | Parse `participant_id` and `period` from the row index; assert no duplicate pairs and that TSV and Parquet columns match | Participant IDs make participant-wise splitting possible; checks catch silent schema problems |
| 2 | **Set weight and BMI to missing** where weight < 80 lb (20 rows) | Impossible for an adult; BMI < 5 in every case |
| 3 | **Treat "0 minutes asleep AND 0 in bed" as device not worn** and set those sleep means to missing (1,206 weekday, 2,041 weekend rows) | A real person is never in bed for 0 minutes; zero would teach the model "sleeps nothing" |
| 4 | **Keep but flag** `steps_awake_mean < 500` (20 rows), BMI > 60 (178 rows) | Unusual but possible; deleting them without evidence would remove real (possibly sicker) people |
| 5 | Do **not** drop rows with missing values and do **not** fill with zero | Missingness is informative here (e.g. 86% missing `steps_mvpa_iqr`) |
| 6 | Target validation: scores in 0-27, categories recomputed from scores and compared | 100% of categories matched the standard bands |
| 7 | **Leakage control:** `phq9_score_end`, `phq9_cat_end` and `phq9_cat_start` are never inputs | The first two are the answer; the third is derived from the start score (kept as a score instead) |
| 8 | **Sensitive columns** (`sex`, `race_*`) are not model inputs; saved in `meta_*.parquet` for fairness checks | Reduces the risk of a model learning demographics; still lets you measure gaps between groups. Change in config if you decide otherwise |
| 9 | **Timing-uncertain features** (20 after de-duplication: the "recent / last 7 days" ones and the "past month" medication/lifestyle flags) are kept out of the main `X_*` files and saved separately as `X_*_timing_uncertain_extra.parquet` | You do not know which dates they cover, so they could sit next to the outcome. Use them only as a labelled sensitivity test |
| 10 | Split by participant (2,835 train / 593 val / 608 test participants), stratified on the yes/no target | No person appears in two splits; positive rate is 31.7% in all three |
| 11 | **Fit everything on training rows only:** clip bounds (0.5%-99.5%), medians/modes, scaling, missing-indicators, constant and near-duplicate detection | Otherwise information from validation/test leaks into training |
| 12 | Winsorise (clip) at training quantiles instead of deleting outliers | Keeps every row; limits the pull of extreme slopes |
| 13 | Median imputation + **missing-indicator columns** for features missing more than 5% in training (71 candidates; identical patterns collapsed into 18) | Keeps the "was it missing?" signal without duplicates |
| 14 | Robust scaling `(x - median) / IQR` for continuous columns; binary columns untouched | Heavy-tailed data; median/IQR are not distorted by outliers |
| 15 | Drop 3 near-duplicate columns (|r| >= 0.98) | `sleep_ratio_asleep_in_bed__intercept`, its `_` twin, and `sleep_ratio_asleep_in_bed_mean_recent` duplicate other columns |
| 16 | Class imbalance: **class weights saved** (`class_weights.json`, from train only); no resampling applied to the files | Weighting happens inside model training, so it cannot leak into validation/test |
| 17 | Validation: 26 independent checks on the written files, including refitting on train IDs to confirm nothing was fit on val/test | See `reports/validation_report.csv` |

## C. Personal baseline and temporal features

* **What the data supports:** several ordered rows per participant, so "compare this period with the same person's earlier periods" is possible.
* **What it does not support:** no timestamps and no daily records, so a **2-4 week daily baseline, daily/weekly series and day-level trends cannot be built**, and I did not fabricate them.
* **What was built (experimental, optional):** for 5 core sleep/steps features, per participant: previous period value, change from previous, mean of **strictly earlier** periods (needs >= 2), and change from that personal baseline. Test `test_history_features_never_use_future_rows` proves the future is never read. Coverage: 90.6% of labelled rows have >= 2 earlier periods. These live in `outputs/temporal/`, not in `X_*`, because they depend on the *assumption* that the period number is an ordered, roughly monthly step.

## D. WESAD and StudentLife

* **StudentLife:** inventoried and PHQ-9 scored only; kept completely separate (see comparison report).
* **WESAD (`src/wesad_preprocessing.py`) - generated, not run on real data.** 60-second windows, 30-second step; windows kept only when >= 80% of the 700 Hz labels agree and the label is baseline, stress or amusement; all signals cut by time so sampling rates stay aligned; invalid readings masked; one subject in memory at a time; subject id kept for evaluation but excluded from features; use leave-one-subject-out. Heart-rate features use a simple peak detector (not clinical-grade).

## E. Known limitations (read before modelling)

1. Undocumented time windows (see A and 9). 2. 84 regression columns have no description in the TSV. 3. Period meaning (monthly) is inferred. 4. Population has chronic conditions; not students. 5. Static context features (money, migraine diagnosis) correlate most with the score (about 0.34) - a model may lean on context rather than behaviour. 6. Self-reported PHQ-9 is a questionnaire, not a diagnosis. 7. Race/sex fairness has to be evaluated using `meta_*.parquet`.
