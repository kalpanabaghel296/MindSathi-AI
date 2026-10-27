# Dataset comparison report

*Generated from the audits that were actually executed (primary Parquet + TSV, and the archive ZIP). WESAD was **not** accessible and was **not inspected**; statements about WESAD come from the official dataset description only.*

## 1. One-line verdict

| Dataset | What it really is | Role in this project |
|---|---|---|
| `anon_processed_df_parquet` + TSV | Wearable (Fitbit-style steps and sleep) **aggregates** per participant-period, with start and end PHQ-9 questionnaire scores | **Primary** - the only dataset suited to depression-related prediction |
| `archive (2).zip` | **StudentLife** (Dartmouth): phone sensing, EMA, app usage, calls, SMS, surveys for 49 users | **Supplementary / descriptive** - labels too few to train on |
| WESAD (local) | Chest + wrist physiology labelled baseline / stress / amusement | **Stress/affect experiment only** - no depression labels |

## 2. Primary dataset - what the files actually show

* **Identity.** The files do not state a dataset name. The content (wearable steps and sleep features, PHQ-9 outcomes, `phq9_*_start/end` for "the 3 month period") is consistent with the PSYCHE-D dataset you named, but that identification could not be confirmed from the files themselves.
* **Size:** 35,694 rows x 154 columns. All 154 TSV feature names match the Parquet columns exactly (nothing missing either way).
* **Unit of analysis:** the row index is `<participant>_<period>`: 4,948 participants, periods 1-12, median 8 rows per participant. No timestamps, no daily data.
* **Raw or engineered?** Everything is already aggregated: 66 statistic/static columns, 84 "linear regression" columns (score / intercept / coeff; none has a description in the TSV), and 4 PHQ-9 outcome columns. **There are no raw sensor streams.**
* **Labels.** PHQ-9 start and end scores are present together on 10,866 rows (30.4%), from 4,036 participants; 912 participants have no label at all. Labels appear only on periods 3, 6, 9 and 12 (3,244 / 2,832 / 2,732 / 2,058 rows). The category columns (0-4) match the standard PHQ-9 bands (0-4, 5-9, 10-14, 15-19, 20-27) on 100% of labelled rows.
* **Quality.** No exact duplicate rows, no duplicate (participant, period) pairs, no constant columns. Impossible values were few: 20 rows with adult weight below 80 lb (BMI under 5), and 1,206 weekday / 2,041 weekend rows where sleep is exactly 0 for both "asleep" and "in bed" (almost certainly the device was not worn). 10 columns are more than 50% missing.
* **Population.** The cohort includes people with cancer, MS, migraines and neuropathic pain. It is **not** a student sample like GLOBEM, so results should not be described as "students".
* **Important limitation.** The time windows of the "recent" (last 4 days) and "last 7 days" features are not documented relative to the questionnaire dates, and neither are those of the regression features. Behaviour features may therefore overlap the period in which the end-of-period PHQ-9 was answered. The recommended task is consequently **estimating** end-of-period symptoms from behaviour, not proven early prediction.

## 3. Archive dataset - StudentLife

* Identified from the files: folders `sensing/` (activity, audio, bluetooth, conversation, dark, gps, phonecharge, phonelock, wifi, wifi_location), `EMA/`, `app_usage/`, `call_log/`, `sms/`, `calendar/`, `dinning/`, `education/`, `survey/`, with per-user files `u00 ... u59` (49 users in sensing). About 3.04 GB uncompressed, 1,984 entries.
* **PHQ-9:** only a pre and a post questionnaire: 84 responses from 46 participants (38 have both). Answers are text ("Not at all", ...); they were scored 0-3 per item into totals and 0-27 severity bands (`outputs/studentlife/phq9_scored.csv`). Mean total: pre 5.5, post 6.3; 8 (pre) and 7 (post) participants score 10 or above.
* **Verdict:** 46 labelled people cannot train or fairly test a model. Its sensing is smartphone-based (GPS, screen lock, ...), not Fitbit-style steps and sleep.

## 4. Can any datasets be combined?

**No.** The three datasets stay separate.

| Pair | Why they must not be merged |
|---|---|
| Primary + StudentLife | Different participants and populations (patients vs students), different sensors and feature definitions, PHQ-9 timing differs (3-month start/end vs term pre/post), only 46 labelled people, no shared columns. Matching column names would not make them comparable. |
| Primary + WESAD | WESAD has stress/affect labels, not depression labels; lab-protocol recordings of 15 people vs. months of daily life. Stress is not depression. |
| StudentLife + WESAD | No shared participants, labels, sensors or protocol. |

**Allowed uses:** StudentLife for a small *descriptive* comparison or a sanity check on the survey scoring; WESAD for a separate, clearly labelled stress-detection experiment (leave-one-subject-out).

## 5. What was and was not verified

| Item | Status |
|---|---|
| Parquet / TSV audit, cleaning, targets, splits, preprocessing, validation | **Executed and validated** (26 checks pass; see `validation_report.csv`) |
| StudentLife inventory, header check, PHQ-9 scoring | **Executed** (sensing streams not parsed) |
| WESAD audit and feature extraction | **Code generated and smoke-tested on one SYNTHETIC subject only**; real WESAD **must be run locally** |
