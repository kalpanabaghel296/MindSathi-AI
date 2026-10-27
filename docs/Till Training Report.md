# AI-Based Early Detection and Support System for Depression: Project Report

**Authors:** Kalpana Baghel and Priyanshi Mehta, B.Tech CS-AI, JK Lakshmipat University
**Report date:** 24 October 2026
**Status:** work in progress. Sections marked **PENDING** have not been run or finished yet.

> **Important.** This system is a research prototype. It does **not diagnose** depression or any other condition. It estimates a *screening risk indicator* (the chance that a PHQ-9 questionnaire score will be 10 or more) and offers supportive, non-clinical conversation. It must not replace a doctor, counsellor or helpline.

---

## 1. Summary

We built a pipeline in five connected parts:

| # | Part | Status |
|---|---|---|
| 1 | Data audit and preprocessing (three datasets, kept separate) | Done and tested |
| 2 | Early-detection model: probability of PHQ-9 ≥ 10 at the end of a 3-month period | Baseline run done; advanced tuning **PENDING** |
| 3 | Explainability (top factors per person) | Code written; used when an explainable model is chosen |
| 4 | Companion bot "Saathi" with safety layer, optional LLM, starter RAG, dashboard | Built and evaluated on scripted cases; **no real-user study, no counsellor review yet** |
| 5 | EHR prototype on synthetic FHIR data | **Not built** |

### Main findings so far
1. Wearable and context data alone gave a **test ROC-AUC of 0.785** for predicting PHQ-9 ≥ 10.
2. A **baseline that uses only the person's starting PHQ-9 score scored higher (0.887)**. So behaviour data did not beat the simple baseline in this first run. We report this as it is.
3. The companion's safety layer flagged **all 23 scripted crisis conversations** (English, Hindi, Hinglish) in our own test set. This is an optimistic number (see section 7), not proof of real-world safety.

---

## 2. Data

Only three datasets are used. They are **never merged**, because they differ in population, sensors, feature definitions and label timing.

| Dataset | What it is | Role |
|---|---|---|
| Wearable aggregates (parquet + feature-explanation TSV) | 35,694 rows × 154 columns, 4,948 participants, steps and sleep summaries per participant-period, with start and end PHQ-9. Consistent with the PSYCHE-D dataset, but the files do not name it | **Primary** (model training) |
| StudentLife (Dartmouth) | Phone sensing, EMA, surveys; 49 users | Supplementary, descriptive only |
| WESAD | Chest and wrist physiology with baseline / stress / amusement labels | Separate stress experiment (**PENDING**, code not yet run on real data) |

Facts about the primary dataset (from executed audits):
- One row per participant-period (periods 1 to 12). No timestamps, no daily data, no raw sensor streams.
- PHQ-9 start and end are both present on **10,866 rows (30.4%)** from 4,036 participants. Labels exist only for periods 3, 6, 9 and 12. The "3-month" meaning of a period is an inference, not documented.
- The cohort includes people with chronic conditions (for example cancer, MS, migraine). It is **not a student sample**.
- 84 regression-based feature columns have no description in the TSV.

---

## 3. What the model predicts

- **Target:** `phq9_score_end` (0 to 27) and the flag `target_ge10` (score ≥ 10, the standard screening cut-off).
- **Output shown to users:** a calibrated probability, turned into a risk level: **low, moderate or high**.
- **Safety override:** a positive PHQ-9 item 9 (thoughts of self-harm) or crisis wording in the chat always gives **high**, regardless of the model.
- **Honest wording:** because some feature time windows are undocumented, this is "estimating end-of-period symptoms from behaviour", not proven forecasting.

---

## 4. Preprocessing

Key decisions (all fitted on training data only):

- **Cleaning:** only provably invalid values changed (20 impossible adult weights set to missing; device-not-worn sleep of 0 treated as missing). Nothing filled with zero, no rows dropped for missing values.
- **Leakage control:** `phq9_score_end`, `phq9_cat_end` and `phq9_cat_start` are never inputs. 20 features with undocumented time windows are kept out of the main set and exported separately for a sensitivity check.
- **Fairness control:** `sex` and `race_*` are not model inputs; they are stored separately for fairness checks.
- **Splits:** participant-wise and stratified (no person appears in two splits).

| Split | Rows | Participants | Positive rate (PHQ-9 ≥ 10) |
|---|---|---|---|
| Train | 7,610 | 2,835 | 31.7% |
| Validation | 1,630 | 593 | 31.7% |
| Test | 1,626 | 608 | 31.7% |

- **Transformations (train only):** quantile clipping, median imputation, missing-indicator columns, robust scaling.
- **Final feature set:** 139 columns (121 features + 18 missing indicators).
- **Class imbalance:** class weights (0.73 / 1.58; XGBoost `scale_pos_weight` 2.16). No resampling.
- **Validation:** 26 automated checks pass; unit tests for the preprocessing pass.

---

## 5. Risk model

### 5.1 First training run (`src/train_models.py`), executed on the real splits

Three models were trained on the training set and compared on the validation set. The best was calibrated (isotonic regression on validation), thresholds were set on validation, and the **test set was used once**.

| Model | Validation ROC-AUC |
|---|---|
| Logistic Regression | 0.7523 |
| HistGradientBoosting (chosen) | 0.7556 |
| XGBoost | 0.7529 |
| *Baseline: start PHQ-9 score only* | *0.8823* |

**Test results (chosen model):**

| Metric | Value | Note |
|---|---|---|
| ROC-AUC | 0.785 | 0.5 is chance |
| PR-AUC | 0.614 | chance level is about 0.317 |
| Brier score | 0.168 | lower is better; predicting the average for everyone gives about 0.217 |
| Macro-F1 at the moderate threshold | 0.676 | |
| Baseline (start score only) ROC-AUC | **0.887** | higher than the model |
| Share of test rows by level | low 49%, moderate 28%, high 23% | |

**Reading the result:** the model finds real signal in behaviour and context data (well above chance), but a person's own earlier PHQ-9 score predicts the later score better. Wearable data alone did not beat that baseline.

### 5.2 Fairness check (test AUC by group)

| Group | n | AUC |
|---|---|---|
| sex = 0 | 1,244 | 0.764 |
| sex = 1 | 379 | 0.789 |
| race_asian = 1 | 68 | 0.766 |
| race_black = 1 | 95 | 0.871 |
| race_hispanic = 1 | 78 | 0.893 |
| race_other = 1 | 63 | 0.838 |
| race_white = 1 | 1,413 | 0.771 |

Several groups are small (n = 63 to 95), so their AUCs vary a lot and **no confident fairness conclusion can be drawn**. The sex gap is small. The group "race_white = 0" (n = 213) is a mixture of all other groups.

### 5.3 Advanced training (`src/train_advanced.py`)

The advanced script compares two input variants (behaviour only; behaviour plus the person's starting PHQ-9 score) across Logistic Regression, HistGradientBoosting, XGBoost (GPU) and a PyTorch MLP (GPU, with epochs and early stopping). Hyper-parameters are tuned by participant-wise 5-fold cross-validation on train and validation only; the test set is used once at the end. It also produces a CV comparison chart, a loss curve, a learning curve and a calibration plot.

**Status:** a quick-mode run started on an NVIDIA GPU. Partial numbers seen so far:

| Item | CV ROC-AUC |
|---|---|
| Baseline (start score only) | 0.8813 |
| behaviour_only, Logistic Regression (quick mode) | 0.7573 ± 0.0128 |

**PENDING:** the full comparison table, the chosen variant and model, test results, and figures. These will be added after the full run. If the "behaviour + start score" variant wins, the model needs a recent PHQ-9 as input, which is a different claim from "early detection from wearables alone", and the report must say so.

---

## 6. The companion: Saathi

Saathi is a listen-first AI companion in the dashboard. It is **not a therapist** and says it is an AI.

![Saathi chat screenshot](images/saathi_chat.png)

*Screenshot of the Talk to Saathi page. The user writes in Hinglish about tension, poor sleep and low motivation; Saathi (running with an LLM) replies in Hinglish with a short grounding exercise. The reply is cut off in the screenshot. Reply length and pacing (listening and asking before advising) are tuning targets.*

### 6.1 How a reply is produced

1. **Safety check first, before anything else.** Rule-based word lists for English, Hindi (Devanagari) and Hinglish. The lists deliberately **over-trigger** (a false alarm costs one extra kind message; a miss costs far more).
2. **Crisis:** a fixed, non-generated message with Tele-MANAS (14416 or 1-800-891-4416) and 112, plus call buttons. The LLM is never used for these. After a crisis message, the next replies stay in a careful mode.
3. **Medical or diagnosis questions:** a fixed polite refusal (no diagnosis, no medicine advice) and an offer to help the person prepare what to tell a doctor.
4. **Normal messages:** an LLM (optional; Groq, Gemini, OpenRouter, Ollama or Anthropic via environment variables) with a counsellor-style system prompt. Its output is rejected if it sounds like diagnosis or medicine advice. If no LLM is configured or the call fails, built-in listen-first templates (English, Hindi, Hinglish) are used.
5. **Reference passages (RAG):** a small TF-IDF retriever adds relevant self-care passages to the LLM prompt. If nothing relevant is found it adds nothing.

### 6.2 Other dashboard features
- **Today:** daily mood and loneliness check-in, 14-day chart, and a comparison of the last 7 days with the person's **own** earlier 2 weeks (a gentle nudge on a sustained decline).
- **Risk insights:** held-out test participants replayed as simulated users, with risk level, probability gauge, top factors (when the model type supports them) and level-specific support suggestions. Shows the baseline comparison openly.
- **Help now:** helplines and a breathing exercise. **Privacy and data:** consent switch and delete-everything button.
- **Voice:** speech-to-text input and read-aloud, for elderly users.
- **Privacy design:** chat text is not stored on the server; only the daily check-in is stored locally.

---

## 7. Evaluation of Saathi

`src/eval_saathi.py` runs 64 scripted conversations and 20 retrieval queries and writes a report plus a spreadsheet for counsellor review. Results with the built-in templates (no LLM):

| Metric | Result |
|---|---|
| Crisis recall (English, Hinglish, Hindi, multi-turn) | 100% (23 of 23) |
| False alarms on normal and distress messages | 0% (0 of 26) |
| Figures of speech such as "dead tired" wrongly flagged | 0% (0 of 6) |
| Diagnosis or medicine questions given the refusal | 100% (9 of 9) |
| Replies free of diagnosis or medicine wording, at most one question | 100% (81 of 81) |
| Reply language matches the user's language | 100% |
| Repetition inside 6-turn scripts | 6 of 6 replies distinct, no repeated sentences |
| RAG: correct passage ranked first (in-scope questions) | 100% (14 of 14) |
| RAG: nothing retrieved for unrelated questions | 100% (6 of 6) |

**The evaluation found real problems before it passed.** It caught an unrecognised spelling of a distress phrase, a missed diagnosis-style question in Hinglish, and repeated replies within a conversation. All three were fixed.

### Caveats (please read)
- The test cases were **written by the project team**, and the rules were fixed after seeing them. **100% is therefore optimistic.** It works as a regression check, not as proof of safety. Crisis wording has endless variants.
- The knowledge base (`knowledge/starter_passages.json`) is **starter text marked UNVERIFIED**. Retrieval quality says nothing about whether the passages are correct. They must be replaced with passages from cited sources (for example WHO, NIMHANS, Tele-MANAS, college counselling) before the system is called evidence-based.
- The retrieval margin is thin (lowest in-scope score 0.131; highest out-of-scope score 0.128). Multilingual embeddings are the planned upgrade.
- **PENDING:** evaluation with the live LLM, fresh test cases from a counsellor (aim for 100 or more, with many Hinglish crisis phrasings), and counsellor ratings of replies.

---

## 8. Limitations

1. Some feature time windows are undocumented, so behaviour features may overlap the period in which the end questionnaire was answered.
2. The meaning of the period index (monthly) is inferred.
3. The cohort includes people with chronic conditions; results must not be described as a student sample.
4. PHQ-9 is a self-reported screening questionnaire, not a diagnosis.
5. Context features (for example money worries, migraine history) may dominate the model.
6. The model was trained on Fitbit-style aggregates; the dashboard replays held-out participants as simulated users. It has **not** been tested on live users.
7. Fairness across sex and race has been checked only on small groups.
8. The crisis word lists are a first draft and need expert review.
9. The mood database is not encrypted (prototype). Consent sharing with a trusted person is recorded but **not implemented**.
10. WESAD code has been run only on synthetic data; StudentLife sensing streams are not parsed.
11. Tele-MANAS numbers were checked online on 21 October 2026 and must be re-verified before any real use.

## 9. Methodology changes from the original proposal
- The planned primary dataset (GLOBEM) was not available. The wearable parquet is now primary; StudentLife and WESAD are separate supporting datasets.
- The data has no timestamps, so the 2 to 4 week daily personal baseline could not be built. "Change from own baseline" uses earlier-period history features instead (experimental), and the dashboard check-in trend uses the person's own earlier check-ins.
- LSTM was replaced by history features with gradient boosting, because there are no daily sequences and at most about 4 labelled steps per person. A neural network (MLP) is included in the advanced comparison.

## 10. Next steps
1. Finish the full `train_advanced` run; add its table and figures to section 5.3.
2. Run an ablation with the timing-uncertain and history features.
3. Replace the starter knowledge base with cited sources; test retrieval and refusals again.
4. Get counsellor review of word lists, replies and a fresh evaluation set; evaluate with the live LLM.
5. Build the EHR prototype on synthetic FHIR (Synthea) data.
6. Run `src/wesad_preprocessing.py` on real WESAD as a separate stress experiment.

## 11. How to reproduce

```bash
pip install -r requirements.txt            # PyTorch with CUDA: see pytorch.org
# raw files go in data/raw/ (not included in the repository)
python -m src.audit
python -m src.studentlife_audit
python -m src.preprocess
python -m src.train_models                 # first model run
python -m src.train_advanced --quick       # then the full run without --quick
python -m src.eval_saathi --no-llm
python -m app.server                       # dashboard at http://127.0.0.1:5000
```

Optional LLM for Saathi (set in the terminal, never in code or on GitHub): `LLM_PROVIDER` (gemini, groq, openrouter, ollama, anthropic), `LLM_API_KEY`, `LLM_MODEL`.

## 12. Ethics, privacy and safety statement
- The system never diagnoses and never gives medicine advice.
- Serious cases lead to a human: fixed crisis message, helpline, and (with prior consent, not yet implemented) a trusted contact.
- Do **not** upload raw datasets, `outputs/primary/*.parquet` with real participant rows, the local database (`app/instance/*.db`) or any API key to a public repository. Check the dataset licences before sharing any derived data.
- Users should be told clearly that Saathi is an AI.
