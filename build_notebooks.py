"""Generates the three notebooks (thin wrappers around src/). Run once: python build_notebooks.py"""
import nbformat as nbf

HEAD = '''import os, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd
pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 80)
from src.utils import load_config, reports_dir, outputs_dir
cfg = load_config()          # uses $DEPRESSION_CONFIG if set, else configs/config.yaml
print("config:", cfg["_config_path"])'''

def nb(cells, path):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, path)

nb([
 ("md", "# 01 - Dataset audit\nRead-only audit of the three datasets. The primary Parquet and the StudentLife archive are audited here; **WESAD is audited locally** with `src/wesad_preprocessing.py --audit-only` (see notebook 03)."),
 ("code", HEAD),
 ("md", "## Primary dataset (Parquet + TSV)"),
 ("code", "from src.audit import audit_primary, assemble_inventory\nsummary = audit_primary(cfg)\npd.Series({k: v for k, v in summary.items() if not isinstance(v, (dict, list))})"),
 ("code", "print('labels exist only on periods:', summary['label_period_distribution'])\nprint('PHQ-9 category columns match the standard bands:', summary['category_matches_standard_bands'])\nprint('TSV <-> parquet name mismatches:', summary['tsv_names_missing_in_parquet'], summary['parquet_columns_missing_in_tsv'])"),
 ("code", "rep = reports_dir(cfg)\npd.read_csv(rep / 'invalid_values_report.csv')"),
 ("code", "pd.read_csv(rep / 'missing_values_report.csv').head(15)"),
 ("code", "pd.read_csv(rep / 'redundant_features.csv')"),
 ("code", "d = pd.read_csv(rep / 'feature_data_dictionary.csv')\nprint(d.role.value_counts()); print(d.tsv_subcategory.value_counts())\nd.head()"),
 ("md", "## Archive dataset (identified as StudentLife)"),
 ("code", "from src.studentlife_audit import audit_studentlife\nsl = audit_studentlife(cfg)\nprint(sl['identified_as']); print(sl['phq9']); print(sl['merge_with_primary'])\npd.read_csv(rep / 'studentlife_inventory.csv')"),
 ("md", "## Combined inventory"),
 ("code", "assemble_inventory(cfg)"),
], "notebooks/01_dataset_audit.ipynb")

nb([
 ("md", "# 02 - Primary preprocessing\nRuns the leakage-safe pipeline in `src/preprocess.py` and shows each decision's evidence. **No models are trained.**"),
 ("code", HEAD),
 ("code", "from src.preprocess import run_pipeline, validate_outputs\nres = run_pipeline(cfg, verbose=False)\nres['clean_log']"),
 ("md", "### Participant-wise splits (no participant appears twice)"),
 ("code", "res['split_summary']"),
 ("md", "### Features: what was excluded and why"),
 ("code", "import json\nout = outputs_dir(cfg, 'primary')\nfs = json.load(open(out / 'feature_sets.json'))\nprint('main feature columns:', len(fs['main_feature_columns']))\nprint('timing-uncertain extras (separate file):', len(fs['timing_uncertain_extra_columns']))\nprint('excluded outcome-period columns:', fs['excluded_outcome_columns'])\nprint('sensitive (fairness only):', fs['sensitive_fairness_only'])\nprint('dropped near-duplicates (dropped -> kept):', fs['dropped_correlated(dropped->kept)'])\nprint('missing indicators:', len(fs['missing_indicator_columns']))"),
 ("md", "### Independent validation of the written files"),
 ("code", "validate_outputs(cfg)"),
 ("md", "### Personal-baseline / history features (experimental)\nOnly strictly earlier periods of the same participant are used. No daily data exists, so a 2-4 week daily baseline is **not** reconstructed."),
 ("code", "from src.feature_engineering import history_coverage\nhist = pd.read_parquet(outputs_dir(cfg, 'temporal') / 'temporal_features_experimental.parquet')\nc = pd.read_parquet(out / 'cleaned_primary_dataset.parquet')\nlab = c.loc[c.is_labelled, ['participant_id', 'period']]\nhistory_coverage(hist, lab, cfg['temporal']['min_history'])"),
 ("code", "import pytest, subprocess\nsubprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests' / 'test_primary.py')], cwd=ROOT)"),
], "notebooks/02_primary_preprocessing.ipynb")

nb([
 ("md", "# 03 - WESAD preprocessing (RUN LOCALLY)\nWESAD (~2.5 GB) is **not** in the project ZIP and was **not available** when this project was generated, so this notebook was **not executed** by the author. Set `WESAD_DIR` and run the cells.\n\nWESAD is a stress/affect dataset. Its labels (baseline / stress / amusement) are never merged into the PHQ-9 depression targets."),
 ("code", HEAD),
 ("code", "WESAD_DIR = Path(cfg['paths']['wesad_dir'])    # edit configs/config.yaml or overwrite here, e.g. Path('/Users/me/data/WESAD')\nprint(WESAD_DIR, 'exists:', WESAD_DIR.is_dir())"),
 ("md", "## Step 1 - structure audit only (fast)"),
 ("code", "from src.wesad_preprocessing import run\nOUT = outputs_dir(cfg, 'wesad')\nrep = run(WESAD_DIR, OUT, cfg, audit_only=True)\nprint('subjects found:', rep['subjects_found'])\nprint('missing expected:', rep['missing_expected_subjects'])\npd.read_csv(OUT / 'wesad_signal_inventory.csv').groupby(['device', 'signal']).agg(fs_hz=('inferred_fs_hz', 'median'), nonfinite=('n_nonfinite', 'sum'), flatline_subjects=('flatline', 'sum'))"),
 ("md", "## Step 2 - window feature extraction (one subject at a time)\nTip: try `subjects=['S2']` first."),
 ("code", "rep = run(WESAD_DIR, OUT, cfg, subjects=None, audit_only=False)\nprint('windows:', rep['total_windows'], '| features:', rep['n_features'])\nrep['label_distribution_total']"),
 ("code", "feats = pd.read_parquet(OUT / 'wesad_windows_features.parquet')\nfeats.groupby('label_name').size()"),
 ("md", "## Step 3 - stress experiment preparation (supplementary, NOT depression)\nUse **leave-one-subject-out** (`loso_folds.csv`). Do not normalise per subject with label-derived baselines."),
 ("code", "import json\ncols = json.load(open(OUT / 'feature_columns.json'))['feature_columns']\nprint(len(cols), 'feature columns; subject is kept only for splitting')\nfeats[['subject', 'label_binary_stress'] + cols[:5]].head()"),
], "notebooks/03_wesad_preprocessing.ipynb")
print("notebooks written")
