# TinyReproduction: WAN / Network Anomaly Detection

Reproduction of core results from:

> Schummer, P., del Rio, A., Serrano, J., Jimenez, D., Sánchez, G., & Llorente, Á. (2024).
> *Machine Learning-Based Network Anomaly Detection: Design, Implementation, and Evaluation.*
> AI, 5(4), 2967–2983. https://www.mdpi.com/2673-2688/5/4/143

The goal is to reproduce the **relative performance trends** reported in the paper using a
Tukey-IQR labeling rule, supervised baselines (Logistic Regression, Random Forest, SVM RBF),
a PyTorch MLP extension, and SHAP-based feature importance.

---

## Code Authorship Summary

All code in this repository was **written from scratch** for this project. No code was copied
from the reference paper's repository or any external repository. Standard library/framework
APIs (scikit-learn, PyTorch, SHAP) were used per their official documentation. See the
[Code Provenance](#code-provenance) section for per-file details and edited line numbers.

---

## Project Structure

```
config/                     config.json — all tunable parameters
data/
  raw/                      raw telemetry CSV (auto-generated or real)
  processed/                feature-engineered parquet dataset
scripts/
  make_synthetic_data.py    generate synthetic WAN telemetry (Step 1)
  build_dataset.py          Tukey-IQR labeling + feature engineering (Step 2)
  train_models.py           train & evaluate all models (Step 3)
  make_shap.py              SHAP feature-importance plots (Step 4)
src/wan_anomaly/
  processing/
    label.py                Tukey-IQR anomaly labeling
    features.py             rolling window & time features
    split.py                time-aware train/test split
  models/
    train.py                classical ML: LogReg, RF, SVM
    mlp_torch.py            PyTorch MLP classifier (added CP2)
  evaluation/
    reports.py              PR-AUC and anomaly-rate comparison plots
  explain/
    shap_plots.py           SHAP bar and beeswarm plots
  utils/
    io.py                   table I/O (CSV / Parquet)
tests/                      pytest unit tests
artifacts/
  metrics_table.csv         accuracy, PR-AUC, F1, ROC-AUC per model
  anomaly_rates.csv         predicted vs true anomaly rate per model
  run_summary.json          train/test split info and true anomaly rate
  models/                   saved model checkpoints (.joblib, .pt)
  plots/                    generated figures (PR-AUC, SHAP, etc.)
```

---

## Dependencies

**Python**: 3.9 or later (tested on 3.9 and 3.12).

All required packages are listed in `requirements.txt`:

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24 | Numerical operations, random seed control |
| `pandas` | ≥2.0 | Telemetry data loading and feature engineering |
| `scikit-learn` | ≥1.3 | LogReg, Random Forest, SVM pipelines and metrics |
| `matplotlib` | ≥3.7 | PR-AUC and anomaly-rate comparison plots |
| `joblib` | ≥1.3 | Model serialization (`.joblib` files) |
| `pyarrow` | ≥14.0 | Parquet read/write for processed dataset |
| `torch` | any recent | PyTorch MLP training (`BCEWithLogitsLoss`) |
| `shap` | ≥0.44 | SHAP TreeExplainer for Random Forest |

Install all dependencies with:
```bash
pip install -r requirements.txt
```

No GPU is required; all experiments run on CPU in 5–10 minutes.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Step 1 — Generate synthetic WAN telemetry (14 days, 20 sites, 15-min intervals)
python scripts/make_synthetic_data.py \
    --out data/raw/telemetry.csv \
    --days 14 --sites 20 --freq-min 15

# Step 2 — Build dataset: Tukey-IQR labels + rolling features + 15% label noise
python scripts/build_dataset.py \
    --in data/raw/telemetry.csv \
    --out data/processed/dataset.parquet

# Step 3 — Train & evaluate all models (LogReg, RF, SVM RBF, PyTorch MLP)
python scripts/train_models.py \
    --data data/processed/dataset.parquet \
    --out artifacts

# Step 4 — Generate SHAP feature importance plots (Random Forest)
python scripts/make_shap.py \
    --data data/processed/dataset.parquet \
    --model artifacts/models/rf.joblib \
    --outdir artifacts/plots
```

Total wall-clock time on a single CPU: approximately 5–10 minutes.
All steps are deterministic given `random_state = 42` in `config/config.json`.

---

## Dataset

### Synthetic data (default — no download required)

`scripts/make_synthetic_data.py` generates a self-contained synthetic dataset with realistic
WAN telemetry across three link types (MPLS, Broadband, LTE/5G). Anomalies are injected as
gradual multi-metric degradation events. Running Step 1 above produces `data/raw/telemetry.csv`.

### Real telemetry dataset (optional)

The reference paper uses a proprietary enterprise telemetry dataset that is not publicly
available. As an alternative, the pipeline accepts any CSV with matching column names
(`timestamp`, `site_id`, `link_id`, and the five metric columns listed in `config/config.json`).

Public network telemetry datasets (e.g., from Kaggle) can be used by renaming columns to match
the schema and running `scripts/build_dataset.py` on the file. The Kaggle CLI requires an account
and API token; see https://www.kaggle.com/docs/api for setup instructions.

---

## Outputs

| File | Description |
|---|---|
| `artifacts/metrics_table.csv` | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC per model |
| `artifacts/anomaly_rates.csv` | Predicted vs true anomaly rate per model |
| `artifacts/run_summary.json` | Train/test split cutoff and true anomaly rate |
| `artifacts/plots/pr_auc.png` | PR-AUC comparison bar chart (analogous to paper Table III) |
| `artifacts/plots/accuracy.png` | Accuracy comparison bar chart |
| `artifacts/plots/anomaly_rates.png` | Predicted vs true anomaly rate |
| `artifacts/plots/shap_rf_bar.png` | SHAP mean absolute feature importance |
| `artifacts/plots/shap_rf_beeswarm.png` | SHAP beeswarm summary plot |
| `artifacts/models/logreg.joblib` | Saved Logistic Regression sklearn pipeline |
| `artifacts/models/rf.joblib` | Saved Random Forest sklearn pipeline |
| `artifacts/models/svm_rbf.joblib` | Saved SVM RBF sklearn pipeline |
| `artifacts/models/mlp_torch.pt` | Saved PyTorch MLP state dict + metadata |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Three test modules cover labeling (`test_labeling.py`), feature engineering
(`test_features.py`), and the time-aware split (`test_split.py`).

---

## Reproducibility

- All random seeds are fixed via `config/config.json` (`"random_state": 42`).
- Train/test split is time-aware (chronological 70/30): training data is strictly earlier
  than test data, preventing any look-ahead leakage.
- Synthetic data generation uses `numpy.random.default_rng(seed)` for deterministic output.
- PyTorch seeds are set with `torch.manual_seed(random_state)` and `np.random.seed(random_state)`
  at the start of `train_mlp_classifier` (`src/wan_anomaly/models/mlp_torch.py`, lines 94–95).

---

## Code Provenance

This section documents the authorship and origin of every file per course submission requirements.

### Written from scratch for this project

All source files were written from scratch for this project. No code was copied from external
repositories.

| File | Lines | Description |
|---|---|---|
| `scripts/make_synthetic_data.py` | 1–119 | Synthetic WAN telemetry generator with realistic anomaly injection |
| `scripts/build_dataset.py` | 1–69 | Dataset build pipeline (labeling + feature engineering + label noise) |
| `scripts/train_models.py` | 1–146 | Model training and evaluation orchestration (classical + MLP) |
| `scripts/make_shap.py` | 1–85 | SHAP explanation script (bar + beeswarm plots) |
| `src/wan_anomaly/processing/label.py` | 1–32 | Tukey-IQR anomaly labeling with device-level grouping |
| `src/wan_anomaly/processing/features.py` | 1–66 | Rolling window features, lag/delta features, cyclical time features |
| `src/wan_anomaly/processing/split.py` | 1–11 | Time-aware train/test split |
| `src/wan_anomaly/models/train.py` | 1–117 | Classical ML training: LogReg, RF, SVM pipelines |
| `src/wan_anomaly/models/mlp_torch.py` | 1–159 | PyTorch MLP classifier with BCE+pos_weight (new in CP2) |
| `src/wan_anomaly/evaluation/reports.py` | 1–32 | PR-AUC and anomaly-rate comparison plots |
| `src/wan_anomaly/explain/shap_plots.py` | 1–60 | SHAP bar chart and beeswarm plots |
| `src/wan_anomaly/utils/io.py` | 1–22 | Table I/O helpers (read/write CSV and Parquet) |
| `tests/test_labeling.py` | all | Unit test: Tukey-IQR labeling output |
| `tests/test_features.py` | all | Unit tests: rolling features, time features, ML table |
| `tests/test_split.py` | all | Unit tests: temporal ordering and coverage |
| `config/config.json` | all | Pipeline configuration |

### Lines added or modified in Checkpoint 2 (relative to Checkpoint 1)

| File | Edited Lines | Change Description |
|---|---|---|
| `scripts/build_dataset.py` | 53–64 | Added label noise injection: flip 15% of anomaly→normal labels using `numpy.random.default_rng` to simulate imperfect real-world labeling |
| `scripts/train_models.py` | 8 | Added `import torch` |
| `scripts/train_models.py` | 18 | Added `from wan_anomaly.models.mlp_torch import train_mlp_classifier` |
| `scripts/train_models.py` | 66–108 | Added MLP training block: call `train_mlp_classifier`, save PyTorch checkpoint, append MLP row to metrics/rates tables |
| `src/wan_anomaly/models/mlp_torch.py` | 1–159 | Entire file is new: `MLPNet` architecture (lines 28–42), `_compute_metrics` (lines 45–67), `train_mlp_classifier` training loop with `BCEWithLogitsLoss` + `pos_weight` (lines 70–159) |
| `scripts/make_synthetic_data.py` | 70–106 | Revised anomaly injection from hard-step to gradual ramp-up/ramp-down with per-metric targeting for more realistic overlap with normal operating range |

### Adapted from standard library / framework patterns

The following patterns were adapted from official documentation. No code was copied verbatim.

- **scikit-learn** `Pipeline`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`,
  `SVC` — constructed per the scikit-learn user guide and API reference.
- **PyTorch** `BCEWithLogitsLoss` with `pos_weight` for class imbalance — pattern from
  PyTorch official docs on weighted loss functions (used in `mlp_torch.py`, lines 121–123).
- **SHAP** `TreeExplainer` — used per the SHAP library documentation quickstart examples
  (`make_shap.py` and `src/wan_anomaly/explain/shap_plots.py`).
- **pandas** `groupby().transform()` — standard pandas pattern, applied in
  `src/wan_anomaly/processing/label.py`, lines 27–28.

### External repositories

No code was copied from external repositories.
