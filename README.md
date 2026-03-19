# Scholarship Retention Early Warning System (SREWS)

SREWS is a university decision-support project for **early scholarship risk detection**.

The system predicts whether a student is **at risk of losing scholarship in the next semester** using engineered academic and behavioral signals. It is an advisory system for early intervention, not an automated scholarship decision engine.

## Refined Problem Statement
Given semester-wise student performance signals, predict:

- `scholarship_at_risk = 1` (At Risk)
- `scholarship_at_risk = 0` (Not At Risk)

This is a **binary classification** task.

The label is rule-engineered per student-semester:

- semester CGPA < 6.0, OR
- average attendance < 65%, OR
- failed courses (`E` or `NC`) > 1

## Project Structure

```text
Scholarship-Retention-Early-Warning-System-main/
├── data/
│   └── .gitkeep
├── data_generator/
│   └── generate_academic_data.py
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── retrain_pipeline.py
│   └── train_model.py
├── dashboard/
│   ├── __init__.py
│   ├── app.py
│   ├── dashboard_charts.py
│   └── dashboard_data.py
├── Dockerfile
├── requirements.txt
├── LICENSE
└── .gitignore
```

Runtime artifacts are generated locally when scripts run and are intentionally not committed:

- `data/academic.db`
- `data/academic_records.csv`
- `models/grade_model_v*.pkl`
- `models/model_registry.json`
- `models/last_trained_run.txt`

## Tech Stack

- Python 3.14 (venv)
- SQLite
- pandas, numpy
- scikit-learn
- Streamlit + Plotly
- joblib

## Data Generation

`data_generator/generate_academic_data.py` builds synthetic multi-semester academic records with:

- academic features: midterm, attendance, assignment, quiz, previous GPA
- engineered signals: `cgpa_this_semester`, `cgpa_trend`, failed courses
- socio-financial signals: family income bracket, fee payment status, hostel resident
- support signals: counseling sessions, library usage, mental health score
- target: `scholarship_at_risk` (binary)

### Drift mode
`--drift-mode True` appends a stressed new batch (higher extracurricular load, lower study hours) to existing SQLite records. This is used to trigger retraining.

## Training Pipeline

`src/train_model.py` trains 3 models:

- Logistic Regression (`class_weight='balanced'`)
- Random Forest (`class_weight='balanced'`)
- Gradient Boosting

Model selection metric: **ROC-AUC** (binary risk, class imbalance aware).

Saved artifacts:

- `models/grade_model_vX.pkl`
- `models/model_registry.json` with:
  - version
  - trained_at
  - n_training_records
  - selected_model
  - roc_auc, precision, recall, f1
  - confusion_matrix

## Automated Retraining Trigger

`src/retrain_pipeline.py` is the viva script for drift handling.

It checks SQLite `data_runs` and compares with `models/last_trained_run.txt`:

- If new `run_id` exists: retraining is triggered.
- If no new `run_id`: pipeline reports up-to-date.

This script is the only retraining trigger mechanism.

## Dashboard

Run with Streamlit:

- Overview and KPIs
- EDA insights (risk distribution, CGPA trend, correlation heatmap, fee vs risk)
- Prominent risk predictor section
- Model history timeline and ROC-AUC trend

No raw JSON dumps are shown in UI.

## Quick Start

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Generate baseline data:

```bash
python data_generator/generate_academic_data.py \
  --n-students 1000 \
  --courses-per-student 6 \
  --n-semesters 4 \
  --seed 42 \
  --out-csv data/academic_records.csv \
  --out-db data/academic.db \
  --run-id run_initial
```

4. Train baseline model:

```bash
python -m src.train_model --db-path data/academic.db
```

5. Initialize retrain marker (first-time):

```bash
python src/retrain_pipeline.py --db-path data/academic.db
```

6. Append drift batch:

```bash
python data_generator/generate_academic_data.py \
  --n-students 120 \
  --courses-per-student 6 \
  --n-semesters 4 \
  --seed 99 \
  --out-csv data/academic_records.csv \
  --out-db data/academic.db \
  --run-id run_drift_1 \
  --drift-mode True
```

7. Trigger automated retraining:

```bash
python src/retrain_pipeline.py --db-path data/academic.db
```

8. Launch dashboard:

```bash
streamlit run dashboard/app.py
```

## Viva Demo Script (Recommended)

1. Show baseline model in registry (`v1`).
2. Run retrain script once with no new data and show: `Pipeline is up to date.`
3. Generate drift batch with new `run_id`.
4. Run retrain script and show: `New data detected ... Retraining triggered... Model v2 saved.`
5. Open dashboard model history tab and show ROC-AUC trend over versions.

## Notes

- Reproducibility is controlled via `--seed`.
- All production scripts use SQLite as primary storage.
- No notebooks are required.
