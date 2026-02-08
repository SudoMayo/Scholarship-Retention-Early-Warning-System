# ScholarGuard: Scholarship Retention Early Warning System

ScholarGuard is a hybrid decision-support system for early warning of scholarship risk. It predicts course-grade categories using traditional ML and then applies the official CGPA formula to determine projected scholarship risk.

## Problem Statement
Students at Vijaybhoomi University must maintain a CGPA >= 7.0 to retain scholarships. Currently, eligibility is checked only at year end. ScholarGuard predicts likely grade outcomes during the term and computes projected CGPA to enable early intervention.

## Repository Structure
- data_generator/ - reproducible synthetic data generator
- src/ - feature engineering, training, evaluation, CGPA engine
- dashboard/ - Streamlit UI for analytics and prediction
- models/ - serialized models and metrics registry
- data/ - local SQLite database (not committed)

## Quick Start
1) Create a virtual environment and install dependencies:
   pip install -r requirements.txt

2) Generate data (writes to data/academic.db and data/academic_records.csv):
   python data_generator/generate_academic_data.py --n-students 800 --courses-per-student 6

3) Train models and select best by macro F1:
   python -m src.train_model --db-path data/academic.db

4) Launch dashboard:
   streamlit run dashboard/app.py

## SQL Storage
All records are stored in SQLite at data/academic.db (table: academic_records). This satisfies the SQL storage requirement and supports periodic appends for retraining.

## Maintenance Timeline
- Retraining cadence: end of each semester after official grades release
- Retraining triggers: grade distribution shift > 10%, recall drop > 15%, policy changes
- Model versioning: models/grade_model_vX.pkl with registry in models/model_registry.json

## Notes
- The pipeline is reproducible via fixed random seeds.
- Scholarship decisions remain advisory; the system provides early-warning projections only.
