"""Dashboard data and model loading helpers for SREWS."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "academic.db"
REGISTRY_PATH = PROJECT_ROOT / "models" / "model_registry.json"


@st.cache_data(ttl=300)
def load_data(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load academic records from SQLite."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM academic_records", conn)


@st.cache_data(ttl=300)
def load_registry(registry_path: Path = REGISTRY_PATH) -> List[dict]:
    """Load model registry entries if available."""
    if not registry_path.exists():
        return []
    return json.loads(registry_path.read_text(encoding="utf-8"))


@st.cache_resource
def load_latest_model(
    registry_path: Path = REGISTRY_PATH,
) -> Tuple[dict | None, dict | None]:
    """Load latest model artifact using highest model version in registry."""
    registry = load_registry(registry_path)
    if not registry:
        return None, None

    latest = max(registry, key=lambda item: int(item.get("version", 0)))
    model_path = PROJECT_ROOT / str(latest.get("model_path", "")).replace("\\", "/")
    if not model_path.exists():
        return None, latest
    artifact = joblib.load(model_path)
    return artifact, latest


def _grade_from_score(score: float) -> str:
    """Map numeric composite score to course grade category."""
    if score >= 90:
        return "A+"
    if score >= 84:
        return "A"
    if score >= 76:
        return "B+"
    if score >= 68:
        return "B"
    if score >= 60:
        return "C+"
    if score >= 52:
        return "C"
    if score >= 45:
        return "D"
    if score >= 38:
        return "E"
    return "NC"


def student_semester_view(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse course rows into student-semester view for risk dashboards."""
    if data.empty:
        return pd.DataFrame()

    agg = {
        "scholarship_at_risk": "max",
        "cgpa_this_semester": "first",
        "attendance_rate": "mean",
        "department": "first",
        "fee_payment_status": "first",
    }
    view = data.groupby(["student_id", "semester"], as_index=False).agg(agg)
    return view


def model_history_table(registry: List[dict]) -> pd.DataFrame:
    """Build a clean timeline table from model registry entries."""
    if not registry:
        return pd.DataFrame(
            columns=["version", "trained_at", "n_records", "roc_auc", "selected_model", "status"]
        )

    ordered = sorted(registry, key=lambda item: int(item.get("version", 0)))
    latest_version = int(ordered[-1].get("version", 0))

    rows = []
    for item in ordered:
        metrics = item.get("metrics", {})
        version = int(item.get("version", 0))
        rows.append(
            {
                "version": f"v{version}",
                "trained_at": str(item.get("trained_at", ""))[:19].replace("T", " "),
                "n_records": int(item.get("n_training_records", 0)),
                "roc_auc": float(metrics.get("roc_auc", 0.0)),
                "selected_model": str(item.get("selected_model", item.get("model_name", "unknown"))),
                "status": "Latest" if version == latest_version else "Previous",
            }
        )
    return pd.DataFrame(rows)


def build_prediction_row(form_values: Dict[str, object]) -> pd.DataFrame:
    """Create a model-ready single-row dataframe from predictor inputs."""
    midterm = float(form_values["midterm_score"])
    attendance = float(form_values["attendance_rate"])
    assignment = float(form_values["assignment_average"])
    quiz = float(form_values["quiz_average"])
    study = float(form_values["study_hours_per_week"])
    extra = float(form_values["extracurricular_load"])
    previous_gpa = float(form_values["previous_sem_gpa"])
    mental = float(form_values["mental_health_score"])

    composite = (
        0.35 * midterm
        + 0.20 * assignment
        + 0.15 * quiz
        + 0.20 * attendance
        + 0.10 * (previous_gpa * 10)
        + 0.8 * study
        - 0.9 * extra
        + 0.6 * mental
    )
    grade_category = _grade_from_score(composite)

    cgpa_this_semester = float(np.clip(previous_gpa + (composite - 65) / 20, 0.0, 10.0))
    cgpa_trend = float(cgpa_this_semester - previous_gpa)

    row = {
        "department": form_values["department"],
        "scholarship_tier": form_values["scholarship_tier"],
        "family_income_bracket": form_values["family_income_bracket"],
        "fee_payment_status": form_values["fee_payment_status"],
        "grade_category": grade_category,
        "year": int(form_values["year"]),
        "midterm_score": midterm,
        "attendance_rate": attendance,
        "assignment_average": assignment,
        "quiz_average": quiz,
        "study_hours_per_week": study,
        "extracurricular_load": extra,
        "previous_sem_gpa": previous_gpa,
        "counselling_sessions_attended": int(form_values["counselling_sessions_attended"]),
        "library_usage_hours_per_week": float(form_values["library_usage_hours_per_week"]),
        "hostel_resident": int(bool(form_values["hostel_resident"])),
        "mental_health_score": mental,
        "cgpa_this_semester": cgpa_this_semester,
        "cgpa_trend": cgpa_trend,
    }

    return pd.DataFrame([row])


def feature_importance_table(artifact: dict | None) -> pd.DataFrame:
    """Return model feature importance in display-friendly form."""
    if not artifact:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = artifact.get("feature_importance", {})
    if not importance:
        model = artifact.get("model")
        if model is not None:
            preprocessor = model.named_steps.get("preprocess")
            estimator = model.named_steps.get("model")
            if preprocessor is not None and estimator is not None:
                names = preprocessor.get_feature_names_out().tolist()
                if hasattr(estimator, "feature_importances_"):
                    vals = estimator.feature_importances_
                elif hasattr(estimator, "coef_"):
                    vals = np.abs(estimator.coef_[0])
                else:
                    vals = []
                importance = {k: float(v) for k, v in zip(names, vals)}

    if not importance:
        return pd.DataFrame(columns=["feature", "importance"])

    table = pd.DataFrame(
        [{"feature": key, "importance": value} for key, value in importance.items()]
    )
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    return table
