# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# dashboard_data.py — Data & model loading (cached)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from pathlib import Path
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "academic.db"
REGISTRY_PATH = PROJECT_ROOT / "models" / "model_registry.json"


@st.cache_data(ttl=3600)
def load_data():
    """Load all academic records from SQLite."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(DB_PATH)) as conn:
        return pd.read_sql_query("SELECT * FROM academic_records", conn)


@st.cache_data(ttl=3600)
def load_registry():
    """Load model registry JSON."""
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text())


@st.cache_resource
def load_model():
    """Load the latest trained model artifact."""
    registry = load_registry()
    if not registry:
        return None, None
    latest = sorted(registry, key=lambda r: r["version"], reverse=True)[0]
    model_path = PROJECT_ROOT / latest["model_path"]
    if not model_path.exists():
        return None, None
    artifact = joblib.load(str(model_path))
    return artifact, latest


def get_data_summary(data):
    """Compute summary statistics for sidebar display."""
    if data.empty:
        return {}
    summary = {
        "total_records": len(data),
        "total_students": data["student_id"].nunique(),
        "total_courses": data["course_id"].nunique(),
    }
    if "semester" in data.columns:
        summary["total_semesters"] = data["semester"].nunique()
    if "department" in data.columns:
        summary["total_departments"] = data["department"].nunique()
    if "scholarship_tier" in data.columns:
        summary["total_tiers"] = data["scholarship_tier"].nunique()
    return summary


def compute_all_cgpa(data):
    """Compute CGPA for all students, merge metadata."""
    from src.cgpa_engine import compute_student_cgpa, scholarship_prob_from_cgpa, risk_category

    cgpa_df = compute_student_cgpa(data, "grade_category")
    cgpa_df["risk_prob"] = cgpa_df["cgpa"].apply(scholarship_prob_from_cgpa)
    cgpa_df["risk_category"] = cgpa_df["risk_prob"].apply(risk_category)

    # Merge metadata
    if "department" in data.columns:
        meta = data.groupby("student_id").agg({
            "department": "first",
            "year": "first",
            "scholarship_tier": "first",
        }).reset_index()
        cgpa_df = cgpa_df.merge(meta, on="student_id", how="left")
    return cgpa_df
