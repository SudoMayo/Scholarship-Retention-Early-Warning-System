from pathlib import Path
import json
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

from src.cgpa_engine import compute_student_cgpa, scholarship_prob_from_cgpa, risk_category
from src.predict import predict_for_courses


st.set_page_config(page_title="ScholarGuard", layout="wide")
st.title("ScholarGuard: Scholarship Retention Early Warning")

DB_PATH = Path("data/academic.db")
REGISTRY_PATH = Path("models/model_registry.json")


def load_registry():
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text())


def load_data():
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM academic_records", conn)


def load_latest_model():
    registry = load_registry()
    if not registry:
        return None, None
    latest = sorted(registry, key=lambda r: r["version"], reverse=True)[0]
    artifact = joblib.load(latest["model_path"])
    return artifact, latest


artifact, latest_meta = load_latest_model()
data = load_data()

with st.sidebar:
    st.header("Model Status")
    if latest_meta:
        st.write(f"Version: v{latest_meta['version']}")
        st.write(f"Trained at: {latest_meta['trained_at']}")
        st.write(f"Model: {latest_meta['model_name']}")
    else:
        st.warning("No trained model found. Run training first.")


tab1, tab2, tab3 = st.tabs(["Individual Projection", "Cohort Analytics", "Model Monitoring"])

with tab1:
    st.subheader("Predict Grades and CGPA")
    st.write("Enter course records for a single student and get projected CGPA.")

    default_rows = [
        {
            "course_id": "C101",
            "credit_value": 3,
            "midterm_score": 75,
            "attendance_rate": 85,
            "assignment_average": 78,
            "quiz_average": 72,
            "previous_sem_gpa": 7.4,
            "prerequisite_grade": 70,
            "course_difficulty_index": 0.6,
        }
    ]

    editor = st.data_editor(
        pd.DataFrame(default_rows),
        num_rows="dynamic",
        use_container_width=True,
    )

    if artifact and not editor.empty:
        if st.button("Run Prediction"):
            output = predict_for_courses(editor, artifact)
            st.success("Prediction complete")
            st.write("Predicted grades per course:")
            st.write(output["predicted_grades"])
            st.metric("Expected CGPA", f"{output['expected_cgpa']:.2f}")
            st.metric("Scholarship Loss Probability", f"{output['risk_probability']:.1%}")
            st.write(f"Risk Category: {output['risk_category']}")
    else:
        st.info("Train a model to enable predictions.")

with tab2:
    st.subheader("Cohort Analytics")
    if data.empty:
        st.info("Generate data to view cohort analytics.")
    else:
        cgpa_df = compute_student_cgpa(data, "grade_category")
        risk_prob = cgpa_df["cgpa"].apply(scholarship_prob_from_cgpa)
        cgpa_df["risk_category"] = risk_prob.apply(risk_category)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(cgpa_df, x="cgpa", nbins=20, title="CGPA Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(
                cgpa_df,
                x="risk_category",
                title="Risk Category Distribution",
                category_orders={
                    "risk_category": ["Low Risk", "Moderate Risk", "High Risk"]
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        at_risk_pct = (cgpa_df["cgpa"] < 7.0).mean() * 100
        st.metric("Students at Risk (<7.0)", f"{at_risk_pct:.1f}%")

with tab3:
    st.subheader("Model Monitoring")
    registry = load_registry()
    if not registry:
        st.info("No model registry available yet.")
    else:
        latest = sorted(registry, key=lambda r: r["version"], reverse=True)[0]
        st.json(latest)
