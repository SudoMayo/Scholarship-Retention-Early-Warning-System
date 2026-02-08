from pathlib import Path
import sys
import json
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Individual Projection",
        "Intervention Simulator",
        "Cohort Analytics",
        "Model Monitoring and Maintenance Timeline",
    ]
)

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

            st.subheader("Primary Drivers")
            avg_prev_gpa = float(editor["previous_sem_gpa"].mean())
            avg_attendance = float(editor["attendance_rate"].mean())
            credit_load = float(editor["credit_value"].sum())

            # Simple heuristics explain risk drivers for advisors.
            drivers = []
            if output["expected_cgpa"] < avg_prev_gpa - 0.3:
                drivers.append("Downward GPA trend")
            if credit_load >= 18:
                drivers.append("High credit load")
            if avg_attendance < 75:
                drivers.append("Low attendance")
            if not drivers:
                drivers.append("Stable indicators")

            st.write("- " + "\n- ".join(drivers))

            st.subheader("Suggested Intervention")
            # Suggested interventions based on weak signals.
            interventions = []
            if avg_attendance < 75:
                interventions.append("Mandatory attendance plan and weekly check-ins")
            if float(editor["midterm_score"].mean()) < 65:
                interventions.append("Midterm recovery plan with tutoring sessions")
            if float(editor["assignment_average"].mean()) < 65:
                interventions.append("Assignment support and deadline monitoring")
            if credit_load >= 18:
                interventions.append("Advise credit load review for next term")
            if not interventions:
                interventions.append("Maintain current support with monthly reviews")

            st.write("- " + "\n- ".join(interventions))
    else:
        st.info("Train a model to enable predictions.")

with tab2:
    st.subheader("Intervention Simulator")
    st.write("Adjust inputs to simulate improved performance and see updated risk.")

    if not artifact:
        st.info("Train a model to enable simulation.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            midterm_delta = st.slider("Midterm improvement", 0, 20, 5)
        with col2:
            attendance_delta = st.slider("Attendance improvement", 0, 20, 5)
        with col3:
            assignment_delta = st.slider("Assignment improvement", 0, 20, 5)

        if editor.empty:
            st.info("Add course rows in the first tab to run a simulation.")
        else:
            simulated = editor.copy()
            # Apply improvements while keeping scores within valid bounds.
            simulated["midterm_score"] = (simulated["midterm_score"] + midterm_delta).clip(0, 100)
            simulated["attendance_rate"] = (simulated["attendance_rate"] + attendance_delta).clip(0, 100)
            simulated["assignment_average"] = (simulated["assignment_average"] + assignment_delta).clip(0, 100)

            if st.button("Run Simulation"):
                output = predict_for_courses(simulated, artifact)
                st.success("Simulation complete")
                st.write("Predicted grades per course:")
                st.write(output["predicted_grades"])
                st.metric("Expected CGPA", f"{output['expected_cgpa']:.2f}")
                st.metric("Scholarship Loss Probability", f"{output['risk_probability']:.1%}")
                st.write(f"Risk Category: {output['risk_category']}")

with tab3:
    st.subheader("Cohort Analytics")
    if data.empty:
        st.info("Generate data to view cohort analytics.")
    else:
        st.markdown("Filters")
        # Filter controls affect only charts below.
        course_options = sorted(data["course_id"].unique())
        credit_options = sorted(data["credit_value"].unique())
        selected_courses = st.multiselect(
            "Courses",
            course_options,
            default=course_options,
        )
        selected_credits = st.multiselect(
            "Credit Values",
            credit_options,
            default=credit_options,
        )

        filtered = data[
            data["course_id"].isin(selected_courses)
            & data["credit_value"].isin(selected_credits)
        ]

        if filtered.empty:
            st.warning("No records match the selected filters.")
        else:
            full_cgpa_df = compute_student_cgpa(data, "grade_category")
            risk_prob = full_cgpa_df["cgpa"].apply(scholarship_prob_from_cgpa)
            full_cgpa_df["risk_category"] = risk_prob.apply(risk_category)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Students", f"{full_cgpa_df['student_id'].nunique():,}")
            col2.metric("Records", f"{len(data):,}")
            col3.metric("Avg CGPA", f"{full_cgpa_df['cgpa'].mean():.2f}")
            at_risk_pct = (full_cgpa_df["cgpa"] < 7.0).mean() * 100
            col4.metric("At Risk (<7.0)", f"{at_risk_pct:.1f}%")

            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    full_cgpa_df, x="cgpa", nbins=20, title="CGPA Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.histogram(
                    full_cgpa_df,
                    x="risk_category",
                    title="Risk Category Distribution",
                    category_orders={
                        "risk_category": ["Low Risk", "Moderate Risk", "High Risk"]
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    filtered,
                    x="grade_category",
                    color="grade_category",
                    title="Grade Category Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(
                    filtered,
                    x="course_id",
                    y="midterm_score",
                    title="Midterm Score by Course",
                )
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(
                    filtered,
                    x="attendance_rate",
                    y="midterm_score",
                    color="grade_category",
                    title="Attendance vs Midterm Score",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(
                    filtered,
                    x="course_difficulty_index",
                    y="grade_point",
                    color="grade_category",
                    title="Difficulty vs Grade Points",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("At-Risk Students (Lowest CGPA)")
            top_risk = full_cgpa_df.sort_values("cgpa").head(10)
            st.dataframe(top_risk, use_container_width=True)

with tab4:
    st.subheader("Model Monitoring")
    registry = load_registry()
    if not registry:
        st.info("No model registry available yet.")
    else:
        latest = sorted(registry, key=lambda r: r["version"], reverse=True)[0]
        st.json(latest)

    st.subheader("Maintenance Timeline")
    st.markdown(
        """
        - Retrain at the end of each semester after official grades release.
        - Retrain if grade distribution shift > 10%.
        - Retrain if scholarship risk recall drops > 15%.
        - Retrain on scholarship policy or curriculum changes.
        """
    )
