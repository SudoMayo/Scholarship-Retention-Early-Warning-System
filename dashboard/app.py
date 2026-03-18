from pathlib import Path
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cgpa_engine import (
    compute_student_cgpa,
    compute_semester_cgpa,
    compute_cgpa_trajectory,
    scholarship_prob_from_cgpa,
    risk_category,
    GRADE_POINTS,
)
from src.predict import predict_for_courses

# ──────────────────────────────────────────────────
# Page Config & Custom CSS
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ScholarGuard — Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium dark-themed CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
    }

    /* Gradient header */
    .hero-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #8e8ea0;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        color: #8e8ea0;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* Alert cards */
    .alert-critical {
        background: linear-gradient(135deg, #2d1f2f 0%, #3d1f1f 100%);
        border-left: 4px solid #ff4757;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background: linear-gradient(135deg, #2d2a1f 0%, #3d351f 100%);
        border-left: 4px solid #ffa502;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-safe {
        background: linear-gradient(135deg, #1f2d22 0%, #1f3d24 100%);
        border-left: 4px solid #2ed573;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Risk badges */
    .badge-high { color: #ff4757; font-weight: 600; }
    .badge-moderate { color: #ffa502; font-weight: 600; }
    .badge-low { color: #2ed573; font-weight: 600; }

    /* Confidence indicator */
    .confidence-high { color: #2ed573; }
    .confidence-mid { color: #ffa502; }
    .confidence-low { color: #ff4757; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎓 ScholarGuard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Scholarship Retention Early Warning System — Vijaybhoomi University</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────
DB_PATH = Path("data/academic.db")
REGISTRY_PATH = Path("models/model_registry.json")


@st.cache_data
def load_registry():
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text())


@st.cache_data
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

# ──────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=60)
    st.header("ScholarGuard")
    st.divider()

    st.subheader("📦 Model Status")
    if latest_meta:
        st.success(f"v{latest_meta['version']} — {latest_meta['model_name']}")
        st.caption(f"Trained: {latest_meta['trained_at'][:10]}")
        acc = latest_meta["metrics"]["accuracy"]
        f1 = latest_meta["metrics"]["macro_f1"]
        st.progress(acc, text=f"Accuracy: {acc:.1%}")
        st.progress(f1, text=f"Macro F1: {f1:.1%}")
    else:
        st.warning("No trained model found.")

    st.divider()
    st.subheader("📊 Data Summary")
    if not data.empty:
        st.metric("Total Records", f"{len(data):,}")
        st.metric("Students", f"{data['student_id'].nunique():,}")
        if "semester" in data.columns:
            st.metric("Semesters", f"{data['semester'].nunique()}")
        if "department" in data.columns:
            st.metric("Departments", f"{data['department'].nunique()}")


# ──────────────────────────────────────────────────
# Plotly theme defaults
# ──────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e0e0e0"),
    margin=dict(l=40, r=40, t=50, b=40),
)
COLOR_PALETTE = ["#667eea", "#764ba2", "#f093fb", "#ffecd2", "#a8edea", "#fed6e3", "#d4fc79"]
RISK_COLORS = {"High Risk": "#ff4757", "Moderate Risk": "#ffa502", "Low Risk": "#2ed573"}

# ──────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "🎯 Individual Projection",
        "🔬 Intervention Simulator",
        "📊 Cohort Analytics",
        "🚨 Early Warning Center",
        "👤 Student Deep Dive",
        "📈 Semester Trends",
        "⚙️ Model Monitoring",
    ]
)


# ══════════════════════════════════════════════════
# TAB 1: Individual Projection (Enhanced)
# ══════════════════════════════════════════════════
with tab1:
    st.subheader("🎯 Predict Grades & Projected CGPA")
    st.caption("Enter course records for a student and get projected CGPA, risk, and personalized insights.")

    default_rows = [
        {
            "course_id": "C101",
            "credit_value": 3,
            "midterm_score": 75,
            "attendance_rate": 85,
            "assignment_average": 78,
            "quiz_average": 72,
            "study_hours_per_week": 15.0,
            "extracurricular_load": 3.0,
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
        if st.button("🔮 Run Prediction", type="primary", key="predict_btn"):
            output = predict_for_courses(editor, artifact)

            # ── KPI Row ──
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Expected CGPA", f"{output['expected_cgpa']:.2f}")
            with col2:
                risk_color = RISK_COLORS.get(output["risk_category"], "#fff")
                st.metric("Risk Category", output["risk_category"])
            with col3:
                st.metric("Loss Probability", f"{output['risk_probability']:.1%}")
            with col4:
                avg_conf = np.mean(output["confidences"])
                st.metric("Avg Confidence", f"{avg_conf:.1%}")

            st.divider()

            # ── Grades table with confidence ──
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.write("**Predicted Grades per Course:**")
                grade_df = editor[["course_id", "credit_value"]].copy()
                grade_df["Predicted Grade"] = output["predicted_grades"]
                grade_df["Grade Points"] = output["predicted_grade_points"]
                grade_df["Confidence"] = [f"{c:.1%}" for c in output["confidences"]]
                st.dataframe(grade_df, use_container_width=True, hide_index=True)

            with col_right:
                # ── Radar chart ──
                st.write("**Student Strengths Profile:**")
                categories = ["Midterm", "Assignment", "Quiz", "Attendance", "Study Hours"]
                values = [
                    float(editor["midterm_score"].mean()) / 100,
                    float(editor["assignment_average"].mean()) / 100,
                    float(editor["quiz_average"].mean()) / 100,
                    float(editor["attendance_rate"].mean()) / 100,
                    float(editor["study_hours_per_week"].mean()) / 35,
                ]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor="rgba(102, 126, 234, 0.3)",
                    line=dict(color="#667eea", width=2),
                    name="Student",
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                    ),
                    showlegend=False,
                    height=300,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ── Drivers & Interventions ──
            col_d, col_i = st.columns(2)
            with col_d:
                st.subheader("🔍 Primary Drivers")
                avg_prev_gpa = float(editor["previous_sem_gpa"].mean())
                avg_attendance = float(editor["attendance_rate"].mean())
                credit_load = float(editor["credit_value"].sum())
                avg_study = float(editor["study_hours_per_week"].mean())

                drivers = []
                if output["expected_cgpa"] < avg_prev_gpa - 0.3:
                    drivers.append("📉 Downward GPA trend detected")
                if credit_load >= 18:
                    drivers.append("📚 High credit load")
                if avg_attendance < 75:
                    drivers.append("🚫 Low attendance rate")
                if avg_study < 10:
                    drivers.append("⏰ Insufficient study hours")
                if float(editor["extracurricular_load"].mean()) > 6:
                    drivers.append("🎭 High extracurricular commitment")
                if not drivers:
                    drivers.append("✅ Stable indicators — no major concerns")

                for d in drivers:
                    st.write(f"- {d}")

            with col_i:
                st.subheader("💡 Suggested Interventions")
                interventions = []
                if avg_attendance < 75:
                    interventions.append("📋 Mandatory attendance plan with weekly advisor check-ins")
                if float(editor["midterm_score"].mean()) < 65:
                    interventions.append("📖 Midterm recovery plan with peer tutoring sessions")
                if float(editor["assignment_average"].mean()) < 65:
                    interventions.append("📝 Assignment support and deadline monitoring system")
                if credit_load >= 18:
                    interventions.append("⚖️ Review credit load — consider dropping one elective")
                if avg_study < 10:
                    interventions.append("📅 Structured study schedule with 15+ hrs/week target")
                if not interventions:
                    interventions.append("🔄 Continue current support with monthly progress reviews")

                for iv in interventions:
                    st.write(f"- {iv}")
    else:
        st.info("Train a model to enable predictions. Run: `python -m src.train_model`")


# ══════════════════════════════════════════════════
# TAB 2: Intervention Simulator (Major Upgrade)
# ══════════════════════════════════════════════════
with tab2:
    st.subheader("🔬 Multi-Scenario Intervention Simulator")
    st.caption("Compare up to 3 improvement scenarios side-by-side to find the best intervention strategy.")

    if not artifact:
        st.info("Train a model to enable simulation.")
    else:
        st.write("**Base Student Input:**")
        sim_default = [
            {
                "course_id": "C101",
                "credit_value": 3,
                "midterm_score": 62,
                "attendance_rate": 70,
                "assignment_average": 58,
                "quiz_average": 55,
                "study_hours_per_week": 10.0,
                "extracurricular_load": 5.0,
                "previous_sem_gpa": 6.5,
                "prerequisite_grade": 60,
                "course_difficulty_index": 0.6,
            },
            {
                "course_id": "C103",
                "credit_value": 4,
                "midterm_score": 55,
                "attendance_rate": 65,
                "assignment_average": 60,
                "quiz_average": 50,
                "study_hours_per_week": 10.0,
                "extracurricular_load": 5.0,
                "previous_sem_gpa": 6.5,
                "prerequisite_grade": 55,
                "course_difficulty_index": 0.7,
            },
        ]

        sim_editor = st.data_editor(
            pd.DataFrame(sim_default), num_rows="dynamic", use_container_width=True, key="sim_editor"
        )

        if not sim_editor.empty:
            st.divider()

            # ── Three scenario columns ──
            scenarios = {}
            cols = st.columns(3)

            scenario_names = ["Scenario A: Moderate", "Scenario B: Intensive", "Scenario C: Custom"]
            default_deltas = [
                {"midterm": 5, "attendance": 5, "assignment": 5, "quiz": 3, "study_hours": 2},
                {"midterm": 15, "attendance": 15, "assignment": 15, "quiz": 10, "study_hours": 5},
                {"midterm": 10, "attendance": 10, "assignment": 10, "quiz": 5, "study_hours": 3},
            ]

            for i, (col, name, defaults) in enumerate(zip(cols, scenario_names, default_deltas)):
                with col:
                    st.markdown(f"**{name}**")
                    midterm_d = st.slider("Midterm Δ", 0, 25, defaults["midterm"], key=f"m_{i}")
                    attend_d = st.slider("Attendance Δ", 0, 25, defaults["attendance"], key=f"a_{i}")
                    assign_d = st.slider("Assignment Δ", 0, 25, defaults["assignment"], key=f"as_{i}")
                    quiz_d = st.slider("Quiz Δ", 0, 25, defaults["quiz"], key=f"q_{i}")
                    study_d = st.slider("Study Hours Δ", 0, 10, defaults["study_hours"], key=f"s_{i}")
                    scenarios[name] = {
                        "midterm": midterm_d, "attendance": attend_d,
                        "assignment": assign_d, "quiz": quiz_d, "study_hours": study_d,
                    }

            if st.button("🚀 Run All Scenarios", type="primary", key="sim_run"):
                # Baseline
                baseline_output = predict_for_courses(sim_editor, artifact)

                results = {"Baseline": baseline_output}
                for sname, deltas in scenarios.items():
                    sim = sim_editor.copy()
                    sim["midterm_score"] = (sim["midterm_score"] + deltas["midterm"]).clip(0, 100)
                    sim["attendance_rate"] = (sim["attendance_rate"] + deltas["attendance"]).clip(0, 100)
                    sim["assignment_average"] = (sim["assignment_average"] + deltas["assignment"]).clip(0, 100)
                    sim["quiz_average"] = (sim["quiz_average"] + deltas["quiz"]).clip(0, 100)
                    sim["study_hours_per_week"] = (sim["study_hours_per_week"] + deltas["study_hours"]).clip(3, 35)
                    results[sname] = predict_for_courses(sim, artifact)

                st.divider()
                st.subheader("📊 Scenario Comparison")

                # Comparison KPI row
                comp_cols = st.columns(4)
                labels = list(results.keys())
                for j, (label, res) in enumerate(results.items()):
                    with comp_cols[j]:
                        delta = res["expected_cgpa"] - baseline_output["expected_cgpa"] if label != "Baseline" else None
                        st.metric(
                            label,
                            f"{res['expected_cgpa']:.2f}",
                            delta=f"{delta:+.2f}" if delta is not None else None,
                        )
                        risk_cat = res["risk_category"]
                        st.caption(f"{risk_cat} ({res['risk_probability']:.1%})")

                # Grade change table
                st.write("**Per-Course Grade Changes:**")
                change_data = {"Course": sim_editor["course_id"].tolist()}
                change_data["Baseline"] = baseline_output["predicted_grades"]
                for sname in scenarios:
                    change_data[sname] = results[sname]["predicted_grades"]
                st.dataframe(
                    pd.DataFrame(change_data),
                    use_container_width=True,
                    hide_index=True,
                )

                # CGPA bar chart
                fig = go.Figure()
                for idx, (label, res) in enumerate(results.items()):
                    color = "#667eea" if label == "Baseline" else COLOR_PALETTE[idx]
                    fig.add_trace(go.Bar(
                        x=[label], y=[res["expected_cgpa"]],
                        name=label,
                        marker_color=color,
                        text=[f"{res['expected_cgpa']:.2f}"],
                        textposition="outside",
                    ))
                fig.add_hline(y=7.0, line_dash="dash", line_color="#ff4757",
                              annotation_text="Scholarship Threshold (7.0)")
                fig.update_layout(
                    title="CGPA Comparison Across Scenarios",
                    yaxis_title="Expected CGPA",
                    yaxis_range=[0, 10],
                    showlegend=False,
                    height=400,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════
# TAB 3: Cohort Analytics (Enhanced)
# ══════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Cohort Analytics Dashboard")
    if data.empty:
        st.info("Generate data first: `python data_generator/generate_academic_data.py`")
    else:
        # ── Filters ──
        with st.expander("🔎 Filters", expanded=True):
            f_cols = st.columns(4)
            with f_cols[0]:
                dept_options = sorted(data["department"].unique()) if "department" in data.columns else []
                selected_depts = st.multiselect("Department", dept_options, default=dept_options)
            with f_cols[1]:
                sem_options = sorted(data["semester"].unique()) if "semester" in data.columns else []
                selected_sems = st.multiselect("Semester", sem_options, default=sem_options)
            with f_cols[2]:
                tier_options = sorted(data["scholarship_tier"].unique()) if "scholarship_tier" in data.columns else []
                selected_tiers = st.multiselect("Scholarship Tier", tier_options, default=tier_options)
            with f_cols[3]:
                course_options = sorted(data["course_id"].unique())
                selected_courses = st.multiselect("Courses", course_options, default=course_options)

        filtered = data.copy()
        if "department" in data.columns:
            filtered = filtered[filtered["department"].isin(selected_depts)]
        if "semester" in data.columns:
            filtered = filtered[filtered["semester"].isin(selected_sems)]
        if "scholarship_tier" in data.columns:
            filtered = filtered[filtered["scholarship_tier"].isin(selected_tiers)]
        filtered = filtered[filtered["course_id"].isin(selected_courses)]

        if filtered.empty:
            st.warning("No records match the selected filters.")
        else:
            # ── KPIs ──
            full_cgpa_df = compute_student_cgpa(filtered, "grade_category")
            risk_prob = full_cgpa_df["cgpa"].apply(scholarship_prob_from_cgpa)
            full_cgpa_df["risk_category"] = risk_prob.apply(risk_category)

            # Merge department info for KPIs
            if "department" in filtered.columns:
                student_dept = filtered.groupby("student_id")["department"].first().reset_index()
                full_cgpa_df = full_cgpa_df.merge(student_dept, on="student_id", how="left")
            if "scholarship_tier" in filtered.columns:
                student_tier = filtered.groupby("student_id")["scholarship_tier"].first().reset_index()
                full_cgpa_df = full_cgpa_df.merge(student_tier, on="student_id", how="left")

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Students", f"{full_cgpa_df['student_id'].nunique():,}")
            k2.metric("Records", f"{len(filtered):,}")
            k3.metric("Avg CGPA", f"{full_cgpa_df['cgpa'].mean():.2f}")
            at_risk_pct = (full_cgpa_df["cgpa"] < 7.0).mean() * 100
            k4.metric("At Risk (<7.0)", f"{at_risk_pct:.1f}%")
            high_risk_pct = (full_cgpa_df["risk_category"] == "High Risk").mean() * 100
            k5.metric("High Risk", f"{high_risk_pct:.1f}%")

            st.divider()

            # ── Row 1: CGPA distribution + Risk distribution ──
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(
                    full_cgpa_df, x="cgpa", nbins=25,
                    title="CGPA Distribution",
                    color_discrete_sequence=["#667eea"],
                )
                fig.add_vline(x=7.0, line_dash="dash", line_color="#ff4757",
                              annotation_text="Threshold")
                fig.update_layout(height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                risk_counts = full_cgpa_df["risk_category"].value_counts().reset_index()
                risk_counts.columns = ["risk_category", "count"]
                fig = px.pie(
                    risk_counts, values="count", names="risk_category",
                    title="Risk Category Distribution",
                    color="risk_category",
                    color_discrete_map=RISK_COLORS,
                    hole=0.4,
                )
                fig.update_layout(height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # ── Row 2: Sunburst + Correlation Heatmap ──
            c1, c2 = st.columns(2)
            with c1:
                if "department" in full_cgpa_df.columns:
                    fig = px.sunburst(
                        full_cgpa_df,
                        path=["department", "risk_category"],
                        title="Department → Risk Breakdown",
                        color="risk_category",
                        color_discrete_map=RISK_COLORS,
                    )
                    fig.update_layout(height=400, **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                numeric_cols = [
                    "midterm_score", "attendance_rate", "assignment_average",
                    "quiz_average", "previous_sem_gpa", "course_difficulty_index",
                ]
                if "study_hours_per_week" in filtered.columns:
                    numeric_cols.append("study_hours_per_week")
                corr = filtered[numeric_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="Feature Correlation Heatmap",
                    aspect="auto",
                )
                fig.update_layout(height=400, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # ── Row 3: Scholarship tier breakdown + Grade distribution ──
            c1, c2 = st.columns(2)
            with c1:
                if "scholarship_tier" in full_cgpa_df.columns:
                    fig = px.box(
                        full_cgpa_df, x="scholarship_tier", y="cgpa",
                        color="scholarship_tier",
                        title="CGPA by Scholarship Tier",
                        color_discrete_sequence=COLOR_PALETTE,
                    )
                    fig.add_hline(y=7.0, line_dash="dash", line_color="#ff4757")
                    fig.update_layout(height=350, **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.histogram(
                    filtered, x="grade_category", color="grade_category",
                    title="Grade Category Distribution",
                    category_orders={"grade_category": list(GRADE_POINTS.keys())},
                    color_discrete_sequence=COLOR_PALETTE,
                )
                fig.update_layout(height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # ── Row 4: Scatter plots ──
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(
                    filtered, x="attendance_rate", y="midterm_score",
                    color="grade_category", title="Attendance vs Midterm",
                    opacity=0.6,
                    color_discrete_sequence=COLOR_PALETTE,
                )
                fig.update_layout(height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.scatter(
                    filtered, x="study_hours_per_week" if "study_hours_per_week" in filtered.columns else "course_difficulty_index",
                    y="grade_point",
                    color="grade_category",
                    title="Study Hours vs Grade Points" if "study_hours_per_week" in filtered.columns else "Difficulty vs Grade Points",
                    opacity=0.6,
                    color_discrete_sequence=COLOR_PALETTE,
                )
                fig.update_layout(height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # ── At-risk table ──
            st.subheader("🚩 At-Risk Students (Lowest CGPA)")
            top_risk = full_cgpa_df.sort_values("cgpa").head(15)
            st.dataframe(top_risk, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════
# TAB 4: Early Warning Command Center (NEW)
# ══════════════════════════════════════════════════
with tab4:
    st.subheader("🚨 Early Warning Command Center")
    st.caption("Real-time alert dashboard for academic advisors — identify at-risk students instantly.")

    if data.empty:
        st.info("Generate data to enable the early warning system.")
    else:
        # Compute CGPA for all students
        all_cgpa = compute_student_cgpa(data, "grade_category")
        all_cgpa["risk_prob"] = all_cgpa["cgpa"].apply(scholarship_prob_from_cgpa)
        all_cgpa["risk_category"] = all_cgpa["risk_prob"].apply(risk_category)

        # Merge metadata
        if "department" in data.columns:
            meta = data.groupby("student_id").agg({
                "department": "first",
                "year": "first",
                "scholarship_tier": "first",
            }).reset_index()
            all_cgpa = all_cgpa.merge(meta, on="student_id", how="left")

        critical = all_cgpa[all_cgpa["cgpa"] < 6.0]
        warning = all_cgpa[(all_cgpa["cgpa"] >= 6.0) & (all_cgpa["cgpa"] < 7.0)]
        safe = all_cgpa[all_cgpa["cgpa"] >= 7.0]

        # ── Alert KPI Cards ──
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""<div class="alert-critical">
                <h2 style="color:#ff4757;margin:0;">🔴 {len(critical)}</h2>
                <p style="color:#ff9f9f;margin:0;">Critical (CGPA &lt; 6.0)</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="alert-warning">
                <h2 style="color:#ffa502;margin:0;">🟡 {len(warning)}</h2>
                <p style="color:#ffd166;margin:0;">Warning (6.0 – 7.0)</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""<div class="alert-safe">
                <h2 style="color:#2ed573;margin:0;">🟢 {len(safe)}</h2>
                <p style="color:#7bed9f;margin:0;">Safe (CGPA ≥ 7.0)</p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Department Risk Breakdown ──
        if "department" in all_cgpa.columns:
            c1, c2 = st.columns(2)
            with c1:
                dept_risk = all_cgpa.groupby("department").agg(
                    avg_cgpa=("cgpa", "mean"),
                    at_risk_count=("cgpa", lambda x: (x < 7.0).sum()),
                    total=("cgpa", "count"),
                ).reset_index()
                dept_risk["at_risk_pct"] = (dept_risk["at_risk_count"] / dept_risk["total"] * 100).round(1)
                dept_risk = dept_risk.sort_values("at_risk_pct", ascending=False)

                fig = px.bar(
                    dept_risk, x="department", y="at_risk_pct",
                    color="at_risk_pct",
                    color_continuous_scale=["#2ed573", "#ffa502", "#ff4757"],
                    title="At-Risk Percentage by Department",
                    text="at_risk_pct",
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(height=350, coloraxis_showscale=False, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                if "scholarship_tier" in all_cgpa.columns:
                    tier_risk = all_cgpa.groupby("scholarship_tier").agg(
                        avg_cgpa=("cgpa", "mean"),
                        at_risk_count=("cgpa", lambda x: (x < 7.0).sum()),
                        total=("cgpa", "count"),
                    ).reset_index()
                    tier_risk["at_risk_pct"] = (tier_risk["at_risk_count"] / tier_risk["total"] * 100).round(1)

                    fig = px.bar(
                        tier_risk, x="scholarship_tier", y="at_risk_pct",
                        color="at_risk_pct",
                        color_continuous_scale=["#2ed573", "#ffa502", "#ff4757"],
                        title="At-Risk Percentage by Scholarship Tier",
                        text="at_risk_pct",
                    )
                    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig.update_layout(height=350, coloraxis_showscale=False, **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

        # ── Critical Students Table ──
        st.subheader("🔴 Critical Alert List")
        alert_filter = st.radio(
            "Show:", ["🔴 Critical Only", "🟡 Warning Only", "All At-Risk"],
            horizontal=True,
            key="alert_filter",
        )

        if alert_filter == "🔴 Critical Only":
            display_df = critical
        elif alert_filter == "🟡 Warning Only":
            display_df = warning
        else:
            display_df = all_cgpa[all_cgpa["cgpa"] < 7.0]

        display_df = display_df.sort_values("cgpa")

        if display_df.empty:
            st.success("No students in this category! 🎉")
        else:
            st.dataframe(
                display_df.style.format({"cgpa": "{:.2f}", "risk_prob": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download At-Risk List (CSV)",
                csv,
                "at_risk_students.csv",
                "text/csv",
                key="download_risk",
            )


# ══════════════════════════════════════════════════
# TAB 5: Student Deep Dive (NEW)
# ══════════════════════════════════════════════════
with tab5:
    st.subheader("👤 Student Deep Dive")
    st.caption("Select a student to view their complete academic journey across semesters.")

    if data.empty:
        st.info("Generate data to explore student profiles.")
    else:
        student_ids = sorted(data["student_id"].unique())
        selected_student = st.selectbox("Select Student ID", student_ids, key="student_select")

        student_data = data[data["student_id"] == selected_student]

        if student_data.empty:
            st.warning("No data found for this student.")
        else:
            # ── Student metadata header ──
            dept = student_data["department"].iloc[0] if "department" in student_data.columns else "N/A"
            year = student_data["year"].iloc[0] if "year" in student_data.columns else "N/A"
            tier = student_data["scholarship_tier"].iloc[0] if "scholarship_tier" in student_data.columns else "N/A"

            # Overall CGPA
            overall_cgpa_df = compute_student_cgpa(student_data, "grade_category")
            overall_cgpa = overall_cgpa_df["cgpa"].iloc[0] if not overall_cgpa_df.empty else 0
            risk_prob = scholarship_prob_from_cgpa(overall_cgpa)
            risk_cat = risk_category(risk_prob)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Student", selected_student)
            m2.metric("Department", dept)
            m3.metric("Year", year)
            m4.metric("Cumulative CGPA", f"{overall_cgpa:.2f}")
            m5.metric("Risk", risk_cat)

            st.divider()

            # ── CGPA Trajectory ──
            if "semester" in student_data.columns:
                trajectory = compute_cgpa_trajectory(student_data, "grade_category")

                c1, c2 = st.columns(2)
                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trajectory["semester"],
                        y=trajectory["cumulative_cgpa"],
                        mode="lines+markers",
                        line=dict(color="#667eea", width=3),
                        marker=dict(size=10, color="#667eea"),
                        name="Cumulative CGPA",
                        fill="tozeroy",
                        fillcolor="rgba(102, 126, 234, 0.1)",
                    ))
                    fig.add_hline(y=7.0, line_dash="dash", line_color="#ff4757",
                                  annotation_text="Scholarship Threshold")
                    fig.update_layout(
                        title="CGPA Trajectory Over Semesters",
                        xaxis_title="Semester",
                        yaxis_title="Cumulative CGPA",
                        yaxis_range=[0, 10],
                        height=350,
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    # Per-semester GPA
                    sem_gpa = compute_semester_cgpa(student_data, "grade_category")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=sem_gpa["semester"],
                        y=sem_gpa["semester_gpa"],
                        marker_color=["#ff4757" if g < 7 else "#2ed573" for g in sem_gpa["semester_gpa"]],
                        text=[f"{g:.2f}" for g in sem_gpa["semester_gpa"]],
                        textposition="outside",
                    ))
                    fig.add_hline(y=7.0, line_dash="dash", line_color="#ff4757")
                    fig.update_layout(
                        title="Semester-wise GPA",
                        xaxis_title="Semester",
                        yaxis_title="GPA",
                        yaxis_range=[0, 10],
                        height=350,
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ── Performance Radar by Semester ──
            if "semester" in student_data.columns:
                st.subheader("🕸️ Performance Radar by Semester")
                semesters = sorted(student_data["semester"].unique())
                fig = go.Figure()
                categories = ["Midterm", "Assignment", "Quiz", "Attendance", "Study Hrs"]

                for sem in semesters:
                    sem_data = student_data[student_data["semester"] == sem]
                    values = [
                        sem_data["midterm_score"].mean() / 100,
                        sem_data["assignment_average"].mean() / 100,
                        sem_data["quiz_average"].mean() / 100,
                        sem_data["attendance_rate"].mean() / 100,
                        sem_data["study_hours_per_week"].mean() / 35 if "study_hours_per_week" in sem_data.columns else 0.5,
                    ]
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        name=f"Sem {sem}",
                        fill="toself",
                        opacity=0.6,
                    ))

                fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                    ),
                    height=400,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Full record table ──
            st.subheader("📋 Complete Course Records")
            display_cols = [
                "semester", "course_id", "credit_value", "midterm_score",
                "attendance_rate", "assignment_average", "quiz_average",
                "grade_category", "grade_point",
            ]
            display_cols = [c for c in display_cols if c in student_data.columns]
            st.dataframe(
                student_data[display_cols].sort_values(["semester", "course_id"]),
                use_container_width=True,
                hide_index=True,
            )

            # ── Personalized Recommendations ──
            st.subheader("💡 Personalized Recommendations")
            latest_sem = student_data["semester"].max() if "semester" in student_data.columns else 1
            latest_data = student_data[student_data["semester"] == latest_sem] if "semester" in student_data.columns else student_data

            avg_mid = latest_data["midterm_score"].mean()
            avg_att = latest_data["attendance_rate"].mean()
            avg_assign = latest_data["assignment_average"].mean()
            avg_study = latest_data["study_hours_per_week"].mean() if "study_hours_per_week" in latest_data.columns else 15

            recs = []
            if overall_cgpa < 6.0:
                recs.append("🔴 **Immediate academic probation review** — schedule meeting with department head")
            if avg_att < 70:
                recs.append("📋 **Attendance intervention required** — current rate below 70%")
            if avg_mid < 55:
                recs.append("📖 **Remedial tutoring** — midterm scores critically low")
            if avg_assign < 60:
                recs.append("📝 **Assignment support** — connect with peer mentors")
            if avg_study < 10:
                recs.append("⏰ **Study skills workshop** — hours significantly below target")
            if overall_cgpa >= 7.0 and overall_cgpa < 7.5:
                recs.append("⚠️ **Monitor closely** — CGPA near scholarship threshold")
            if overall_cgpa >= 8.0:
                recs.append("🌟 **Strong performer** — consider for academic excellence program")
            if not recs:
                recs.append("✅ **On track** — maintain current performance with regular check-ins")

            for r in recs:
                st.write(f"- {r}")


# ══════════════════════════════════════════════════
# TAB 6: Semester Trend Analysis (NEW)
# ══════════════════════════════════════════════════
with tab6:
    st.subheader("📈 Semester Trend Analysis")
    st.caption("Track cohort-wide performance trends over semesters.")

    if data.empty or "semester" not in data.columns:
        st.info("Generate multi-semester data to view trends.")
    else:
        # ── Average CGPA trend per semester ──
        sem_cgpa_all = compute_semester_cgpa(data, "grade_category")
        avg_by_sem = sem_cgpa_all.groupby("semester")["semester_gpa"].mean().reset_index()
        avg_by_sem.columns = ["Semester", "Avg GPA"]

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=avg_by_sem["Semester"],
                y=avg_by_sem["Avg GPA"],
                mode="lines+markers",
                line=dict(color="#667eea", width=3),
                marker=dict(size=10),
                fill="tozeroy",
                fillcolor="rgba(102, 126, 234, 0.1)",
            ))
            fig.add_hline(y=7.0, line_dash="dash", line_color="#ff4757",
                          annotation_text="Threshold")
            fig.update_layout(
                title="Average GPA Trend by Semester",
                xaxis_title="Semester",
                yaxis_title="Average GPA",
                yaxis_range=[4, 10],
                height=350,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # At-risk % per semester
            cumulative_risk = []
            for sem in sorted(data["semester"].unique()):
                sem_data = data[data["semester"] <= sem]
                cgpa_df = compute_student_cgpa(sem_data, "grade_category")
                risk_pct = (cgpa_df["cgpa"] < 7.0).mean() * 100
                cumulative_risk.append({"Semester": sem, "At-Risk %": risk_pct})

            risk_trend = pd.DataFrame(cumulative_risk)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=risk_trend["Semester"],
                y=risk_trend["At-Risk %"],
                mode="lines+markers",
                line=dict(color="#ff4757", width=3),
                marker=dict(size=10),
                fill="tozeroy",
                fillcolor="rgba(255, 71, 87, 0.1)",
            ))
            fig.update_layout(
                title="Cumulative At-Risk % Over Semesters",
                xaxis_title="Semester",
                yaxis_title="At Risk %",
                height=350,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Department-wise GPA trend ──
        if "department" in data.columns:
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                dept_sem = data.groupby(["semester", "department"]).agg(
                    avg_midterm=("midterm_score", "mean"),
                ).reset_index()

                fig = px.line(
                    dept_sem, x="semester", y="avg_midterm",
                    color="department",
                    markers=True,
                    title="Average Midterm by Department Over Semesters",
                    color_discrete_sequence=COLOR_PALETTE,
                )
                fig.update_layout(height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                # Grade distribution shift (stacked area)
                grade_by_sem = data.groupby(["semester", "grade_category"]).size().reset_index(name="count")
                sem_totals = grade_by_sem.groupby("semester")["count"].transform("sum")
                grade_by_sem["percentage"] = grade_by_sem["count"] / sem_totals * 100

                fig = px.area(
                    grade_by_sem, x="semester", y="percentage",
                    color="grade_category",
                    title="Grade Distribution Shift Over Semesters",
                    category_orders={"grade_category": list(GRADE_POINTS.keys())},
                    color_discrete_sequence=COLOR_PALETTE,
                )
                fig.update_layout(height=350, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        # ── Department Leaderboard ──
        if "department" in data.columns:
            st.divider()
            st.subheader("🏆 Department Leaderboard")

            dept_cgpa = data.merge(
                compute_student_cgpa(data, "grade_category"), on="student_id"
            ).drop_duplicates("student_id")

            leaderboard = dept_cgpa.groupby("department").agg(
                avg_cgpa=("cgpa", "mean"),
                median_cgpa=("cgpa", "median"),
                top_performers=("cgpa", lambda x: (x >= 8.0).sum()),
                at_risk=("cgpa", lambda x: (x < 7.0).sum()),
                total_students=("cgpa", "count"),
            ).reset_index()
            leaderboard["at_risk_pct"] = (leaderboard["at_risk"] / leaderboard["total_students"] * 100).round(1)
            leaderboard = leaderboard.sort_values("avg_cgpa", ascending=False)

            st.dataframe(
                leaderboard.style.format({
                    "avg_cgpa": "{:.2f}",
                    "median_cgpa": "{:.2f}",
                    "at_risk_pct": "{:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════
# TAB 7: Model Monitoring (Enhanced)
# ══════════════════════════════════════════════════
with tab7:
    st.subheader("⚙️ Model Monitoring & Maintenance")

    registry = load_registry()
    if not registry:
        st.info("No model registry found. Train a model first.")
    else:
        # ── Latest model metrics cards ──
        latest = sorted(registry, key=lambda r: r["version"], reverse=True)[0]

        st.write("**Latest Model:**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Version", f"v{latest['version']}")
        m2.metric("Model", latest["model_name"])
        m3.metric("Accuracy", f"{latest['metrics']['accuracy']:.1%}")
        m4.metric("Macro F1", f"{latest['metrics']['macro_f1']:.1%}")

        st.divider()

        # ── Detailed metrics ──
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Classification Metrics:**")
            metrics_df = pd.DataFrame([
                {"Metric": "Accuracy", "Value": latest["metrics"]["accuracy"]},
                {"Metric": "Macro F1", "Value": latest["metrics"]["macro_f1"]},
            ])
            fig = px.bar(
                metrics_df, x="Metric", y="Value",
                text="Value",
                color_discrete_sequence=["#667eea"],
                title="Classification Performance",
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(yaxis_range=[0, 1], height=300, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.write("**Scholarship Risk Metrics:**")
            risk_m = latest.get("risk_metrics", {})
            risk_df = pd.DataFrame([
                {"Metric": "Recall", "Value": risk_m.get("recall", 0)},
                {"Metric": "F1", "Value": risk_m.get("f1", 0)},
                {"Metric": "ROC AUC", "Value": risk_m.get("roc_auc", 0)},
            ])
            fig = px.bar(
                risk_df, x="Metric", y="Value",
                text="Value",
                color_discrete_sequence=["#764ba2"],
                title="Risk Prediction Performance",
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(yaxis_range=[0, 1], height=300, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        st.metric("CGPA Projection RMSE", f"{latest.get('cgpa_rmse', 0):.4f}")

        # ── Model Version History ──
        if len(registry) > 1:
            st.divider()
            st.write("**Model Version History:**")
            history = pd.DataFrame([
                {
                    "Version": f"v{r['version']}",
                    "Model": r["model_name"],
                    "Accuracy": r["metrics"]["accuracy"],
                    "Macro F1": r["metrics"]["macro_f1"],
                    "CGPA RMSE": r.get("cgpa_rmse", 0),
                    "Trained At": r["trained_at"][:19],
                }
                for r in registry
            ])
            st.dataframe(
                history.style.format({
                    "Accuracy": "{:.1%}",
                    "Macro F1": "{:.1%}",
                    "CGPA RMSE": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # ── Full JSON (collapsible) ──
        with st.expander("📄 Raw Model Registry JSON"):
            st.json(registry)

    st.divider()
    st.subheader("📅 Maintenance Timeline")
    st.markdown(
        """
        | Trigger | Action | Cadence |
        |---------|--------|---------|
        | End of semester | Full retrain with new grades | Every semester |
        | Grade distribution shift > 10% | Retrain + recalibrate thresholds | As detected |
        | Scholarship risk recall drop > 15% | Emergency retrain | Immediate |
        | Policy or curriculum changes | Update features + retrain | As needed |
        """
    )
    st.caption("Model versioning: `models/grade_model_vX.pkl` with registry in `models/model_registry.json`")
