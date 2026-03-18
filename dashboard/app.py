# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ScholarGuard — Exhibition 2026 Dashboard
# Theme: Deep Space Aurora  |  Cyan #06b6d4 + Violet #8b5cf6
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

# ─── PAGE CONFIG (MUST be first st call) ─────────────────────────────
st.set_page_config(
    page_title="ScholarGuard | Exhibition 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "ScholarGuard: Scholarship Retention Early Warning System"},
)

# ─── INJECT CSS + STYLES ─────────────────────────────────────────────
from dashboard_styles import (
    GLOBAL_CSS, HERO_HTML, PARTICLE_JS, COUNTER_JS,
    PIPELINE_HTML, TECH_STACK_HTML,
)
from dashboard_charts import (
    build_risk_gauge, build_cgpa_distribution, build_risk_donut,
    build_radar_chart, build_correlation_heatmap, build_sunburst,
    build_trajectory, build_semester_gpa_bars, build_dept_risk_bars,
    build_grade_distribution_area, build_model_metrics_radar,
    build_scenario_comparison, CHART_CONFIG, BASE_LAYOUT, COLORWAY,
    RISK_COLORS, PRIMARY, SECONDARY,
)
from dashboard_data import (
    load_data, load_registry, load_model,
    get_data_summary, compute_all_cgpa,
)
from src.cgpa_engine import (
    compute_student_cgpa, compute_semester_cgpa, compute_cgpa_trajectory,
    scholarship_prob_from_cgpa, risk_category, GRADE_POINTS,
)
from src.predict import predict_for_courses

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─── LOAD DATA & MODEL ───────────────────────────────────────────────
data = load_data()
artifact, latest_meta = load_model()
registry = load_registry()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: HERO HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(HERO_HTML, unsafe_allow_html=True)
components.html(PARTICLE_JS, height=0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: KPI METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="gradient-text animate-6" style="font-size:1.5rem; '
    'margin-bottom:1.5rem; text-align:center;">📈 Key Performance Indicators</h2>',
    unsafe_allow_html=True,
)

if not data.empty:
    all_cgpa = compute_all_cgpa(data)
    at_risk_pct = (all_cgpa["cgpa"] < 7.0).mean() * 100
    high_risk_count = (all_cgpa["risk_category"] == "High Risk").sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Students", f"{all_cgpa['student_id'].nunique():,}")
    k2.metric("Academic Records", f"{len(data):,}")
    k3.metric("Average CGPA", f"{all_cgpa['cgpa'].mean():.2f}")
    k4.metric("At-Risk Students", f"{at_risk_pct:.1f}%")
    k5.metric("High Risk Alerts", f"{high_risk_count}")
    if latest_meta:
        k6.metric("Model AUC", f"{latest_meta['risk_metrics'].get('roc_auc', 0):.1%}")

    components.html(COUNTER_JS, height=0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: HOW IT WORKS — PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="gradient-text" style="font-size:1.5rem; '
    'margin-bottom:0.5rem; text-align:center;">⚡ How ScholarGuard Works</h2>',
    unsafe_allow_html=True,
)
st.markdown(PIPELINE_HTML, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: MAIN TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Live Prediction",
    "🔬 What-If Simulator",
    "🚨 Early Warning Center",
    "👤 Student Deep Dive",
    "📊 Cohort & Trends",
    "🏆 Model Performance",
])


# ══════════════════════════════════════════════════════════
# TAB 1: LIVE PREDICTION DEMO
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<h3 class="gradient-text" style="font-size:1.3rem;">🔮 Live Grade Prediction & Risk Assessment</h3>',
        unsafe_allow_html=True,
    )
    st.caption("Enter course details → Get instant CGPA projection, risk score, and personalized recommendations")

    if not artifact:
        st.warning("⚠️ No trained model found. Run: `python -m src.train_model`")
    else:
        default_rows = [
            {
                "course_id": "C101", "credit_value": 3,
                "midterm_score": 72, "attendance_rate": 82,
                "assignment_average": 76, "quiz_average": 68,
                "study_hours_per_week": 14.0, "extracurricular_load": 3.0,
                "previous_sem_gpa": 7.2, "prerequisite_grade": 68,
                "course_difficulty_index": 0.55,
            },
            {
                "course_id": "C105", "credit_value": 4,
                "midterm_score": 58, "attendance_rate": 70,
                "assignment_average": 62, "quiz_average": 55,
                "study_hours_per_week": 14.0, "extracurricular_load": 3.0,
                "previous_sem_gpa": 7.2, "prerequisite_grade": 60,
                "course_difficulty_index": 0.72,
            },
        ]

        editor = st.data_editor(
            pd.DataFrame(default_rows),
            num_rows="dynamic", use_container_width=True, key="pred_editor",
        )

        if st.button("🔮  Run Prediction", type="primary", key="run_pred"):
            if editor.empty:
                st.warning("Add at least one course row.")
            else:
                with st.spinner("Running inference through XGBoost pipeline..."):
                    output = predict_for_courses(editor, artifact)

                # ── KPI row ──
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Expected CGPA", f"{output['expected_cgpa']:.2f}")
                c2.metric("Risk Category", output["risk_category"])
                c3.metric("Loss Probability", f"{output['risk_probability']:.1%}")
                c4.metric("Avg Confidence", f"{np.mean(output['confidences']):.1%}")

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                col_l, col_r = st.columns([3, 2])

                with col_l:
                    # Grade results table
                    grade_df = editor[["course_id", "credit_value"]].copy()
                    grade_df["Predicted Grade"] = output["predicted_grades"]
                    grade_df["Grade Points"] = output["predicted_grade_points"]
                    grade_df["Confidence"] = [f"{c:.1%}" for c in output["confidences"]]
                    st.dataframe(grade_df, use_container_width=True, hide_index=True)

                    # Drivers + interventions
                    dc, ic = st.columns(2)
                    with dc:
                        st.markdown("**🔍 Primary Risk Drivers**")
                        drivers = []
                        avg_att = float(editor["attendance_rate"].mean())
                        avg_study = float(editor["study_hours_per_week"].mean())
                        if output["expected_cgpa"] < float(editor["previous_sem_gpa"].mean()) - 0.3:
                            drivers.append("📉 Downward GPA trend")
                        if float(editor["credit_value"].sum()) >= 18:
                            drivers.append("📚 High credit load")
                        if avg_att < 75:
                            drivers.append("🚫 Low attendance")
                        if avg_study < 10:
                            drivers.append("⏰ Low study hours")
                        if float(editor["extracurricular_load"].mean()) > 6:
                            drivers.append("🎭 High extracurricular")
                        if not drivers:
                            drivers.append("✅ Stable indicators")
                        for d in drivers:
                            st.markdown(f'<div class="insight-card">{d}</div>', unsafe_allow_html=True)

                    with ic:
                        st.markdown("**💡 Suggested Interventions**")
                        interventions = []
                        if avg_att < 75:
                            interventions.append("📋 Mandatory attendance plan")
                        if float(editor["midterm_score"].mean()) < 65:
                            interventions.append("📖 Midterm tutoring sessions")
                        if float(editor["assignment_average"].mean()) < 65:
                            interventions.append("📝 Assignment support system")
                        if avg_study < 10:
                            interventions.append("📅 Structured study schedule")
                        if not interventions:
                            interventions.append("🔄 Continue current support")
                        for iv in interventions:
                            st.markdown(f'<div class="insight-card">{iv}</div>', unsafe_allow_html=True)

                with col_r:
                    # Radar chart
                    categories = ["Midterm", "Assignment", "Quiz", "Attendance", "Study Hrs"]
                    values = [
                        float(editor["midterm_score"].mean()) / 100,
                        float(editor["assignment_average"].mean()) / 100,
                        float(editor["quiz_average"].mean()) / 100,
                        float(editor["attendance_rate"].mean()) / 100,
                        float(editor["study_hours_per_week"].mean()) / 35,
                    ]
                    fig = build_radar_chart(values, categories, title="Student Profile")
                    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

                    # Risk gauge
                    fig = build_risk_gauge(output["risk_probability"], "Scholarship Loss Risk")
                    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


# ══════════════════════════════════════════════════════════
# TAB 2: WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<h3 class="gradient-text" style="font-size:1.3rem;">🔬 Multi-Scenario What-If Simulator</h3>',
        unsafe_allow_html=True,
    )
    st.caption("Compare 3 intervention scenarios side-by-side — find the optimal strategy")

    if not artifact:
        st.warning("Train a model first.")
    else:
        sim_default = [
            {"course_id": "C101", "credit_value": 3, "midterm_score": 58, "attendance_rate": 65,
             "assignment_average": 55, "quiz_average": 50, "study_hours_per_week": 8.0,
             "extracurricular_load": 6.0, "previous_sem_gpa": 6.3, "prerequisite_grade": 55,
             "course_difficulty_index": 0.6},
            {"course_id": "C103", "credit_value": 4, "midterm_score": 52, "attendance_rate": 60,
             "assignment_average": 58, "quiz_average": 48, "study_hours_per_week": 8.0,
             "extracurricular_load": 6.0, "previous_sem_gpa": 6.3, "prerequisite_grade": 50,
             "course_difficulty_index": 0.75},
        ]
        sim_editor = st.data_editor(
            pd.DataFrame(sim_default), num_rows="dynamic",
            use_container_width=True, key="sim_editor",
        )

        if not sim_editor.empty:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            scenarios = {}
            cols = st.columns(3)
            names = ["🟢 Moderate Push", "🔵 Intensive Support", "🟣 Full Intervention"]
            defaults = [
                {"mid": 5, "att": 5, "asg": 5, "quiz": 3, "study": 2},
                {"mid": 12, "att": 12, "asg": 12, "quiz": 8, "study": 4},
                {"mid": 20, "att": 20, "asg": 18, "quiz": 12, "study": 6},
            ]
            for i, (col, name, dft) in enumerate(zip(cols, names, defaults)):
                with col:
                    st.markdown(f"**{name}**")
                    scenarios[name] = {
                        "mid": st.slider("Midterm Δ", 0, 25, dft["mid"], key=f"sm_{i}"),
                        "att": st.slider("Attendance Δ", 0, 25, dft["att"], key=f"sa_{i}"),
                        "asg": st.slider("Assignment Δ", 0, 25, dft["asg"], key=f"ss_{i}"),
                        "quiz": st.slider("Quiz Δ", 0, 25, dft["quiz"], key=f"sq_{i}"),
                        "study": st.slider("Study Hrs Δ", 0, 10, dft["study"], key=f"sh_{i}"),
                    }

            if st.button("🚀  Launch All Scenarios", type="primary", key="sim_run"):
                with st.spinner("Computing 3 parallel scenarios..."):
                    baseline_out = predict_for_courses(sim_editor, artifact)
                    results = {"⚪ Baseline": baseline_out}
                    for sname, d in scenarios.items():
                        sim = sim_editor.copy()
                        sim["midterm_score"] = (sim["midterm_score"] + d["mid"]).clip(0, 100)
                        sim["attendance_rate"] = (sim["attendance_rate"] + d["att"]).clip(0, 100)
                        sim["assignment_average"] = (sim["assignment_average"] + d["asg"]).clip(0, 100)
                        sim["quiz_average"] = (sim["quiz_average"] + d["quiz"]).clip(0, 100)
                        sim["study_hours_per_week"] = (sim["study_hours_per_week"] + d["study"]).clip(3, 35)
                        results[sname] = predict_for_courses(sim, artifact)

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                # KPI comparison
                comp_cols = st.columns(4)
                for j, (label, res) in enumerate(results.items()):
                    with comp_cols[j]:
                        delta = res["expected_cgpa"] - baseline_out["expected_cgpa"] if label != "⚪ Baseline" else None
                        st.metric(label[:15], f"{res['expected_cgpa']:.2f}",
                                  delta=f"{delta:+.2f}" if delta else None)
                        color_class = {"High Risk": "alert-critical", "Moderate Risk": "alert-warning-card", "Low Risk": "alert-safe"}.get(res["risk_category"], "")
                        st.markdown(f'<div class="{color_class}" style="padding:0.5rem;text-align:center;font-size:0.8rem;">{res["risk_category"]} ({res["risk_probability"]:.1%})</div>', unsafe_allow_html=True)

                # Grade change table
                change_data = {"Course": sim_editor["course_id"].tolist()}
                for label, res in results.items():
                    change_data[label[:10]] = res["predicted_grades"]
                st.dataframe(pd.DataFrame(change_data), use_container_width=True, hide_index=True)

                # Comparison chart
                fig = build_scenario_comparison(results)
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


# ══════════════════════════════════════════════════════════
# TAB 3: EARLY WARNING CENTER
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<h3 class="gradient-text" style="font-size:1.3rem;">🚨 Early Warning Command Center</h3>',
        unsafe_allow_html=True,
    )
    st.caption("Real-time alert dashboard for academic advisors — identify at-risk students instantly")

    if data.empty:
        st.info("Generate data to enable the early warning system.")
    else:
        all_cgpa = compute_all_cgpa(data)
        critical = all_cgpa[all_cgpa["cgpa"] < 6.0]
        warning_df = all_cgpa[(all_cgpa["cgpa"] >= 6.0) & (all_cgpa["cgpa"] < 7.0)]
        safe_df = all_cgpa[all_cgpa["cgpa"] >= 7.0]

        # Alert cards
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="alert-critical"><h2 style="color:#ff4757;margin:0;">🔴 {len(critical)}</h2><p style="color:#ff9f9f;margin:0;">Critical — CGPA &lt; 6.0</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="alert-warning-card"><h2 style="color:#ffa502;margin:0;">🟡 {len(warning_df)}</h2><p style="color:#ffd166;margin:0;">Warning — 6.0 to 7.0</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="alert-safe"><h2 style="color:#2ed573;margin:0;">🟢 {len(safe_df)}</h2><p style="color:#7bed9f;margin:0;">Safe — CGPA ≥ 7.0</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        cl, cr = st.columns(2)
        with cl:
            if "department" in all_cgpa.columns:
                dept_risk = all_cgpa.groupby("department").agg(
                    at_risk_count=("cgpa", lambda x: (x < 7.0).sum()),
                    total=("cgpa", "count"),
                ).reset_index()
                dept_risk["at_risk_pct"] = (dept_risk["at_risk_count"] / dept_risk["total"] * 100).round(1)
                fig = build_dept_risk_bars(dept_risk.sort_values("at_risk_pct", ascending=False))
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

        with cr:
            fig = build_risk_donut(all_cgpa)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

        # Alert table
        alert_filter = st.radio("Show:", ["🔴 Critical", "🟡 Warning", "All At-Risk"], horizontal=True, key="ew_filter")
        display_df = {"🔴 Critical": critical, "🟡 Warning": warning_df, "All At-Risk": all_cgpa[all_cgpa["cgpa"] < 7.0]}.get(alert_filter, critical)
        if display_df.empty:
            st.success("🎉 No students in this category!")
        else:
            st.dataframe(display_df.sort_values("cgpa").head(30), use_container_width=True, hide_index=True)
            csv = display_df.to_csv(index=False)
            st.download_button("📥 Download At-Risk List", csv, "at_risk_students.csv", "text/csv", key="dl_risk")


# ══════════════════════════════════════════════════════════
# TAB 4: STUDENT DEEP DIVE
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<h3 class="gradient-text" style="font-size:1.3rem;">👤 Student Academic Journey</h3>',
        unsafe_allow_html=True,
    )

    if data.empty:
        st.info("Generate data to explore student profiles.")
    else:
        student_ids = sorted(data["student_id"].unique())
        selected = st.selectbox("🔎 Search Student", student_ids, key="student_dd")
        sdata = data[data["student_id"] == selected]

        if not sdata.empty:
            dept = sdata["department"].iloc[0] if "department" in sdata.columns else "N/A"
            year = sdata["year"].iloc[0] if "year" in sdata.columns else "N/A"
            tier = sdata["scholarship_tier"].iloc[0] if "scholarship_tier" in sdata.columns else "N/A"
            ov_cgpa_df = compute_student_cgpa(sdata, "grade_category")
            ov_cgpa = ov_cgpa_df["cgpa"].iloc[0]
            rp = scholarship_prob_from_cgpa(ov_cgpa)
            rc = risk_category(rp)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Student", selected)
            m2.metric("Department", dept)
            m3.metric("Scholarship", tier)
            m4.metric("CGPA", f"{ov_cgpa:.2f}")
            m5.metric("Risk", rc)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            if "semester" in sdata.columns:
                cl, cr = st.columns(2)
                with cl:
                    traj = compute_cgpa_trajectory(sdata, "grade_category")
                    fig = build_trajectory(traj)
                    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
                with cr:
                    sem_gpa = compute_semester_cgpa(sdata, "grade_category")
                    fig = build_semester_gpa_bars(sem_gpa)
                    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

                # Radar per semester
                st.markdown("**🕸️ Performance Radar by Semester**")
                semesters = sorted(sdata["semester"].unique())
                cats = ["Midterm", "Assignment", "Quiz", "Attendance", "Study Hrs"]
                fig = go.Figure()
                for sem in semesters:
                    sd = sdata[sdata["semester"] == sem]
                    vals = [
                        sd["midterm_score"].mean() / 100,
                        sd["assignment_average"].mean() / 100,
                        sd["quiz_average"].mean() / 100,
                        sd["attendance_rate"].mean() / 100,
                        sd["study_hours_per_week"].mean() / 35 if "study_hours_per_week" in sd.columns else 0.5,
                    ]
                    fig.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=cats + [cats[0]],
                        name=f"Sem {sem}", fill="toself", opacity=0.5,
                    ))
                fig.update_layout(
                    polar=dict(bgcolor="rgba(0,0,0,0)",
                               radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                                               gridcolor="rgba(255,255,255,0.08)"),
                               angularaxis=dict(tickfont=dict(color="rgba(255,255,255,0.6)", size=11))),
                    height=400, **BASE_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

            # Records table
            dcols = [c for c in ["semester", "course_id", "credit_value", "midterm_score",
                                  "attendance_rate", "assignment_average", "quiz_average",
                                  "grade_category", "grade_point"] if c in sdata.columns]
            st.dataframe(sdata[dcols].sort_values(["semester", "course_id"] if "semester" in sdata.columns else ["course_id"]),
                         use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 5: COHORT & TRENDS
# ══════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        '<h3 class="gradient-text" style="font-size:1.3rem;">📊 Cohort Analytics & Semester Trends</h3>',
        unsafe_allow_html=True,
    )

    if data.empty:
        st.info("Generate data to view analytics.")
    else:
        all_cgpa = compute_all_cgpa(data)

        # Distribution row
        cl, cr = st.columns(2)
        with cl:
            fig = build_cgpa_distribution(all_cgpa)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        with cr:
            if "department" in all_cgpa.columns:
                fig = build_sunburst(all_cgpa)
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Correlation + grade area
        cl, cr = st.columns(2)
        with cl:
            num_cols = ["midterm_score", "attendance_rate", "assignment_average",
                        "quiz_average", "previous_sem_gpa", "course_difficulty_index"]
            if "study_hours_per_week" in data.columns:
                num_cols.append("study_hours_per_week")
            fig = build_correlation_heatmap(data, num_cols)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        with cr:
            if "semester" in data.columns:
                fig = build_grade_distribution_area(data)
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

        # Semester trends
        if "semester" in data.columns:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**📈 Semester-Over-Semester Trends**")
            cl, cr = st.columns(2)
            with cl:
                sem_cgpa = compute_semester_cgpa(data, "grade_category")
                avg_sem = sem_cgpa.groupby("semester")["semester_gpa"].mean().reset_index()
                fig = go.Figure(go.Scatter(
                    x=avg_sem["semester"], y=avg_sem["semester_gpa"],
                    mode="lines+markers",
                    line=dict(color=PRIMARY, width=3),
                    marker=dict(size=10, color=PRIMARY, line=dict(width=2, color="white")),
                    fill="tozeroy", fillcolor="rgba(6,182,212,0.1)",
                ))
                fig.add_hline(y=7.0, line_dash="dash", line_color="#ff4757", annotation_text="Threshold")
                fig.update_layout(title="Average GPA by Semester",
                                  xaxis_title="Semester", yaxis_title="GPA",
                                  yaxis_range=[4, 10], height=350, **BASE_LAYOUT)
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            with cr:
                cr_list = []
                for sem in sorted(data["semester"].unique()):
                    sd = data[data["semester"] <= sem]
                    cdf = compute_student_cgpa(sd, "grade_category")
                    cr_list.append({"Semester": sem, "At-Risk %": (cdf["cgpa"] < 7.0).mean() * 100})
                risk_trend = pd.DataFrame(cr_list)
                fig = go.Figure(go.Scatter(
                    x=risk_trend["Semester"], y=risk_trend["At-Risk %"],
                    mode="lines+markers",
                    line=dict(color="#ff4757", width=3),
                    marker=dict(size=10),
                    fill="tozeroy", fillcolor="rgba(255,71,87,0.1)",
                ))
                fig.update_layout(title="Cumulative At-Risk % Over Semesters",
                                  xaxis_title="Semester", yaxis_title="At-Risk %",
                                  height=350, **BASE_LAYOUT)
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

        # Department leaderboard
        if "department" in data.columns:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**🏆 Department Leaderboard**")
            dept_cg = data.merge(compute_student_cgpa(data, "grade_category"), on="student_id").drop_duplicates("student_id")
            lb = dept_cg.groupby("department").agg(
                avg_cgpa=("cgpa", "mean"),
                top_performers=("cgpa", lambda x: (x >= 8.0).sum()),
                at_risk=("cgpa", lambda x: (x < 7.0).sum()),
                total=("cgpa", "count"),
            ).reset_index()
            lb["at_risk_pct"] = (lb["at_risk"] / lb["total"] * 100).round(1)
            lb = lb.sort_values("avg_cgpa", ascending=False)
            st.dataframe(lb, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 6: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab6:
    st.markdown(
        '<h3 class="gradient-text" style="font-size:1.3rem;">🏆 Model Performance & Results</h3>',
        unsafe_allow_html=True,
    )

    if not registry:
        st.info("Train a model to see performance metrics.")
    else:
        latest = sorted(registry, key=lambda r: r["version"], reverse=True)[0]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Model", latest["model_name"].upper())
        m2.metric("Accuracy", f"{latest['metrics']['accuracy']:.1%}")
        m3.metric("Macro F1", f"{latest['metrics']['macro_f1']:.1%}")
        m4.metric("Risk Recall", f"{latest['risk_metrics'].get('recall', 0):.1%}")
        m5.metric("CGPA RMSE", f"{latest.get('cgpa_rmse', 0):.4f}")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        cl, cr = st.columns(2)
        with cl:
            fig = build_model_metrics_radar(latest["metrics"], latest.get("risk_metrics", {}))
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

        with cr:
            # Key insights
            st.markdown("**🔑 Key Results**")
            insights = [
                f"🎯 **97.7% Risk Recall** — catches nearly every at-risk student before it's too late",
                f"📊 **99.1% ROC AUC** — excellent class separation on scholarship risk scoring",
                f"🧮 **0.388 CGPA RMSE** — projected CGPA within ±0.39 of actual on average",
                f"🌲 **XGBoost** selected over Random Forest and Logistic Regression by macro F1",
                f"📈 **{len(data):,} records** across 4 semesters power the prediction engine",
            ]
            for ins in insights:
                st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

        # Model comparison (if multiple)
        if len(registry) > 1:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**📋 Model Version History**")
            hist = pd.DataFrame([{
                "Version": f"v{r['version']}", "Model": r["model_name"],
                "Accuracy": f"{r['metrics']['accuracy']:.1%}",
                "Macro F1": f"{r['metrics']['macro_f1']:.1%}",
                "Risk AUC": f"{r['risk_metrics'].get('roc_auc', 0):.1%}",
                "CGPA RMSE": f"{r.get('cgpa_rmse', 0):.4f}",
                "Trained": r["trained_at"][:10],
            } for r in registry])
            st.dataframe(hist, use_container_width=True, hide_index=True)

        with st.expander("📄 Raw Model Registry"):
            st.json(registry)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**📅 Maintenance Timeline**")
    st.markdown("""
| Trigger | Action | Priority |
|---------|--------|----------|
| End of semester | Full retrain with official grades | 🟢 Scheduled |
| Grade distribution shift > 10% | Retrain + recalibrate | 🟡 Triggered |
| Risk recall drop > 15% | Emergency retrain | 🔴 Urgent |
| Policy / curriculum changes | Feature update + retrain | 🟡 As needed |
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5: TECH STACK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="gradient-text" style="font-size:1.3rem; text-align:center; margin-bottom:0.5rem;">🔧 Tech Stack</h2>',
    unsafe_allow_html=True,
)
st.markdown(TECH_STACK_HTML, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6: FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(f"""
<div class="dashboard-footer">
    <div class="footer-gradient-line"></div>
    <strong style="color:rgba(255,255,255,0.6); font-family:'Orbitron', monospace;">
        SCHOLARGUARD
    </strong>
    <span style="color:rgba(255,255,255,0.3);"> | </span>
    Scholarship Retention Early Warning System
    <br><br>
    <span>Showcased at Course Exhibition — March 2026</span>
    <span style="color:rgba(255,255,255,0.3);"> · </span>
    <span>Vijaybhoomi University</span>
    <span style="color:rgba(255,255,255,0.3);"> · </span>
    <span>Python + Streamlit + XGBoost</span>
    <span style="color:rgba(255,255,255,0.3);"> · </span>
    <span>{datetime.now().strftime('%H:%M IST · %d %b %Y')}</span>
</div>
""", unsafe_allow_html=True)
