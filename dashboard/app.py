"""Streamlit dashboard for Scholarship Retention Early Warning System — Dark Retro Pixel Theme."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.dashboard_charts import (
    cgpa_trend_line,
    feature_correlation_heatmap,
    fee_payment_vs_risk,
    prediction_contribution_chart,
    risk_distribution_donut_by_department,
    risk_factor_breakdown,
    roc_auc_history,
)
from dashboard.dashboard_data import (
    build_prediction_row,
    feature_importance_table,
    load_data,
    load_latest_model,
    load_registry,
    model_history_table,
    student_semester_view,
)

st.set_page_config(page_title="SREWS // Dashboard", page_icon="🎓", layout="wide")

# ═══════════════════════════════════════════════════════════════
# DARK RETRO PIXEL THEME — Inspired by Agent Arena & sarthi.exe
# ═══════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Google Fonts ──────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

    /* ── CSS Variables ────────────────────────────────────── */
    :root {
        --bg-body: #000000;
        --bg-surface: #0a0a0a;
        --bg-card: #111111;
        --bg-card-hover: #1a1a1a;
        --bg-input: #080808;
        --border: #2a2a2a;
        --border-light: #3a3a3a;
        --text-primary: #a0a0a0;
        --text-bright: #ffffff;
        --text-dim: #555555;
        --accent: #ffffff;
        --alert: #ff4444;
        --safe: #c8c8c8;
        --font-pixel: 'Press Start 2P', monospace;
        --font-vt: 'VT323', monospace;
    }

    /* ── Body & App Background ────────────────────────────── */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .stMainBlockContainer,
    .main .block-container {
        background-color: var(--bg-body) !important;
        color: var(--text-primary) !important;
    }
    html, body, [data-testid="stAppViewContainer"] > section {
        background-color: var(--bg-body) !important;
    }

    /* ── CRT Scanline Overlay ─────────────────────────────── */
    .stApp::after {
        content: '';
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 9999;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 0, 0, 0.06) 2px,
            rgba(0, 0, 0, 0.06) 4px
        );
    }

    /* ── All Text → Monospace ─────────────────────────────── */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] label,
    [data-testid="stText"],
    .stMarkdown, .stText, p, span, label, div {
        font-family: var(--font-vt) !important;
        color: var(--text-primary) !important;
    }

    /* ── Headings — Pixel Font ────────────────────────────── */
    h1 {
        font-family: var(--font-pixel) !important;
        color: var(--text-bright) !important;
        font-size: 1.1rem !important;
        text-shadow: 0 0 12px rgba(255,255,255,0.3), 2px 2px 0 rgba(0,0,0,0.8);
        letter-spacing: 2px !important;
        text-transform: uppercase;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid var(--border) !important;
    }
    h2, h3 {
        font-family: var(--font-pixel) !important;
        color: var(--text-bright) !important;
        font-size: 0.7rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase;
        text-shadow: 0 0 8px rgba(255,255,255,0.2);
    }

    /* ── Caption / Subtitle ───────────────────────────────── */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        font-family: var(--font-pixel) !important;
        font-size: 0.5rem !important;
        color: var(--text-dim) !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 2px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
        font-family: var(--font-vt) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: var(--font-pixel) !important;
        color: var(--text-bright) !important;
        font-size: 0.6rem !important;
        text-shadow: 0 0 8px rgba(255,255,255,0.3);
    }

    /* ── Tabs — Retro Terminal Nav ─────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-surface) !important;
        border-bottom: 2px solid var(--border) !important;
        gap: 0 !important;
        padding: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-vt) !important;
        font-size: 1.2rem !important;
        color: var(--text-dim) !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 10px 20px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.1s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-bright) !important;
        text-shadow: 0 0 8px rgba(255,255,255,0.3);
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-bright) !important;
        border-bottom: 2px solid var(--text-bright) !important;
        text-shadow: 0 0 8px rgba(255,255,255,0.3);
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--text-bright) !important;
        height: 2px !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Metric Cards — Pixel HUD boxes ───────────────────── */
    .metric-card {
        background: var(--bg-card) !important;
        border: 2px solid var(--border) !important;
        border-radius: 0 !important;
        padding: 16px 20px !important;
        box-shadow: 4px 4px 0 rgba(0,0,0,0.5) !important;
        position: relative;
        transition: all 0.15s;
    }
    .metric-card:hover {
        border-color: var(--text-bright) !important;
        box-shadow: 4px 4px 0 rgba(255,255,255,0.1), 0 0 12px rgba(255,255,255,0.05) !important;
    }
    .metric-card .metric-label {
        font-family: var(--font-pixel) !important;
        font-size: 0.45rem !important;
        color: var(--text-dim) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        display: block;
    }
    .metric-card .metric-value {
        font-family: var(--font-pixel) !important;
        font-size: 1.1rem !important;
        color: var(--text-bright) !important;
        text-shadow: 0 0 6px rgba(255,255,255,0.3);
        display: block;
    }

    /* ── Prediction Result Cards ───────────────────────────── */
    .predict-card {
        border-radius: 0 !important;
        padding: 18px 24px !important;
        font-family: var(--font-pixel) !important;
        font-size: 0.65rem !important;
        font-weight: 400 !important;
        text-align: center;
        margin-top: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 4px 4px 0 rgba(0,0,0,0.5);
    }
    .predict-risk {
        background: var(--bg-card) !important;
        color: var(--alert) !important;
        border: 2px solid var(--alert) !important;
        text-shadow: 0 0 12px rgba(255,68,68,0.4);
        box-shadow: 4px 4px 0 rgba(255,68,68,0.15), 0 0 20px rgba(255,68,68,0.08);
    }
    .predict-safe {
        background: var(--bg-card) !important;
        color: var(--text-bright) !important;
        border: 2px solid var(--text-bright) !important;
        text-shadow: 0 0 12px rgba(255,255,255,0.4);
        box-shadow: 4px 4px 0 rgba(255,255,255,0.1), 0 0 20px rgba(255,255,255,0.05);
    }

    /* ── Buttons — Pixel Style ────────────────────────────── */
    .stButton > button,
    button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        font-family: var(--font-pixel) !important;
        font-size: 0.55rem !important;
        background: var(--text-bright) !important;
        color: #000 !important;
        border: 2px solid #cccccc !important;
        border-radius: 0 !important;
        padding: 12px 24px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 3px 3px 0 rgba(0,0,0,0.5) !important;
        cursor: pointer;
        transition: all 0.1s !important;
    }
    .stButton > button:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background: #dddddd !important;
        transform: translate(-1px, -1px);
        box-shadow: 4px 4px 0 rgba(255,255,255,0.2) !important;
    }
    .stButton > button:active,
    [data-testid="stBaseButton-primary"]:active {
        transform: translate(2px, 2px);
        box-shadow: 1px 1px 0 rgba(0,0,0,0.5) !important;
    }

    /* ── Select Boxes & Sliders ───────────────────────────── */
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label,
    [data-testid="stCheckbox"] label {
        font-family: var(--font-pixel) !important;
        font-size: 0.45rem !important;
        color: var(--text-dim) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-baseweb="select"] {
        background: var(--bg-input) !important;
        border: 2px solid var(--border) !important;
        border-radius: 0 !important;
        box-shadow: inset 2px 2px 0 rgba(0,0,0,0.3) !important;
    }
    [data-baseweb="select"] * {
        font-family: var(--font-vt) !important;
        color: var(--text-bright) !important;
        background: var(--bg-input) !important;
    }
    [data-baseweb="select"]:focus-within {
        border-color: var(--text-bright) !important;
        box-shadow: inset 2px 2px 0 rgba(0,0,0,0.3), 0 0 8px rgba(255,255,255,0.15) !important;
    }

    /* ── Select dropdown menu ─────────────────────────────── */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"],
    ul[data-baseweb="menu"] {
        background: var(--bg-card) !important;
        border: 2px solid var(--border) !important;
        border-radius: 0 !important;
    }
    [data-baseweb="menu"] li,
    [role="option"] {
        font-family: var(--font-vt) !important;
        color: var(--text-primary) !important;
        background: var(--bg-card) !important;
    }
    [data-baseweb="menu"] li:hover,
    [role="option"]:hover {
        background: var(--bg-card-hover) !important;
        color: var(--text-bright) !important;
    }
    [aria-selected="true"][role="option"] {
        background: rgba(255,255,255,0.08) !important;
        color: var(--text-bright) !important;
    }

    /* ── Slider Track ─────────────────────────────────────── */
    [data-baseweb="slider"] {
        padding-top: 12px !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: var(--text-bright) !important;
        border: 2px solid #ccc !important;
        border-radius: 0 !important;
        width: 14px !important;
        height: 14px !important;
        box-shadow: 2px 2px 0 rgba(0,0,0,0.4) !important;
    }
    [data-testid="stSlider"] [data-testid="stTickBar"] {
        background: var(--border) !important;
    }

    /* ── Checkbox ──────────────────────────────────────────── */
    [data-testid="stCheckbox"] span[role="checkbox"] {
        border: 2px solid var(--border) !important;
        border-radius: 0 !important;
        background: var(--bg-input) !important;
    }
    [data-testid="stCheckbox"] span[role="checkbox"][aria-checked="true"] {
        background: var(--text-bright) !important;
        border-color: #ccc !important;
    }

    /* ── Data Table ────────────────────────────────────────── */
    [data-testid="stDataFrame"],
    .stDataFrame {
        border: 2px solid var(--border) !important;
        border-radius: 0 !important;
    }
    [data-testid="stDataFrame"] * {
        font-family: var(--font-vt) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stDataFrame"] th {
        font-family: var(--font-pixel) !important;
        font-size: 0.4rem !important;
        text-transform: uppercase !important;
        color: var(--text-dim) !important;
        background: var(--bg-surface) !important;
        border-bottom: 2px solid var(--border) !important;
    }
    [data-testid="stDataFrame"] td {
        background: var(--bg-card) !important;
        border-bottom: 1px solid var(--border) !important;
    }
    [data-testid="stDataFrame"] tr:hover td {
        background: var(--bg-card-hover) !important;
    }

    /* ── Plotly Charts — Dark Container ────────────────────── */
    [data-testid="stPlotlyChart"] {
        border: 2px solid var(--border) !important;
        background: var(--bg-card) !important;
        box-shadow: 4px 4px 0 rgba(0,0,0,0.5) !important;
        border-radius: 0 !important;
    }

    /* ── Info / Warning Boxes ──────────────────────────────── */
    [data-testid="stAlert"] {
        background: var(--bg-card) !important;
        border: 2px solid var(--border) !important;
        border-radius: 0 !important;
        color: var(--text-primary) !important;
        font-family: var(--font-vt) !important;
        box-shadow: 3px 3px 0 rgba(0,0,0,0.4) !important;
    }

    /* ── Accent Header ────────────────────────────────────── */
    .accent-header {
        font-family: var(--font-pixel) !important;
        font-size: 0.55rem !important;
        color: var(--text-bright) !important;
        text-shadow: 0 0 8px rgba(255,255,255,0.3);
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── System Status Sidebar Badge ──────────────────────── */
    .system-status {
        background: var(--bg-card) !important;
        border: 2px solid var(--border) !important;
        padding: 12px 14px !important;
        font-family: var(--font-vt) !important;
        font-size: 1rem !important;
        color: var(--text-primary) !important;
        box-shadow: 3px 3px 0 rgba(0,0,0,0.4) !important;
        margin-top: 8px;
    }
    .system-status .status-label {
        font-family: var(--font-pixel) !important;
        font-size: 0.4rem !important;
        color: var(--text-dim) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: block;
        margin-bottom: 4px;
    }
    .system-status .status-value {
        color: var(--text-bright) !important;
        text-shadow: 0 0 6px rgba(255,255,255,0.3);
    }

    /* ── Pixel Divider ────────────────────────────────────── */
    .pixel-divider {
        height: 2px;
        background: repeating-linear-gradient(
            90deg,
            var(--border) 0px, var(--border) 4px,
            transparent 4px, transparent 8px
        );
        margin: 16px 0;
    }

    /* ── Custom Scrollbar ─────────────────────────────────── */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-body); }
    ::-webkit-scrollbar-thumb { background: var(--border); }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

    /* ── Hide Streamlit Branding ───────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] {
        background: var(--bg-body) !important;
        backdrop-filter: none !important;
    }

    /* ── Expander ──────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 2px solid var(--border) !important;
        border-radius: 0 !important;
    }
    [data-testid="stExpander"] summary {
        font-family: var(--font-pixel) !important;
        font-size: 0.5rem !important;
        color: var(--text-bright) !important;
    }

    /* ── Remove column gap borders ────────────────────────── */
    [data-testid="stHorizontalBlock"] {
        gap: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Title ─────────────────────────────────────────────────────
st.title("SCHOLARSHIP RETENTION EARLY WARNING SYSTEM")
st.caption("HACKATHON 3 // STSE203 // VIJAYBHOOMI UNIVERSITY")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙ SYSTEM STATUS")
    st.markdown(
        """
        <div class='system-status'>
            <span class='status-label'>THEME</span>
            <span class='status-value'>DARK RETRO PIXEL</span>
        </div>
        <div class='system-status'>
            <span class='status-label'>ENGINE</span>
            <span class='status-value'>SREWS v2.0</span>
        </div>
        <div class='system-status'>
            <span class='status-label'>MODE</span>
            <span class='status-value'>● ONLINE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='pixel-divider'></div>", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────
data = load_data()
registry = load_registry()
artifact, _ = load_latest_model()
sem_view = student_semester_view(data)
importance_df = feature_importance_table(artifact)
history_df = model_history_table(registry)

if data.empty:
    st.warning("No academic data found. Run the data generator first.")

if artifact is None:
    st.warning("No trained model artifact found. Run training to enable prediction.")

# ── KPI Metric Cards ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"<div class='metric-card'>"
        f"<span class='metric-label'>RECORDS</span>"
        f"<span class='metric-value'>{len(data):,}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='metric-card'>"
        f"<span class='metric-label'>STUDENTS</span>"
        f"<span class='metric-value'>{data['student_id'].nunique() if not data.empty else 0:,}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
with c3:
    avg_risk = float(sem_view["scholarship_at_risk"].mean() * 100) if not sem_view.empty else 0.0
    st.markdown(
        f"<div class='metric-card'>"
        f"<span class='metric-label'>AT-RISK RATE</span>"
        f"<span class='metric-value'>{avg_risk:.1f}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
with c4:
    roc_val = float(history_df["roc_auc"].iloc[-1]) if not history_df.empty else 0.0
    st.markdown(
        f"<div class='metric-card'>"
        f"<span class='metric-label'>LATEST ROC-AUC</span>"
        f"<span class='metric-value'>{roc_val:.3f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────
overview_tab, eda_tab, predictor_tab, history_tab = st.tabs(
    ["OVERVIEW", "EDA INSIGHTS", "🎯 RISK PREDICTOR", "MODEL HISTORY"]
)

with overview_tab:
    st.subheader("Overview")
    oc1, oc2 = st.columns(2)
    with oc1:
        st.plotly_chart(risk_distribution_donut_by_department(sem_view), use_container_width=True)
    with oc2:
        st.plotly_chart(cgpa_trend_line(sem_view), use_container_width=True)

with eda_tab:
    st.subheader("EDA and Risk Drivers")
    ec1, ec2 = st.columns(2)
    with ec1:
        numeric_features = [
            "midterm_score",
            "attendance_rate",
            "assignment_average",
            "quiz_average",
            "study_hours_per_week",
            "extracurricular_load",
            "previous_sem_gpa",
            "counselling_sessions_attended",
            "library_usage_hours_per_week",
            "mental_health_score",
            "cgpa_this_semester",
            "cgpa_trend",
            "scholarship_at_risk",
        ]
        st.plotly_chart(feature_correlation_heatmap(data, numeric_features), use_container_width=True)
    with ec2:
        st.plotly_chart(risk_factor_breakdown(importance_df), use_container_width=True)

    st.plotly_chart(fee_payment_vs_risk(sem_view), use_container_width=True)

with predictor_tab:
    st.subheader("🎯 Scholarship Risk Predictor")

    if artifact is None:
        st.info("Train a model to use prediction.")
    else:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            department = st.selectbox("Department", ["CS", "ECE", "ME", "CE", "EE"])
            year = st.selectbox("Year", [1, 2, 3, 4], index=1)
            scholarship_tier = st.selectbox(
                "Scholarship Tier",
                ["Merit-100%", "Merit-75%", "Merit-50%", "Need-Based"],
            )
            family_income_bracket = st.selectbox(
                "Family Income Bracket", ["low", "lower_mid", "mid", "upper_mid"]
            )
            fee_payment_status = st.selectbox(
                "Fee Payment Status", ["on_time", "late", "defaulted"]
            )
            hostel_resident = st.checkbox("Hostel Resident", value=True)

        with fc2:
            midterm_score = st.slider("Midterm Score", 0.0, 100.0, 68.0)
            attendance_rate = st.slider("Attendance Rate", 35.0, 100.0, 78.0)
            assignment_average = st.slider("Assignment Average", 0.0, 100.0, 70.0)
            quiz_average = st.slider("Quiz Average", 0.0, 100.0, 66.0)
            study_hours_per_week = st.slider("Study Hours/Week", 3.0, 35.0, 14.0)
            extracurricular_load = st.slider("Extracurricular Load", 0.0, 10.0, 3.0)

        with fc3:
            previous_sem_gpa = st.slider("Previous Semester GPA", 0.0, 10.0, 7.1)
            counselling_sessions_attended = st.slider("Counselling Sessions", 0, 5, 1)
            library_usage_hours_per_week = st.slider("Library Usage Hours/Week", 0.0, 30.0, 6.0)
            mental_health_score = st.slider("Mental Health Score", 1.0, 10.0, 6.5)

        if st.button("⚡ RUN RISK PREDICTION", type="primary"):
            form_values = {
                "department": department,
                "year": year,
                "scholarship_tier": scholarship_tier,
                "midterm_score": midterm_score,
                "attendance_rate": attendance_rate,
                "assignment_average": assignment_average,
                "quiz_average": quiz_average,
                "study_hours_per_week": study_hours_per_week,
                "extracurricular_load": extracurricular_load,
                "previous_sem_gpa": previous_sem_gpa,
                "family_income_bracket": family_income_bracket,
                "fee_payment_status": fee_payment_status,
                "hostel_resident": hostel_resident,
                "counselling_sessions_attended": counselling_sessions_attended,
                "library_usage_hours_per_week": library_usage_hours_per_week,
                "mental_health_score": mental_health_score,
            }

            row_df = build_prediction_row(form_values)
            pipeline = artifact["model"]
            proba = float(pipeline.predict_proba(row_df)[0, 1])
            pred = int(proba >= 0.5)

            if pred == 1:
                st.markdown(
                    f"<div class='predict-card predict-risk'>⚠ AT RISK // PROBABILITY: {proba:.1%}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='predict-card predict-safe'>✓ SAFE // PROBABILITY: {(1-proba):.1%}</div>",
                    unsafe_allow_html=True,
                )

            # Approximate contributions from model importance and current values.
            contrib_rows = []
            importance_lookup = dict(zip(importance_df["feature"], importance_df["importance"]))
            for col, value in row_df.iloc[0].items():
                if isinstance(value, str):
                    feat_name = f"cat__{col}_{value}"
                    contribution = float(importance_lookup.get(feat_name, 0.0))
                else:
                    feat_name = f"num__{col}"
                    base = float(importance_lookup.get(feat_name, 0.0))
                    contribution = float(base * value)
                contrib_rows.append({"feature": col, "contribution": contribution})

            contrib_df = (
                pd.DataFrame(contrib_rows)
                .assign(contribution=lambda frame: frame["contribution"] / (frame["contribution"].abs().max() + 1e-6))
                .sort_values("contribution", key=lambda s: s.abs(), ascending=False)
            )
            st.plotly_chart(prediction_contribution_chart(contrib_df), use_container_width=True)

with history_tab:
    st.subheader("Model Timeline and Post-Mortem")

    if history_df.empty:
        st.info("No model registry entries found.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.plotly_chart(roc_auc_history(history_df), use_container_width=True)

        latest_version = history_df.iloc[-1]["version"]
        st.markdown(
            f"<p class='accent-header'>▸ LATEST PRODUCTION MODEL: {latest_version}</p>",
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────
st.markdown("<div class='pixel-divider'></div>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-family: var(--font-pixel); font-size: 0.4rem; "
    "color: var(--text-dim); letter-spacing: 1px;'>"
    "🎓 SREWS // SCHOLARSHIP RETENTION EARLY WARNING SYSTEM // VIJAYBHOOMI UNIVERSITY"
    "</p>",
    unsafe_allow_html=True,
)
