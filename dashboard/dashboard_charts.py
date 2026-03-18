# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# dashboard_charts.py — Plotly figure builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

PRIMARY = "#06b6d4"
SECONDARY = "#8b5cf6"
COLORWAY = [PRIMARY, SECONDARY, "#f59e0b", "#10b981", "#ef4444", "#ec4899", "#a3e635"]
RISK_COLORS = {"High Risk": "#ff4757", "Moderate Risk": "#ffa502", "Low Risk": "#2ed573"}

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Space Grotesk, sans-serif", color="#e2e8f0", size=12),
    title_font=dict(family="Orbitron, monospace", size=15, color="#ffffff"),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        linecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color="rgba(255,255,255,0.55)", size=11),
        title_font=dict(color="rgba(255,255,255,0.7)"),
        showgrid=True, zeroline=False,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        linecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color="rgba(255,255,255,0.55)", size=11),
        title_font=dict(color="rgba(255,255,255,0.7)"),
        showgrid=True, zeroline=False,
    ),
    margin=dict(l=40, r=20, t=55, b=40),
    legend=dict(
        bgcolor="rgba(255,255,255,0.05)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        font=dict(color="rgba(255,255,255,0.75)", size=11),
    ),
    hoverlabel=dict(
        bgcolor="rgba(10,15,35,0.96)",
        bordercolor="rgba(255,255,255,0.2)",
        font=dict(family="Space Grotesk", color="white", size=12),
    ),
    colorway=COLORWAY,
)

CHART_CONFIG = {"displayModeBar": False}


def build_risk_gauge(value, title="Scholarship Risk"):
    """Build a risk gauge indicator."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"family": "Orbitron", "size": 40, "color": "white"}},
        title={"text": title, "font": {"family": "Space Grotesk", "size": 14, "color": "rgba(255,255,255,0.6)"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)",
                     "tickfont": {"color": "rgba(255,255,255,0.4)"}},
            "bar": {"color": PRIMARY, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(46,213,115,0.15)"},
                {"range": [30, 60], "color": "rgba(255,165,2,0.15)"},
                {"range": [60, 100], "color": "rgba(255,71,87,0.15)"},
            ],
            "threshold": {
                "line": {"color": SECONDARY, "width": 3},
                "thickness": 0.8,
                "value": value * 100,
            },
        },
    ))
    fig.update_layout(height=280, **BASE_LAYOUT)
    return fig


def build_cgpa_distribution(cgpa_df):
    """Build CGPA histogram with threshold line."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=cgpa_df["cgpa"], nbinsx=30,
        marker=dict(
            color=cgpa_df["cgpa"].apply(lambda x: PRIMARY if x >= 7 else "#ff4757"),
            line=dict(width=0),
        ),
        opacity=0.8,
    ))
    fig.add_vline(x=7.0, line_dash="dash", line_color="#ff4757", line_width=2,
                  annotation_text="Scholarship Threshold 7.0",
                  annotation_font_color="#ff4757")
    fig.update_layout(
        title="CGPA Distribution Across All Students",
        xaxis_title="CGPA", yaxis_title="Count",
        height=380, **BASE_LAYOUT,
    )
    return fig


def build_risk_donut(cgpa_df):
    """Build risk category donut chart."""
    from src.cgpa_engine import scholarship_prob_from_cgpa, risk_category
    cgpa_df = cgpa_df.copy()
    cgpa_df["risk"] = cgpa_df["cgpa"].apply(lambda x: risk_category(scholarship_prob_from_cgpa(x)))
    counts = cgpa_df["risk"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [RISK_COLORS.get(l, "#666") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0.3)", width=2)),
        textfont=dict(color="white", family="Space Grotesk", size=13),
        textinfo="label+percent",
        pull=[0.05 if l == "High Risk" else 0 for l in labels],
    ))
    fig.add_annotation(
        text=f"<b>{len(cgpa_df)}</b><br><span style='font-size:11px;color:rgba(255,255,255,0.5)'>Students</span>",
        x=0.5, y=0.5, font=dict(size=22, color="white", family="Orbitron"),
        showarrow=False,
    )
    fig.update_layout(
        title="Risk Category Breakdown",
        height=380, showlegend=True, **BASE_LAYOUT,
    )
    return fig


def build_radar_chart(values, categories, title="Student Profile"):
    """Build radar/spider chart for student strengths."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(6,182,212,0.2)",
        line=dict(color=PRIMARY, width=2.5),
        marker=dict(size=6, color=PRIMARY),
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(tickfont=dict(color="rgba(255,255,255,0.6)", size=11,
                                           family="Space Grotesk")),
        ),
        showlegend=False, title=title,
        height=350, **BASE_LAYOUT,
    )
    return fig


def build_correlation_heatmap(df, columns):
    """Build feature correlation heatmap."""
    corr = df[columns].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, "#0a0a1a"], [0.5, PRIMARY], [1, SECONDARY]],
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(title="Feature Correlation Matrix", height=450, **BASE_LAYOUT)
    return fig


def build_sunburst(cgpa_df_with_meta):
    """Build sunburst: Department → Risk."""
    fig = px.sunburst(
        cgpa_df_with_meta, path=["department", "risk_category"],
        color="risk_category", color_discrete_map=RISK_COLORS,
    )
    fig.update_layout(title="Department → Risk Breakdown", height=420, **BASE_LAYOUT)
    fig.update_traces(textfont=dict(family="Space Grotesk", size=12))
    return fig


def build_trajectory(trajectory_df):
    """Build CGPA trajectory line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trajectory_df["semester"], y=trajectory_df["cumulative_cgpa"],
        mode="lines+markers",
        line=dict(color=PRIMARY, width=3),
        marker=dict(size=10, color=PRIMARY, line=dict(width=2, color="white")),
        fill="tozeroy", fillcolor="rgba(6,182,212,0.1)",
        name="Cumulative CGPA",
    ))
    fig.add_hline(y=7.0, line_dash="dash", line_color="#ff4757", line_width=2,
                  annotation_text="Scholarship Threshold")
    fig.update_layout(
        title="CGPA Trajectory Over Semesters",
        xaxis_title="Semester", yaxis_title="Cumulative CGPA",
        yaxis_range=[0, 10], height=350, **BASE_LAYOUT,
    )
    return fig


def build_semester_gpa_bars(sem_gpa_df):
    """Build semester-wise GPA bar chart."""
    colors = [PRIMARY if g >= 7 else "#ff4757" for g in sem_gpa_df["semester_gpa"]]
    fig = go.Figure(go.Bar(
        x=sem_gpa_df["semester"], y=sem_gpa_df["semester_gpa"],
        marker_color=colors,
        text=[f"{g:.2f}" for g in sem_gpa_df["semester_gpa"]],
        textposition="outside",
        textfont=dict(color="white", family="Orbitron", size=13),
    ))
    fig.add_hline(y=7.0, line_dash="dash", line_color="#ff4757", line_width=2)
    fig.update_layout(
        title="Semester-wise GPA", xaxis_title="Semester", yaxis_title="GPA",
        yaxis_range=[0, 10], height=350, **BASE_LAYOUT,
    )
    return fig


def build_dept_risk_bars(dept_risk_df):
    """Build department at-risk percentage bars."""
    fig = go.Figure(go.Bar(
        x=dept_risk_df["department"], y=dept_risk_df["at_risk_pct"],
        marker=dict(
            color=dept_risk_df["at_risk_pct"],
            colorscale=[[0, "#2ed573"], [0.5, "#ffa502"], [1, "#ff4757"]],
        ),
        text=[f"{v:.1f}%" for v in dept_risk_df["at_risk_pct"]],
        textposition="outside",
        textfont=dict(color="white", family="Space Grotesk", size=12),
    ))
    fig.update_layout(
        title="At-Risk % by Department", xaxis_title="Department",
        yaxis_title="At-Risk %", height=350, coloraxis_showscale=False,
        **BASE_LAYOUT,
    )
    return fig


def build_grade_distribution_area(data):
    """Build grade distribution stacked area chart."""
    grade_by_sem = data.groupby(["semester", "grade_category"]).size().reset_index(name="count")
    totals = grade_by_sem.groupby("semester")["count"].transform("sum")
    grade_by_sem["pct"] = grade_by_sem["count"] / totals * 100

    from src.cgpa_engine import GRADE_POINTS
    fig = px.area(
        grade_by_sem, x="semester", y="pct", color="grade_category",
        category_orders={"grade_category": list(GRADE_POINTS.keys())},
        color_discrete_sequence=COLORWAY,
    )
    fig.update_layout(
        title="Grade Distribution Shift Over Semesters",
        xaxis_title="Semester", yaxis_title="Percentage %",
        height=380, **BASE_LAYOUT,
    )
    return fig


def build_model_metrics_radar(metrics, risk_metrics):
    """Build model performance radar chart."""
    cats = ["Accuracy", "Macro F1", "Risk Recall", "Risk F1", "Risk AUC"]
    vals = [
        metrics["accuracy"], metrics["macro_f1"],
        risk_metrics.get("recall", 0), risk_metrics.get("f1", 0),
        risk_metrics.get("roc_auc", 0),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(139,92,246,0.2)",
        line=dict(color=SECONDARY, width=2.5),
        marker=dict(size=8, color=SECONDARY),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=True,
                            tickfont=dict(color="rgba(255,255,255,0.4)", size=9),
                            gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(tickfont=dict(color="rgba(255,255,255,0.7)", size=12,
                                           family="Space Grotesk")),
        ),
        showlegend=False, title="Model Performance Radar",
        height=400, **BASE_LAYOUT,
    )
    return fig


def build_scenario_comparison(results_dict):
    """Build scenario comparison bar chart."""
    labels = list(results_dict.keys())
    cgpas = [r["expected_cgpa"] for r in results_dict.values()]
    colors = [PRIMARY if i == 0 else COLORWAY[i % len(COLORWAY)] for i in range(len(labels))]

    fig = go.Figure()
    for i, (label, cgpa) in enumerate(zip(labels, cgpas)):
        fig.add_trace(go.Bar(
            x=[label], y=[cgpa], name=label,
            marker_color=colors[i],
            text=[f"{cgpa:.2f}"], textposition="outside",
            textfont=dict(color="white", family="Orbitron", size=14),
        ))
    fig.add_hline(y=7.0, line_dash="dash", line_color="#ff4757", line_width=2,
                  annotation_text="Scholarship Threshold (7.0)")
    fig.update_layout(
        title="CGPA: Scenario Comparison",
        yaxis_title="Expected CGPA", yaxis_range=[0, 10],
        showlegend=False, height=400, **BASE_LAYOUT,
    )
    return fig
