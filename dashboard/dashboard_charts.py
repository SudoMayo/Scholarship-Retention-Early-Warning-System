"""Plotly chart builders for the SREWS dashboard — Dark Retro Pixel Theme."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Retro monochrome palette ──────────────────────────────────
ACCENT     = "#ffffff"
ACCENT_DIM = "#a0a0a0"
GOOD       = "#c8c8c8"
ALERT      = "#ff4444"
GRID_COLOR = "#2a2a2a"
SURFACE    = "#111111"
BG_DARK    = "#0a0a0a"
TEXT_COLOR  = "#a0a0a0"
TEXT_BRIGHT = "#ffffff"

BASE_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=BG_DARK,
    font=dict(family="VT323, Courier New, monospace", color=TEXT_COLOR, size=16),
    margin=dict(l=48, r=24, t=64, b=48),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(color=TEXT_COLOR, size=14),
        bgcolor="rgba(0,0,0,0)",
    ),
    title_font=dict(
        family="'Press Start 2P', monospace",
        size=11,
        color=TEXT_BRIGHT,
    ),
)

# Axis defaults applied separately to avoid duplicate keyword conflicts
AXIS_DEFAULTS = dict(
    gridcolor=GRID_COLOR,
    zerolinecolor=GRID_COLOR,
    linecolor=GRID_COLOR,
    tickfont=dict(color=TEXT_COLOR),
    title_font=dict(color=TEXT_COLOR),
)


def _apply_dark_axes(fig: go.Figure) -> go.Figure:
    """Apply dark theme axis styling to a figure."""
    fig.update_xaxes(**AXIS_DEFAULTS)
    fig.update_yaxes(**AXIS_DEFAULTS)
    return fig


def risk_distribution_donut_by_department(semester_view: pd.DataFrame) -> go.Figure:
    """Create one donut per department showing at-risk vs not-at-risk shares."""
    if semester_view.empty:
        return go.Figure()

    departments = sorted(semester_view["department"].dropna().unique().tolist())
    fig = make_subplots(
        rows=1,
        cols=max(1, len(departments)),
        specs=[[{"type": "domain"}] * max(1, len(departments))],
        subplot_titles=departments,
    )

    for idx, dept in enumerate(departments, start=1):
        subset = semester_view[semester_view["department"] == dept]
        at_risk = int((subset["scholarship_at_risk"] == 1).sum())
        safe = int((subset["scholarship_at_risk"] == 0).sum())
        fig.add_trace(
            go.Pie(
                labels=["At Risk", "Not At Risk"],
                values=[at_risk, safe],
                hole=0.65,
                marker=dict(
                    colors=[ALERT, GOOD],
                    line=dict(color=SURFACE, width=2),
                ),
                textinfo="percent",
                textfont=dict(family="VT323, monospace", size=14, color=TEXT_BRIGHT),
                showlegend=(idx == 1),
            ),
            row=1,
            col=idx,
        )

    fig.update_layout(
        title="RISK DISTRIBUTION BY DEPARTMENT",
        height=360,
        **BASE_LAYOUT,
    )
    # Style subplot titles
    for annotation in fig.layout.annotations:
        annotation.font = dict(
            family="'Press Start 2P', monospace",
            size=9,
            color=TEXT_BRIGHT,
        )
    return fig


def cgpa_trend_line(semester_view: pd.DataFrame) -> go.Figure:
    """Average semester CGPA trend line across students."""
    if semester_view.empty:
        return go.Figure()

    trend = (
        semester_view.groupby("semester", as_index=False)["cgpa_this_semester"]
        .mean()
        .sort_values("semester")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["semester"],
            y=trend["cgpa_this_semester"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8, color=ACCENT, symbol="square"),
            name="AVG CGPA",
        )
    )
    fig.add_hline(
        y=6.0,
        line_dash="dash",
        line_color=ALERT,
        annotation_text="RISK THRESHOLD",
        annotation_font=dict(
            family="'Press Start 2P', monospace",
            size=8,
            color=ALERT,
        ),
    )
    fig.update_layout(
        title="CGPA TREND BY SEMESTER",
        xaxis_title="SEMESTER",
        yaxis_title="AVERAGE CGPA",
        height=360,
        **BASE_LAYOUT,
    )
    fig.update_yaxes(range=[0, 10])
    _apply_dark_axes(fig)
    return fig


def feature_correlation_heatmap(data: pd.DataFrame, features: Iterable[str]) -> go.Figure:
    """Correlation matrix heatmap for numeric features — dark theme."""
    cols = [f for f in features if f in data.columns]
    if not cols:
        return go.Figure()

    corr = data[cols].corr(numeric_only=True)

    # Custom dark color scale: deep black → dim gray → bright white
    dark_scale = [
        [0.0, "#1a0000"],
        [0.25, "#4a2020"],
        [0.5, "#2a2a2a"],
        [0.75, "#607080"],
        [1.0, "#ffffff"],
    ]

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=dark_scale,
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(
        title="FEATURE CORRELATION HEATMAP",
        height=520,
        coloraxis_colorbar=dict(
            tickfont=dict(color=TEXT_COLOR),
            title_font=dict(color=TEXT_COLOR),
        ),
        **BASE_LAYOUT,
    )
    fig.update_traces(
        textfont=dict(family="VT323, monospace", size=11, color=TEXT_COLOR),
    )
    _apply_dark_axes(fig)
    return fig


def risk_factor_breakdown(importance_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Horizontal bar chart for strongest risk-driving features."""
    if importance_df.empty:
        return go.Figure()

    top = importance_df.head(top_n).sort_values("importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker_color=ACCENT,
            marker_line=dict(color=ACCENT_DIM, width=1),
        )
    )
    fig.update_layout(
        title="RISK FACTOR BREAKDOWN (MODEL IMPORTANCE)",
        xaxis_title="IMPORTANCE",
        yaxis_title="FEATURE",
        height=420,
        **BASE_LAYOUT,
    )
    _apply_dark_axes(fig)
    return fig


def fee_payment_vs_risk(semester_view: pd.DataFrame) -> go.Figure:
    """Grouped bar chart for risk rates by fee payment behavior."""
    if semester_view.empty or "fee_payment_status" not in semester_view.columns:
        return go.Figure()

    grouped = (
        semester_view.groupby(["department", "fee_payment_status"], as_index=False)["scholarship_at_risk"]
        .mean()
    )
    grouped["risk_rate_pct"] = grouped["scholarship_at_risk"] * 100

    # Monochrome palette for departments
    mono_palette = ["#ffffff", "#cccccc", "#999999", "#777777", "#555555"]

    fig = px.bar(
        grouped,
        x="fee_payment_status",
        y="risk_rate_pct",
        color="department",
        barmode="group",
        color_discrete_sequence=mono_palette,
    )
    fig.update_layout(
        title="FEE PAYMENT STATUS VS AT-RISK RATE",
        xaxis_title="FEE PAYMENT STATUS",
        yaxis_title="AT-RISK RATE (%)",
        height=380,
        **BASE_LAYOUT,
    )
    _apply_dark_axes(fig)
    return fig


def roc_auc_history(history_df: pd.DataFrame) -> go.Figure:
    """Line chart showing ROC-AUC across model versions."""
    if history_df.empty:
        return go.Figure()

    temp = history_df.copy()
    temp["version_num"] = temp["version"].str.replace("v", "", regex=False).astype(int)

    fig = go.Figure(
        go.Scatter(
            x=temp["version_num"],
            y=temp["roc_auc"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8, color=ACCENT, symbol="square"),
        )
    )
    fig.update_layout(
        title="ROC-AUC ACROSS MODEL VERSIONS",
        xaxis_title="MODEL VERSION",
        yaxis_title="ROC-AUC",
        height=320,
        **BASE_LAYOUT,
    )
    fig.update_yaxes(range=[0, 1])
    _apply_dark_axes(fig)
    return fig


def prediction_contribution_chart(contrib_df: pd.DataFrame) -> go.Figure:
    """Small horizontal chart for per-prediction contribution approximations."""
    if contrib_df.empty:
        return go.Figure()

    top = contrib_df.head(8).sort_values("contribution", ascending=True)
    colors = [ALERT if v < 0 else ACCENT for v in top["contribution"]]

    fig = go.Figure(
        go.Bar(
            x=top["contribution"],
            y=top["feature"],
            orientation="h",
            marker_color=colors,
            marker_line=dict(color=ACCENT_DIM, width=1),
        )
    )
    fig.update_layout(
        title="PREDICTION FEATURE CONTRIBUTIONS",
        xaxis_title="RELATIVE CONTRIBUTION",
        yaxis_title="",
        height=320,
        **BASE_LAYOUT,
    )
    _apply_dark_axes(fig)
    return fig
