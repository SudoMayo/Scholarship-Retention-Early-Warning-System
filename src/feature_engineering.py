"""Feature engineering utilities for scholarship risk classification.

This module centralizes:
1) loading records from SQLite,
2) target/feature schema selection,
3) train-test splitting,
4) sklearn preprocessing with missing-value handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import sqlite3

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "scholarship_at_risk"

CATEGORICAL_FEATURES: List[str] = [
    "department",
    "scholarship_tier",
    "family_income_bracket",
    "fee_payment_status",
    "grade_category",
]

NUMERIC_FEATURES: List[str] = [
    "year",
    "midterm_score",
    "attendance_rate",
    "assignment_average",
    "quiz_average",
    "study_hours_per_week",
    "extracurricular_load",
    "previous_sem_gpa",
    "counselling_sessions_attended",
    "library_usage_hours_per_week",
    "hostel_resident",
    "mental_health_score",
    "cgpa_this_semester",
    "cgpa_trend",
]


@dataclass
class SplitData:
    """Bundle train-test partitions and held-out metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    meta_test: pd.DataFrame


def load_data(db_path: str) -> pd.DataFrame:
    """Load full academic_records table and enforce required columns.

    The loader is defensive so retraining can run without manual data cleaning.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM academic_records", conn)

    if df.empty:
        return df

    if "cgpa_trend" not in df.columns and "cgpa_this_semester" in df.columns:
        df["cgpa_trend"] = df["cgpa_this_semester"] - df.get("previous_sem_gpa", 0)

    # Ensure binary integer target values.
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0).astype(int)

    # Ensure hostel_resident is numeric 0/1.
    if "hostel_resident" in df.columns:
        df["hostel_resident"] = (
            df["hostel_resident"]
            .replace({True: 1, False: 0, "True": 1, "False": 0})
            .fillna(0)
            .astype(int)
        )

    return df


def build_preprocessor(
    categorical_features: List[str],
    numeric_features: List[str],
) -> ColumnTransformer:
    """Create preprocessing graph with imputation + encoding.

    Missing values are handled in-pipeline to keep production scoring robust.
    """
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_features),
            ("num", num_pipeline, numeric_features),
        ]
    )


def make_train_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    categorical_features: List[str] | None = None,
    numeric_features: List[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitData:
    """Split features and target into stratified train-test subsets."""
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric_features = numeric_features or NUMERIC_FEATURES

    required_cols = categorical_features + numeric_features + [target_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input data: {missing}")

    feature_cols = categorical_features + numeric_features
    X = df[feature_cols].copy()
    y = df[target_col].copy().astype(int)

    meta_cols = [col for col in ["student_id", "semester", "run_id"] if col in df.columns]
    meta_df = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    X_train, X_test, y_train, y_test, _, meta_test = train_test_split(
        X,
        y,
        meta_df,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        meta_test=meta_test.reset_index(drop=True),
    )
