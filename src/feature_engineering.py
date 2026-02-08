from dataclasses import dataclass
from typing import List, Tuple
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    meta_test: pd.DataFrame


def load_data(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM academic_records", conn)
    return df


def build_preprocessor(categorical_features: List[str], numeric_features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )


def make_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    categorical_features: List[str],
    numeric_features: List[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitData:
    features = categorical_features + numeric_features
    X = df[features].copy()
    y = df[target_col].copy()

    meta_cols = ["student_id", "course_id", "credit_value"]
    meta_test = df[meta_cols].copy()

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta_test, test_size=test_size, random_state=seed, stratify=y
    )

    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        meta_test=meta_test.reset_index(drop=True),
    )
