"""Train and register binary scholarship-risk models.

The script trains multiple traditional ML models and selects the best one by
ROC-AUC on a stratified hold-out split.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import classification_metrics
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    load_data,
    make_train_test_split,
)


def _next_model_version(models_dir: Path) -> int:
    """Compute the next version number from saved model artifacts."""
    existing = sorted(models_dir.glob("grade_model_v*.pkl"))
    if not existing:
        return 1
    latest = max(int(path.stem.replace("grade_model_v", "")) for path in existing)
    return latest + 1


def _extract_feature_importance(pipeline: Pipeline) -> Dict[str, float]:
    """Extract a feature-importance mapping from the fitted pipeline model.

    Supports tree feature importances and linear-model coefficients.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out().tolist()

    if hasattr(model, "feature_importances_"):
        raw_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_values = np.abs(model.coef_[0])
    else:
        return {}

    scores = {name: float(value) for name, value in zip(feature_names, raw_values)}
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_scores


def _load_registry(registry_path: Path) -> List[Dict[str, Any]]:
    """Load the existing model registry if present."""
    if not registry_path.exists():
        return []
    return json.loads(registry_path.read_text(encoding="utf-8"))


def train_and_evaluate(db_path: str, models_dir: Path, seed: int = 42) -> Dict[str, Any]:
    """Train candidate models, select best by ROC-AUC, and persist artifacts."""
    df = load_data(db_path)
    if df.empty:
        raise ValueError("No records found in academic_records. Generate data first.")

    split = make_train_test_split(
        df,
        target_col=TARGET_COLUMN,
        categorical_features=CATEGORICAL_FEATURES,
        numeric_features=NUMERIC_FEATURES,
        seed=seed,
    )

    preprocessor = build_preprocessor(CATEGORICAL_FEATURES, NUMERIC_FEATURES)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=500,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=seed,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
        ),
    }

    results: List[Dict[str, Any]] = []

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(split.X_train, split.y_train)

        y_pred = pipeline.predict(split.X_test)
        y_proba = pipeline.predict_proba(split.X_test)[:, 1]

        metrics = classification_metrics(split.y_test.to_numpy(), y_pred, y_proba)
        results.append(
            {
                "model_name": model_name,
                "pipeline": pipeline,
                "metrics": metrics,
            }
        )

    best = max(results, key=lambda item: item["metrics"]["roc_auc"])

    models_dir.mkdir(parents=True, exist_ok=True)
    version = _next_model_version(models_dir)
    model_path = models_dir / f"grade_model_v{version}.pkl"

    artifact = {
        "model": best["pipeline"],
        "target_column": TARGET_COLUMN,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "feature_importance": _extract_feature_importance(best["pipeline"]),
        "positive_label": 1,
    }
    joblib.dump(artifact, model_path)

    registry_path = models_dir / "model_registry.json"
    registry = _load_registry(registry_path)

    registry_entry = {
        "version": version,
        "model_path": str(model_path).replace("/", "\\"),
        "trained_at": datetime.now(UTC).isoformat(),
        "n_training_records": int(len(df)),
        "selection_metric": "roc_auc",
        "selected_model": best["model_name"],
        "metrics": {
            "roc_auc": float(best["metrics"]["roc_auc"]),
            "precision": float(best["metrics"]["precision"]),
            "recall": float(best["metrics"]["recall"]),
            "f1": float(best["metrics"]["f1"]),
            "confusion_matrix": best["metrics"]["confusion_matrix"],
        },
    }
    registry.append(registry_entry)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"Saved model v{version} to {model_path}")
    print(json.dumps(registry_entry, indent=2))

    return registry_entry


def main() -> None:
    """CLI entry point for training binary scholarship-risk models."""
    parser = argparse.ArgumentParser(description="Train scholarship-risk models")
    parser.add_argument("--db-path", type=str, default="data/academic.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    train_and_evaluate(
        db_path=args.db_path,
        models_dir=Path(args.models_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
