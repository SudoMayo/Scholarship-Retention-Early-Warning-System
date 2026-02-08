import argparse
import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.feature_engineering import load_data, build_preprocessor, make_train_test_split
from src.evaluate import classification_metrics, cgpa_rmse, scholarship_metrics
from src.cgpa_engine import GRADE_POINTS


def _next_model_version(models_dir: Path) -> int:
    existing = sorted(models_dir.glob("grade_model_v*.pkl"))
    if not existing:
        return 1
    last = existing[-1].stem.replace("grade_model_v", "")
    return int(last) + 1


def train_and_evaluate(db_path: str, models_dir: Path, seed: int) -> None:
    df = load_data(db_path)
    categorical_features = ["course_id"]
    numeric_features = [
        "credit_value",
        "midterm_score",
        "attendance_rate",
        "assignment_average",
        "quiz_average",
        "previous_sem_gpa",
        "prerequisite_grade",
        "course_difficulty_index",
    ]

    split = make_train_test_split(
        df,
        target_col="grade_category",
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        seed=seed,
    )

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(split.y_train)
    y_test_enc = label_encoder.transform(split.y_test)

    preprocessor = build_preprocessor(categorical_features, numeric_features)
    class_labels = list(label_encoder.classes_)

    # Train multiple traditional models and select by macro F1.
    models = {
        "logistic_regression": LogisticRegression(max_iter=300, multi_class="multinomial"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=len(class_labels),
            eval_metric="mlogloss",
            random_state=seed,
        ),
    }

    results = []
    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(split.X_train, y_train_enc)
        y_pred = pipeline.predict(split.X_test)
        proba = pipeline.predict_proba(split.X_test)

        metrics = classification_metrics(y_test_enc, y_pred)
        rmse = cgpa_rmse(split.meta_test, _to_points(label_encoder, y_test_enc), _to_points(label_encoder, y_pred))
        risk_metrics = scholarship_metrics(
            split.meta_test,
            _to_points(label_encoder, y_test_enc),
            _to_points(label_encoder, y_pred),
            proba,
            class_labels,
        )

        results.append(
            {
                "name": name,
                "pipeline": pipeline,
                "metrics": metrics,
                "cgpa_rmse": rmse,
                "risk_metrics": risk_metrics,
            }
        )

    best = sorted(results, key=lambda r: r["metrics"]["macro_f1"], reverse=True)[0]

    version = _next_model_version(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"grade_model_v{version}.pkl"

    # Bundle model and metadata for prediction and dashboard use.
    artifact = {
        "model": best["pipeline"],
        "label_encoder": label_encoder,
        "class_labels": class_labels,
        "grade_points": GRADE_POINTS,
    }
    joblib.dump(artifact, model_path)

    registry_path = models_dir / "model_registry.json"
    registry = []
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())

    registry.append(
        {
            "version": version,
            "model_path": str(model_path),
            "trained_at": datetime.utcnow().isoformat(),
            "selection_metric": "macro_f1",
            "model_name": best["name"],
            "metrics": best["metrics"],
            "cgpa_rmse": best["cgpa_rmse"],
            "risk_metrics": best["risk_metrics"],
        }
    )
    registry_path.write_text(json.dumps(registry, indent=2))

    print(f"Saved model v{version} to {model_path}")
    print("Metrics:")
    print(json.dumps(best["metrics"], indent=2))


def _to_points(label_encoder: LabelEncoder, y_encoded: np.ndarray) -> np.ndarray:
    labels = label_encoder.inverse_transform(y_encoded)
    return np.array([GRADE_POINTS[label] for label in labels], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train grade prediction models.")
    parser.add_argument("--db-path", type=str, default="data/academic.db")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_and_evaluate(args.db_path, Path("models"), args.seed)


if __name__ == "__main__":
    main()
