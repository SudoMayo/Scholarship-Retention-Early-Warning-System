import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.cgpa_engine import (
    expected_cgpa_from_probs,
    scholarship_prob_from_cgpa,
    risk_category,
)


def load_latest_model(models_dir: Path):
    registry_path = models_dir / "model_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("Model registry not found. Train a model first.")

    import json

    registry = json.loads(registry_path.read_text())
    latest = sorted(registry, key=lambda r: r["version"], reverse=True)[0]
    artifact = joblib.load(latest["model_path"])
    return artifact, latest


def predict_for_courses(df: pd.DataFrame, artifact: dict) -> dict:
    model = artifact["model"]
    class_labels = artifact["class_labels"]

    proba = model.predict_proba(df)
    pred_idx = proba.argmax(axis=1)
    pred_labels = [class_labels[i] for i in pred_idx]

    credits = df["credit_value"].to_numpy(dtype=float)
    expected_cgpa = expected_cgpa_from_probs(proba, class_labels, credits)
    risk_prob = scholarship_prob_from_cgpa(expected_cgpa)

    return {
        "predicted_grades": pred_labels,
        "expected_cgpa": expected_cgpa,
        "risk_probability": risk_prob,
        "risk_category": risk_category(risk_prob),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict grades for new inputs.")
    parser.add_argument("--input-csv", type=str, required=True)
    args = parser.parse_args()

    artifact, meta = load_latest_model(Path("models"))
    df = pd.read_csv(args.input_csv)
    output = predict_for_courses(df, artifact)

    print("Predicted grades:")
    print(output["predicted_grades"])
    print(f"Expected CGPA: {output['expected_cgpa']:.2f}")
    print(f"Risk probability: {output['risk_probability']:.2%}")
    print(f"Risk category: {output['risk_category']}")


if __name__ == "__main__":
    main()
