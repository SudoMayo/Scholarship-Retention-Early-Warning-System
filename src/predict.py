"""Prediction helpers for binary scholarship-risk models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def load_latest_model(models_dir: Path) -> tuple[dict, dict]:
    """Load latest artifact metadata and model bundle from registry."""
    registry_path = models_dir / "model_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("Model registry not found. Train a model first.")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not registry:
        raise ValueError("Model registry is empty. Train a model first.")

    latest = max(registry, key=lambda item: int(item.get("version", 0)))
    model_path = Path(str(latest["model_path"]).replace("\\", "/"))
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    artifact = joblib.load(model_path)
    return artifact, latest


def predict_risk(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Predict scholarship risk probabilities for input rows."""
    pipeline = artifact["model"]
    prob = pipeline.predict_proba(df)[:, 1]
    pred = (prob >= 0.5).astype(int)

    output = df.copy()
    output["risk_probability"] = prob
    output["scholarship_at_risk_pred"] = pred
    return output


def main() -> None:
    """CLI entry point for scoring CSV files with latest model."""
    parser = argparse.ArgumentParser(description="Predict scholarship risk for new records")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    artifact, meta = load_latest_model(Path(args.models_dir))
    input_df = pd.read_csv(args.input_csv)
    scored_df = predict_risk(input_df, artifact)

    out_csv = args.out_csv
    if out_csv:
        scored_df.to_csv(out_csv, index=False)
        print(f"Scored predictions written to {out_csv}")

    print(f"Model version: v{meta.get('version')}")
    print(scored_df[["risk_probability", "scholarship_at_risk_pred"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
