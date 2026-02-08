from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from src.cgpa_engine import compute_cgpa, expected_cgpa_from_probs, scholarship_prob_from_cgpa


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def cgpa_rmse(
    meta_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    df = meta_test.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    actual = []
    projected = []
    # Aggregate predictions at student level to compute CGPA error.
    for student_id, group in df.groupby("student_id"):
        credits = group["credit_value"].to_numpy(dtype=float)
        actual_points = group["y_true"].to_numpy(dtype=float)
        projected_points = group["y_pred"].to_numpy(dtype=float)
        actual.append(compute_cgpa(actual_points, credits))
        projected.append(compute_cgpa(projected_points, credits))

    actual = np.array(actual, dtype=float)
    projected = np.array(projected, dtype=float)
    rmse = float(np.sqrt(np.mean((actual - projected) ** 2)))
    return rmse


def scholarship_metrics(
    meta_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    class_labels: List[str],
) -> Dict[str, float]:
    df = meta_test.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    actual_risks = []
    predicted_risks = []
    risk_probs = []

    # Compute risk labels and probabilities per student.
    for student_id, group in df.groupby("student_id"):
        credits = group["credit_value"].to_numpy(dtype=float)
        actual_points = group["y_true"].to_numpy(dtype=float)
        predicted_points = group["y_pred"].to_numpy(dtype=float)

        actual_cgpa = compute_cgpa(actual_points, credits)
        projected_cgpa = compute_cgpa(predicted_points, credits)

        actual_risks.append(1 if actual_cgpa < 7.0 else 0)
        predicted_risks.append(1 if projected_cgpa < 7.0 else 0)

        proba_matrix = proba[group.index]
        expected_cgpa = expected_cgpa_from_probs(proba_matrix, class_labels, credits)
        risk_probs.append(scholarship_prob_from_cgpa(expected_cgpa))

    actual_risks = np.array(actual_risks)
    predicted_risks = np.array(predicted_risks)
    risk_probs = np.array(risk_probs)

    metrics = {
        "recall": float(recall_score(actual_risks, predicted_risks, zero_division=0)),
        "f1": float(f1_score(actual_risks, predicted_risks, zero_division=0)),
    }

    if len(np.unique(actual_risks)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(actual_risks, risk_probs))
    else:
        metrics["roc_auc"] = 0.0

    return metrics
