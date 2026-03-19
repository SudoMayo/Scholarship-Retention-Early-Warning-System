"""Evaluation helpers for binary scholarship-risk models."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute ROC-AUC with fallback when test labels contain one class."""
    unique = np.unique(y_true)
    if len(unique) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_proba))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float | List[List[int]]]:
    """Return binary classification metrics used for model selection and tracking."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "roc_auc": _safe_roc_auc(y_true, y_proba),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
    }
