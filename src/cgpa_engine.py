from typing import Dict, List
import numpy as np
import pandas as pd

GRADE_POINTS: Dict[str, int] = {
    "A+": 10,
    "A": 9,
    "B+": 8,
    "B": 7,
    "C+": 6,
    "C": 5,
    "D": 4,
    "E": 2,
    "NC": 0,
}


def compute_cgpa(grade_points: np.ndarray, credits: np.ndarray) -> float:
    total_credits = float(np.sum(credits))
    if total_credits == 0:
        return 0.0
    return float(np.sum(grade_points * credits) / total_credits)


def compute_student_cgpa(df: pd.DataFrame, grade_col: str) -> pd.DataFrame:
    results = []
    for student_id, group in df.groupby("student_id"):
        points = group[grade_col].map(GRADE_POINTS).to_numpy(dtype=float)
        credits = group["credit_value"].to_numpy(dtype=float)
        results.append(
            {
                "student_id": student_id,
                "cgpa": compute_cgpa(points, credits),
            }
        )
    return pd.DataFrame(results)


def compute_semester_cgpa(df: pd.DataFrame, grade_col: str) -> pd.DataFrame:
    """Compute CGPA per student per semester."""
    results = []
    for (student_id, semester), group in df.groupby(["student_id", "semester"]):
        points = group[grade_col].map(GRADE_POINTS).to_numpy(dtype=float)
        credits = group["credit_value"].to_numpy(dtype=float)
        results.append(
            {
                "student_id": student_id,
                "semester": semester,
                "semester_gpa": compute_cgpa(points, credits),
            }
        )
    return pd.DataFrame(results)


def compute_cgpa_trajectory(df: pd.DataFrame, grade_col: str) -> pd.DataFrame:
    """Compute cumulative CGPA trajectory across semesters for each student."""
    results = []
    for student_id, student_df in df.groupby("student_id"):
        semesters = sorted(student_df["semester"].unique())
        for i, sem in enumerate(semesters):
            # Cumulative up to this semester
            cum_data = student_df[student_df["semester"] <= sem]
            points = cum_data[grade_col].map(GRADE_POINTS).to_numpy(dtype=float)
            credits = cum_data["credit_value"].to_numpy(dtype=float)
            results.append(
                {
                    "student_id": student_id,
                    "semester": sem,
                    "cumulative_cgpa": compute_cgpa(points, credits),
                }
            )
    return pd.DataFrame(results)


def expected_cgpa_from_probs(
    prob_matrix: np.ndarray,
    class_labels: List[str],
    credits: np.ndarray,
) -> float:
    # Convert class probabilities into expected grade points.
    grade_points = np.array([GRADE_POINTS[label] for label in class_labels], dtype=float)
    expected_points = prob_matrix @ grade_points
    return compute_cgpa(expected_points, credits)


def scholarship_risk_from_cgpa(cgpa: float) -> int:
    return 1 if cgpa < 7.0 else 0


def scholarship_prob_from_cgpa(cgpa: float) -> float:
    # Smooth probability around the 7.0 threshold.
    return float(1.0 / (1.0 + np.exp((cgpa - 7.0) * 2.0)))


def risk_category(prob: float) -> str:
    if prob >= 0.6:
        return "High Risk"
    if prob >= 0.3:
        return "Moderate Risk"
    return "Low Risk"
