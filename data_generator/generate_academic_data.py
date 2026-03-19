"""Generate synthetic university records for scholarship-risk modeling.

This script creates reproducible, semester-wise academic records and stores them in
both SQLite and CSV. The ML target is `scholarship_at_risk` with rule-based label
engineering at student-semester level.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

GRADE_CATEGORIES = ["A+", "A", "B+", "B", "C+", "C", "D", "E", "NC"]
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

DEPARTMENTS = ["CS", "ECE", "ME", "CE", "EE"]
SCHOLARSHIP_TIERS = ["Merit-100%", "Merit-75%", "Merit-50%", "Need-Based"]
INCOME_BRACKETS = ["low", "lower_mid", "mid", "upper_mid"]
FEE_PAYMENT_STATUS = ["on_time", "late", "defaulted"]


def _str_to_bool(value: str | bool) -> bool:
    """Parse common boolean CLI values safely."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def _grade_from_score(score: float) -> str:
    """Map a latent course score to a university grade bucket."""
    if score >= 90:
        return "A+"
    if score >= 84:
        return "A"
    if score >= 76:
        return "B+"
    if score >= 68:
        return "B"
    if score >= 60:
        return "C+"
    if score >= 52:
        return "C"
    if score >= 45:
        return "D"
    if score >= 38:
        return "E"
    return "NC"


@dataclass(frozen=True)
class GeneratorConfig:
    """Container for generator settings to keep function signatures tidy."""

    n_students: int
    courses_per_student: int
    n_semesters: int
    seed: int
    drift_mode: bool


def _tier_probs_for_income(income: str) -> List[float]:
    """Return scholarship tier probabilities conditioned on family income."""
    if income == "low":
        return [0.05, 0.15, 0.25, 0.55]
    if income == "lower_mid":
        return [0.12, 0.24, 0.28, 0.36]
    if income == "mid":
        return [0.24, 0.32, 0.28, 0.16]
    return [0.35, 0.32, 0.22, 0.11]


def _fee_status_probs(income: str, tier: str) -> List[float]:
    """Return payment status probabilities driven by financial pressure."""
    base = {
        "low": np.array([0.56, 0.29, 0.15]),
        "lower_mid": np.array([0.65, 0.25, 0.10]),
        "mid": np.array([0.76, 0.18, 0.06]),
        "upper_mid": np.array([0.85, 0.12, 0.03]),
    }[income].astype(float)

    if tier == "Need-Based":
        base += np.array([-0.05, 0.02, 0.03])
    if tier == "Merit-100%":
        base += np.array([0.03, -0.02, -0.01])

    base = np.clip(base, 0.01, None)
    base = base / base.sum()
    return base.tolist()


def _ability_baseline(tier: str, dept: str, rng: np.random.Generator) -> float:
    """Create student-level academic baseline with tier and department effects."""
    tier_offset = {
        "Merit-100%": 10.0,
        "Merit-75%": 6.0,
        "Merit-50%": 2.0,
        "Need-Based": -1.5,
    }[tier]
    dept_offset = {"CS": 1.0, "ECE": 0.5, "ME": -0.4, "CE": -0.8, "EE": 0.0}[dept]
    return float(np.clip(rng.normal(66 + tier_offset + dept_offset, 8.5), 40, 95))


def _compute_cgpa_from_rows(rows: Iterable[dict]) -> float:
    """Compute weighted semester CGPA from course rows."""
    rows_list = list(rows)
    total_credits = float(sum(r["credit_value"] for r in rows_list))
    if total_credits <= 0:
        return 0.0
    weighted_points = float(sum(r["grade_point"] * r["credit_value"] for r in rows_list))
    return weighted_points / total_credits


def generate_data(config: GeneratorConfig) -> pd.DataFrame:
    """Generate synthetic records with realistic multi-factor scholarship risk signals.

    The label `scholarship_at_risk` is engineered at student-semester level and is 1
    when one of the following holds:
    1) semester CGPA < 6.0
    2) average attendance < 65%
    3) failed courses (E or NC) > 1
    """
    rng = np.random.default_rng(config.seed)

    course_pool = [f"C{100 + i}" for i in range(1, 25)]
    course_difficulty = {cid: float(rng.uniform(0.25, 0.95)) for cid in course_pool}
    course_credits = {cid: int(rng.choice([2, 3, 4], p=[0.2, 0.6, 0.2])) for cid in course_pool}

    records: List[dict] = []

    for sid_num in range(1, config.n_students + 1):
        student_id = f"S{sid_num:05d}"
        department = str(rng.choice(DEPARTMENTS))
        year = int(rng.choice([1, 2, 3, 4], p=[0.28, 0.27, 0.24, 0.21]))

        income = str(rng.choice(INCOME_BRACKETS, p=[0.24, 0.31, 0.30, 0.15]))
        tier = str(rng.choice(SCHOLARSHIP_TIERS, p=_tier_probs_for_income(income)))

        hostel_resident = bool(rng.random() < 0.58)
        baseline_ability = _ability_baseline(tier, department, rng)
        base_prev_gpa = float(np.clip((baseline_ability / 10.0) + rng.normal(0, 0.55), 4.8, 9.6))

        for semester in range(1, config.n_semesters + 1):
            # Semester-level latent wellbeing and discipline factors.
            mental_health = float(
                np.clip(
                    rng.normal(6.6 + (baseline_ability - 66) * 0.04 - 0.14 * semester, 1.2),
                    1.0,
                    10.0,
                )
            )

            fee_status = str(rng.choice(FEE_PAYMENT_STATUS, p=_fee_status_probs(income, tier)))
            counselling_sessions = int(
                np.clip(
                    rng.poisson(1.2 + (5.5 - mental_health) * 0.18),
                    0,
                    5,
                )
            )

            # Drift mode simulates a stressed intake: more extracurricular burden,
            # less study hours.
            drift_extra = 1.5 if config.drift_mode else 0.0
            drift_study = -1.5 if config.drift_mode else 0.0

            study_hours = float(
                np.clip(
                    6.0
                    + 0.75 * mental_health
                    + rng.normal(0, 2.3)
                    - (0.85 if hostel_resident else 0.0)
                    + drift_study,
                    3.0,
                    35.0,
                )
            )

            library_usage = float(
                np.clip(
                    2.0
                    + 0.42 * study_hours
                    + (1.2 if hostel_resident else 0.3)
                    + rng.normal(0, 1.0),
                    0.5,
                    30.0,
                )
            )

            extracurricular_load = float(
                np.clip(
                    rng.normal(3.0 + drift_extra + (year - 1) * 0.2, 1.4),
                    0.0,
                    10.0,
                )
            )

            previous_sem_gpa = float(np.clip(base_prev_gpa + rng.normal(0, 0.30), 4.0, 10.0))

            # Fee stress explicitly worsens attendance/performance so defaulted students
            # have substantially higher risk without overriding the target definition.
            fee_attendance_penalty = {"on_time": 0.0, "late": 3.0, "defaulted": 8.5}[fee_status]
            fee_score_penalty = {"on_time": 0.0, "late": 3.0, "defaulted": 7.5}[fee_status]

            selected_courses = rng.choice(
                course_pool,
                size=min(config.courses_per_student, len(course_pool)),
                replace=False,
            )

            semester_rows: List[dict] = []

            for course_id in selected_courses:
                difficulty = float(course_difficulty[course_id])
                credit_value = int(course_credits[course_id])

                attendance = float(
                    np.clip(
                        59
                        + 2.8 * mental_health
                        + rng.normal(0, 8.5)
                        - fee_attendance_penalty,
                        35,
                        100,
                    )
                )

                prerequisite_grade = float(
                    np.clip(
                        rng.normal(previous_sem_gpa * 10 + (baseline_ability - 66) * 0.35, 9.5),
                        30,
                        98,
                    )
                )

                midterm = float(
                    np.clip(
                        baseline_ability
                        + 0.40 * study_hours
                        + 0.11 * attendance
                        - 10.5 * difficulty
                        - 0.85 * extracurricular_load
                        - fee_score_penalty
                        + 0.11 * prerequisite_grade
                        + 0.6 * counselling_sessions
                        - 3.0
                        + rng.normal(0, 8.0),
                        0,
                        100,
                    )
                )

                assignment = float(np.clip(midterm + rng.normal(2.0, 6.5), 0, 100))
                quiz = float(np.clip(midterm + rng.normal(-1.0, 8.0), 0, 100))

                composite_score = (
                    0.38 * midterm
                    + 0.22 * assignment
                    + 0.12 * quiz
                    + 0.18 * attendance
                    + 0.10 * prerequisite_grade
                    - 5.0 * difficulty
                    + rng.normal(0, 2.5)
                )
                grade_category = _grade_from_score(composite_score)

                semester_rows.append(
                    {
                        "student_id": student_id,
                        "semester": int(semester),
                        "department": department,
                        "year": int(year),
                        "scholarship_tier": tier,
                        "family_income_bracket": income,
                        "fee_payment_status": fee_status,
                        "hostel_resident": int(hostel_resident),
                        "counselling_sessions_attended": counselling_sessions,
                        "mental_health_score": round(mental_health, 2),
                        "library_usage_hours_per_week": round(library_usage, 2),
                        "course_id": str(course_id),
                        "credit_value": int(credit_value),
                        "midterm_score": round(midterm, 2),
                        "attendance_rate": round(attendance, 2),
                        "assignment_average": round(assignment, 2),
                        "quiz_average": round(quiz, 2),
                        "study_hours_per_week": round(study_hours, 2),
                        "extracurricular_load": round(extracurricular_load, 2),
                        "previous_sem_gpa": round(previous_sem_gpa, 3),
                        "prerequisite_grade": round(prerequisite_grade, 2),
                        "course_difficulty_index": round(difficulty, 3),
                        "grade_category": grade_category,
                        "grade_point": int(GRADE_POINTS[grade_category]),
                    }
                )

            sem_cgpa = _compute_cgpa_from_rows(semester_rows)
            avg_attendance = float(np.mean([r["attendance_rate"] for r in semester_rows]))
            failed_count = int(sum(1 for r in semester_rows if r["grade_category"] in {"E", "NC"}))

            scholarship_at_risk = int(
                (sem_cgpa < 6.0) or (avg_attendance < 65.0) or (failed_count > 1)
            )

            for row in semester_rows:
                row["cgpa_this_semester"] = round(sem_cgpa, 3)
                row["cgpa_trend"] = round(sem_cgpa - row["previous_sem_gpa"], 3)
                row["failed_courses_this_semester"] = failed_count
                row["scholarship_at_risk"] = scholarship_at_risk
                records.append(row)

            # Update semester carry-forward state.
            base_prev_gpa = float(np.clip(sem_cgpa, 0.0, 10.0))

    df = pd.DataFrame(records)

    return df


def write_to_sqlite(df: pd.DataFrame, db_path: Path, run_id: str, drift_mode: bool) -> None:
    """Persist records and run metadata to SQLite.

    In normal mode tables are reset to create a clean baseline dataset.
    In drift mode, rows are appended so retraining triggers can detect new runs.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        if not drift_mode:
            conn.execute("DROP TABLE IF EXISTS academic_records")
            conn.execute("DROP TABLE IF EXISTS data_runs")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS academic_records (
                student_id TEXT,
                semester INTEGER,
                department TEXT,
                year INTEGER,
                scholarship_tier TEXT,
                family_income_bracket TEXT,
                fee_payment_status TEXT,
                hostel_resident INTEGER,
                counselling_sessions_attended INTEGER,
                mental_health_score REAL,
                library_usage_hours_per_week REAL,
                course_id TEXT,
                credit_value INTEGER,
                midterm_score REAL,
                attendance_rate REAL,
                assignment_average REAL,
                quiz_average REAL,
                study_hours_per_week REAL,
                extracurricular_load REAL,
                previous_sem_gpa REAL,
                prerequisite_grade REAL,
                course_difficulty_index REAL,
                grade_category TEXT,
                grade_point INTEGER,
                cgpa_this_semester REAL,
                cgpa_trend REAL,
                failed_courses_this_semester INTEGER,
                scholarship_at_risk INTEGER,
                run_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT,
                n_records INTEGER
            )
            """
        )

        payload = df.copy()
        payload["run_id"] = run_id
        payload.to_sql("academic_records", conn, if_exists="append", index=False)
        conn.execute(
            "INSERT OR REPLACE INTO data_runs (run_id, created_at, n_records) VALUES (?, ?, ?)",
            (run_id, datetime.now(UTC).isoformat(), int(len(payload))),
        )


def summarize_generation(df: pd.DataFrame) -> str:
    """Build a short generation summary for CLI logs."""
    sem_view = (
        df.groupby(["student_id", "semester"], as_index=False)["scholarship_at_risk"].max()
        if not df.empty
        else pd.DataFrame(columns=["scholarship_at_risk"])
    )
    risk_rate = float(sem_view["scholarship_at_risk"].mean() * 100) if len(sem_view) else 0.0
    return (
        f"Rows={len(df):,} | Students={df['student_id'].nunique():,} | "
        f"Semesters={df['semester'].nunique():,} | "
        f"At-risk (student-semester)={risk_rate:.1f}%"
    )


def main() -> None:
    """CLI entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic academic data for SREWS")
    parser.add_argument("--n-students", type=int, default=800)
    parser.add_argument("--courses-per-student", type=int, default=6)
    parser.add_argument("--n-semesters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default="data/academic_records.csv")
    parser.add_argument("--out-db", type=str, default="data/academic.db")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--drift-mode", type=_str_to_bool, default=False)
    args = parser.parse_args()

    run_id = args.run_id or f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    config = GeneratorConfig(
        n_students=args.n_students,
        courses_per_student=args.courses_per_student,
        n_semesters=args.n_semesters,
        seed=args.seed,
        drift_mode=bool(args.drift_mode),
    )

    df = generate_data(config)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    write_to_sqlite(df, Path(args.out_db), run_id=run_id, drift_mode=config.drift_mode)

    print("Synthetic data generation complete.")
    print(f"Run ID: {run_id}")
    print(summarize_generation(df))
    print(f"CSV path: {out_csv}")
    print(f"SQLite path: {Path(args.out_db)}")


if __name__ == "__main__":
    main()
