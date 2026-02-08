import argparse
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

GRADE_CATEGORIES = ["A+", "A", "B+", "B", "C+", "C", "D", "E", "NC"]
GRADE_POINTS = {
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


def _grade_from_score(score: float) -> str:
    if score >= 90:
        return "A+"
    if score >= 85:
        return "A"
    if score >= 78:
        return "B+"
    if score >= 70:
        return "B"
    if score >= 62:
        return "C+"
    if score >= 55:
        return "C"
    if score >= 48:
        return "D"
    if score >= 40:
        return "E"
    return "NC"


def generate_data(n_students: int, courses_per_student: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    course_ids = [f"C{100+i}" for i in range(1, 11)]
    course_difficulty = {cid: rng.uniform(0.2, 1.0) for cid in course_ids}
    course_credits = {cid: rng.choice([2, 3, 4]) for cid in course_ids}

    records = []
    for student_id in range(1, n_students + 1):
        prev_gpa = rng.uniform(5.0, 9.8)
        baseline = rng.normal(70, 10)
        selected_courses = rng.choice(course_ids, size=courses_per_student, replace=False)

        for course_id in selected_courses:
            difficulty = course_difficulty[course_id]
            credit_value = course_credits[course_id]

            attendance = np.clip(rng.normal(80, 10), 40, 100)
            midterm = np.clip(rng.normal(baseline, 12), 0, 100)
            assignment = np.clip(rng.normal(baseline + 3, 10), 0, 100)
            quiz = np.clip(rng.normal(baseline - 2, 12), 0, 100)
            prereq_grade = np.clip(rng.normal(prev_gpa * 10, 12), 0, 100)

            # Weighted score combines performance and course difficulty.
            latent_score = (
                0.35 * midterm
                + 0.2 * assignment
                + 0.15 * quiz
                + 0.2 * attendance
                + 0.1 * prereq_grade
                - 12 * difficulty
                + rng.normal(0, 3)
            )

            grade_category = _grade_from_score(latent_score)

            records.append(
                {
                    "student_id": f"S{student_id:04d}",
                    "course_id": course_id,
                    "credit_value": int(credit_value),
                    "midterm_score": float(midterm),
                    "attendance_rate": float(attendance),
                    "assignment_average": float(assignment),
                    "quiz_average": float(quiz),
                    "previous_sem_gpa": float(prev_gpa),
                    "prerequisite_grade": float(prereq_grade),
                    "course_difficulty_index": float(difficulty),
                    "grade_category": grade_category,
                    "grade_point": GRADE_POINTS[grade_category],
                }
            )

    return pd.DataFrame(records)


def write_to_sqlite(df: pd.DataFrame, db_path: Path, run_id: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        # Keep a simple schema to support appends across runs.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS academic_records (
                student_id TEXT,
                course_id TEXT,
                credit_value INTEGER,
                midterm_score REAL,
                attendance_rate REAL,
                assignment_average REAL,
                quiz_average REAL,
                previous_sem_gpa REAL,
                prerequisite_grade REAL,
                course_difficulty_index REAL,
                grade_category TEXT,
                grade_point REAL,
                run_id TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT,
                n_records INTEGER
            );
            """
        )

        df = df.copy()
        df["run_id"] = run_id
        df.to_sql("academic_records", conn, if_exists="append", index=False)
        conn.execute(
            "INSERT OR REPLACE INTO data_runs (run_id, created_at, n_records) VALUES (?, ?, ?)",
            (run_id, datetime.utcnow().isoformat(), int(len(df))),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic academic data.")
    parser.add_argument("--n-students", type=int, default=800)
    parser.add_argument("--courses-per-student", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default="data/academic_records.csv")
    parser.add_argument("--out-db", type=str, default="data/academic.db")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"run_{args.seed}_{args.n_students}_{args.courses_per_student}"

    df = generate_data(args.n_students, args.courses_per_student, args.seed)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    write_to_sqlite(df, Path(args.out_db), run_id)
    print(f"Wrote {len(df)} rows to {args.out_csv} and {args.out_db}")


if __name__ == "__main__":
    main()
