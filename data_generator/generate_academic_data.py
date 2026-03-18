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

DEPARTMENTS = ["CS", "ECE", "ME", "CE", "EE"]
SCHOLARSHIP_TIERS = ["Merit-100%", "Merit-75%", "Merit-50%", "Need-Based"]


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


def generate_data(
    n_students: int,
    courses_per_student: int,
    n_semesters: int,
    seed: int,
) -> pd.DataFrame:
    """Generate multi-semester academic records with student metadata."""
    rng = np.random.default_rng(seed)

    course_pool = [f"C{100+i}" for i in range(1, 21)]  # 20 courses
    course_difficulty = {cid: rng.uniform(0.2, 1.0) for cid in course_pool}
    course_credits = {cid: int(rng.choice([2, 3, 4])) for cid in course_pool}

    # Pre-assign stable student metadata
    student_departments = {
        f"S{sid:04d}": rng.choice(DEPARTMENTS) for sid in range(1, n_students + 1)
    }
    student_tiers = {
        f"S{sid:04d}": rng.choice(SCHOLARSHIP_TIERS) for sid in range(1, n_students + 1)
    }
    student_years = {
        f"S{sid:04d}": int(rng.choice([1, 2, 3, 4])) for sid in range(1, n_students + 1)
    }

    records = []
    for student_id_num in range(1, n_students + 1):
        sid = f"S{student_id_num:04d}"
        department = student_departments[sid]
        tier = student_tiers[sid]
        year = student_years[sid]

        # Each student has a baseline ability that evolves.
        baseline = rng.normal(70, 10)
        prev_gpa = rng.uniform(5.0, 9.8)

        # Base study hours and extracurricular load (stable across semesters)
        base_study_hours = np.clip(rng.normal(15, 5), 3, 35)
        base_extra_load = np.clip(rng.normal(3, 2), 0, 10)

        for semester in range(1, n_semesters + 1):
            # GPA momentum: performance in later semesters depends on earlier ones.
            if semester > 1:
                gpa_delta = rng.normal(0, 0.4)
                # Slight mean-reversion toward 7.0
                gpa_delta += (7.0 - prev_gpa) * 0.05
                prev_gpa = np.clip(prev_gpa + gpa_delta, 2.0, 10.0)
                baseline = np.clip(baseline + rng.normal(0, 3), 30, 95)

            # Study hours and extracurricular evolve slightly each semester
            study_hours = np.clip(
                base_study_hours + rng.normal(0, 2) + (semester - 1) * 0.5, 3, 35
            )
            extra_load = np.clip(
                base_extra_load + rng.normal(0, 0.5), 0, 10
            )

            selected_courses = rng.choice(
                course_pool, size=min(courses_per_student, len(course_pool)), replace=False
            )

            for course_id in selected_courses:
                difficulty = course_difficulty[course_id]
                credit_value = course_credits[course_id]

                attendance = np.clip(rng.normal(80, 10), 40, 100)
                midterm = np.clip(rng.normal(baseline, 12), 0, 100)
                assignment = np.clip(rng.normal(baseline + 3, 10), 0, 100)
                quiz = np.clip(rng.normal(baseline - 2, 12), 0, 100)
                prereq_grade = np.clip(rng.normal(prev_gpa * 10, 12), 0, 100)

                # Study hours and extracurricular influence grades
                study_bonus = (study_hours - 15) * 0.3
                extra_penalty = max(0, extra_load - 5) * 0.8

                latent_score = (
                    0.35 * midterm
                    + 0.2 * assignment
                    + 0.15 * quiz
                    + 0.2 * attendance
                    + 0.1 * prereq_grade
                    - 12 * difficulty
                    + study_bonus
                    - extra_penalty
                    + rng.normal(0, 3)
                )

                grade_category = _grade_from_score(latent_score)

                records.append(
                    {
                        "student_id": sid,
                        "semester": int(semester),
                        "department": department,
                        "year": int(year),
                        "scholarship_tier": tier,
                        "course_id": course_id,
                        "credit_value": int(credit_value),
                        "midterm_score": round(float(midterm), 2),
                        "attendance_rate": round(float(attendance), 2),
                        "assignment_average": round(float(assignment), 2),
                        "quiz_average": round(float(quiz), 2),
                        "study_hours_per_week": round(float(study_hours), 1),
                        "extracurricular_load": round(float(extra_load), 1),
                        "previous_sem_gpa": round(float(prev_gpa), 2),
                        "prerequisite_grade": round(float(prereq_grade), 2),
                        "course_difficulty_index": round(float(difficulty), 3),
                        "grade_category": grade_category,
                        "grade_point": GRADE_POINTS[grade_category],
                    }
                )

            # Update prev_gpa from this semester's actual performance
            sem_records = [r for r in records if r["student_id"] == sid and r["semester"] == semester]
            if sem_records:
                total_credits = sum(r["credit_value"] for r in sem_records)
                if total_credits > 0:
                    sem_gpa = sum(
                        r["grade_point"] * r["credit_value"] for r in sem_records
                    ) / total_credits
                    prev_gpa = sem_gpa

    return pd.DataFrame(records)


def write_to_sqlite(df: pd.DataFrame, db_path: Path, run_id: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        # Drop old table for clean regeneration
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
    parser.add_argument("--n-semesters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default="data/academic_records.csv")
    parser.add_argument("--out-db", type=str, default="data/academic.db")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"run_{args.seed}_{args.n_students}_{args.courses_per_student}_s{args.n_semesters}"

    df = generate_data(args.n_students, args.courses_per_student, args.n_semesters, args.seed)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    write_to_sqlite(df, Path(args.out_db), run_id)
    print(f"Wrote {len(df)} rows to {args.out_csv} and {args.out_db}")
    print(f"Students: {df['student_id'].nunique()}, Semesters: {df['semester'].nunique()}")
    print(f"Departments: {df['department'].nunique()}, Tiers: {df['scholarship_tier'].nunique()}")


if __name__ == "__main__":
    main()
