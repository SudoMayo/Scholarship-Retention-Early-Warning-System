"""Automated retraining trigger based on new data run IDs.

This script is intended for live demo/viva usage to prove drift-ready retraining.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import sys
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_model import train_and_evaluate


def _fetch_data_runs(db_path: Path) -> List[Tuple[str, str, int]]:
    """Return run history ordered by creation timestamp."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT run_id, created_at, n_records FROM data_runs ORDER BY created_at"
        ).fetchall()
    return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]


def _new_run_ids_since(runs: List[Tuple[str, str, int]], last_run_id: str | None) -> List[str]:
    """Return run IDs that were created after the stored last trained run."""
    if not runs:
        return []
    run_ids = [run[0] for run in runs]
    if not last_run_id or last_run_id not in run_ids:
        return run_ids
    index = run_ids.index(last_run_id)
    return run_ids[index + 1 :]


def main() -> None:
    """CLI entry point for data-run-driven retraining."""
    parser = argparse.ArgumentParser(description="Trigger retraining when new run_ids are detected")
    parser.add_argument("--db-path", type=str, default="data/academic.db")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    models_dir = Path(args.models_dir)
    marker_path = models_dir / "last_trained_run.txt"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    runs = _fetch_data_runs(db_path)
    if not runs:
        print("No data runs available. Generate data first.")
        return

    last_trained_run = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else None
    pending_run_ids = _new_run_ids_since(runs, last_trained_run)

    if not pending_run_ids:
        print("No new data detected. Pipeline is up to date.")
        return

    placeholders = ",".join("?" for _ in pending_run_ids)
    with sqlite3.connect(db_path) as conn:
        new_records = conn.execute(
            f"SELECT COUNT(*) FROM academic_records WHERE run_id IN ({placeholders})",
            pending_run_ids,
        ).fetchone()[0]

    print(f"New data detected ({int(new_records)} new records). Retraining triggered...")

    summary = train_and_evaluate(db_path=str(db_path), models_dir=models_dir, seed=args.seed)

    latest_run_id = pending_run_ids[-1]
    models_dir.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(latest_run_id, encoding="utf-8")

    print(f"Model v{summary['version']} saved.")
    print(f"Updated last trained run marker: {latest_run_id}")


if __name__ == "__main__":
    main()
