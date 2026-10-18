"""Recreate the dataset, train all three tasks, and rewrite the artifacts.

Run from the project root:

    python run_all.py --student-id ug2527014@ines.ac.rw --seed 7014
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LLLL, VERIFICATION_CODE
from src.generate import generate_and_save
from src.modeling import train_all
from src.preprocess import drop_duplicate_farms, split_train_test
from src.reporting import write_training_artifacts
from src.validate import DataValidationError, validate_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the student id and the registration seed."""
    parser = argparse.ArgumentParser(description="Train the Musanze Harvest Intelligence Service.")
    parser.add_argument("--student-id", required=True, help="Institutional username")
    parser.add_argument("--seed", required=True, type=int, help=f"Must be {LLLL}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Generate data, clean it, train the three tasks, and list the artifacts."""
    args = parse_args(argv)
    try:
        generated = generate_and_save(student_id=args.student_id, seed=args.seed)
        frame = pd.read_csv(generated["csv_path"])
        validate_dataset(frame)
        cleaned = drop_duplicate_farms(frame)
        train_frame, test_frame = split_train_test(cleaned)
        results = train_all(train_frame, test_frame)
        written = write_training_artifacts(results, str(generated["sha256"]), PROJECT_ROOT)
    except DataValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Verification code: {VERIFICATION_CODE}")
    print(f"SHA-256: {generated['sha256']}")
    print(f"Rows in CSV: {generated['rows']}")
    print(f"Rows after dropping duplicates: {len(cleaned)}")
    print(f"Train rows: {results['train_rows']}")
    print(f"Test rows: {results['test_rows']}")
    print(f"Selected regression model: {results['regression']['selected_model']}")
    print(results["regression"]["selection_reason"])
    print(f"Selected classification model: {results['classification']['selected_model']}")
    print(results["classification"]["selection_reason"])
    print(f"Selected clusters: K={results['clustering']['selected_k']}")
    print(results["clustering"]["selection_reason"])
    print("Artifacts:")
    print(f"  {generated['fingerprint_path']}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
