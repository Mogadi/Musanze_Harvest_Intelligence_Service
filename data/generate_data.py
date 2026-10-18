"""Command-line entry point for the individual dataset generator.

Run from the project root:

    python data/generate_data.py --student-id ug2527014@ines.ac.rw --seed 7014
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LLLL, STUDENT_ID, VERIFICATION_CODE
from src.generate import DEFAULT_CSV_PATH, DEFAULT_FINGERPRINT_PATH, generate_and_save
from src.validate import DataValidationError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the student id and seed used to recreate the CSV."""
    parser = argparse.ArgumentParser(
        description="Generate the Musanze harvest dataset for AI-A2-7014."
    )
    parser.add_argument("--student-id", required=True, help="Institutional username")
    parser.add_argument("--seed", required=True, type=int, help=f"Must be {LLLL}")
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--fingerprint", type=Path, default=DEFAULT_FINGERPRINT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Generate the CSV, validate it, and print the verification code."""
    args = parse_args(argv)
    try:
        summary = generate_and_save(
            student_id=args.student_id,
            seed=args.seed,
            csv_path=args.output,
            fingerprint_path=args.fingerprint,
        )
    except DataValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Verification code: {summary['verification_code']}")
    print(f"Student id: {STUDENT_ID}")
    print(f"Seed: {LLLL}")
    print(f"Rows: {summary['rows']} ({summary['base_rows']} base + duplicates)")
    print(f"Missing soil_moisture_pct: {summary['missing_soil_moisture_pct']}")
    print(f"Missing fertilizer_kg: {summary['missing_fertilizer_kg']}")
    print(f"Extra duplicate rows: {summary['extra_duplicate_rows']}")
    print(f"SHA-256: {summary['sha256']}")
    print(f"Wrote {summary['csv_path']}")
    print(f"Wrote {summary['fingerprint_path']}")


if __name__ == "__main__":
    main()
