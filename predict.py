"""Score one farm from a JSON file with the saved models.

Run from the project root:

    python predict.py --json samples/valid.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import PredictionInputError, load_json_record, predict_record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the path to one farm JSON record."""
    parser = argparse.ArgumentParser(description="Predict harvest, risk, and cluster for one farm.")
    parser.add_argument("--json", required=True, type=Path, help="Path to one farm JSON object")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Validate the record, print the prediction JSON, and exit with an error code on bad input."""
    args = parse_args(argv)
    try:
        result = predict_record(load_json_record(args.json))
    except PredictionInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
