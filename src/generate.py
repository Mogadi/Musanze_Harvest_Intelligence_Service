"""Build the individual Musanze potato dataset and its SHA-256 fingerprint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ALL_COLUMNS,
    BASE_ROW_COUNT,
    CSV_NAME,
    DUPLICATE_ROW_COUNT,
    FEATURE_COLUMNS,
    LLLL,
    MISSING_COUNT,
    REGISTRATION_NUMBER,
    STUDENT_NAME,
    VERIFICATION_CODE,
)
from src.validate import require_seed, require_student_id, validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / CSV_NAME
DEFAULT_FINGERPRINT_PATH = PROJECT_ROOT / "artifacts" / "data_fingerprint.txt"


def generate_frame(seed: int = LLLL) -> pd.DataFrame:
    """Create the base rows, plant the required defects, and append duplicates.

    Random draws use one ``numpy.random.default_rng(seed)`` in this order:
    rainfall, temperature, soil moisture, fertilizer, field size, season,
    cooperative support, yield noise, missing moisture positions, missing
    fertilizer positions, then the three rows that are copied.

    Yield is computed on the complete rows, clipped at 0.50 tons, and only
    then are moisture and fertilizer cells cleared. ``high_risk`` is 1 when
    yield per hectare is strictly below the median of the base rows.
    """
    require_seed(seed)
    rng = np.random.default_rng(seed)
    row_count = BASE_ROW_COUNT

    rainfall = rng.uniform(600.0, 1400.0, size=row_count)
    temperature = rng.uniform(12.0, 26.0, size=row_count)
    moisture = rng.uniform(20.0, 80.0, size=row_count)
    fertilizer = rng.uniform(0.0, 250.0, size=row_count)
    field_size = rng.uniform(0.2, 4.0, size=row_count)
    season = rng.choice(np.array(["A", "B"]), size=row_count)
    support = rng.integers(0, 2, size=row_count)
    noise = rng.normal(0.0, 0.80, size=row_count)

    season_effect = np.where(season == "A", 0.50, -0.20)
    yield_tons = (
        0.004 * rainfall
        + 0.06 * moisture
        + 0.02 * fertilizer
        + 1.30 * field_size
        - 0.25 * np.abs(temperature - 18.0)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.maximum(yield_tons, 0.50)
    yield_per_hectare = yield_tons / field_size
    high_risk = (yield_per_hectare < np.median(yield_per_hectare)).astype(int)

    moisture = moisture.astype(float)
    fertilizer = fertilizer.astype(float)
    moisture[rng.choice(row_count, size=MISSING_COUNT, replace=False)] = np.nan
    fertilizer[rng.choice(row_count, size=MISSING_COUNT, replace=False)] = np.nan
    duplicate_positions = rng.choice(row_count, size=DUPLICATE_ROW_COUNT, replace=False)

    frame = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": temperature,
            "soil_moisture_pct": moisture,
            "fertilizer_kg": fertilizer,
            "field_size_ha": field_size,
            "season": season,
            "cooperative_support": support.astype(int),
            "yield_tons": yield_tons,
            "high_risk": high_risk.astype(int),
        }
    )
    copies = frame.iloc[duplicate_positions].copy()
    frame = pd.concat([frame, copies], ignore_index=True)
    frame = frame[ALL_COLUMNS]
    return frame


def write_dataset(frame: pd.DataFrame, csv_path: Path) -> None:
    """Save numeric columns to four decimal places without an index column."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, float_format="%.4f", lineterminator="\n")


def file_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_fingerprint(fingerprint_path: Path, csv_path: Path, digest: str, rows: int) -> None:
    """Store the verification code and the CSV digest."""
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"verification_code={VERIFICATION_CODE}\n"
        f"student_name={STUDENT_NAME}\n"
        f"registration_number={REGISTRATION_NUMBER}\n"
        f"seed={LLLL}\n"
        f"csv={csv_path.name}\n"
        f"rows={rows}\n"
        f"sha256={digest}\n"
    )
    fingerprint_path.write_text(content, encoding="utf-8")


def generate_and_save(
    student_id: str,
    seed: int,
    csv_path: Path = DEFAULT_CSV_PATH,
    fingerprint_path: Path = DEFAULT_FINGERPRINT_PATH,
) -> dict[str, str | int]:
    """Validate the run settings, write the CSV, and fingerprint its bytes."""
    require_student_id(student_id)
    require_seed(seed)
    frame = generate_frame(seed)
    write_dataset(frame, csv_path)
    saved = pd.read_csv(csv_path)
    try:
        summary = validate_dataset(saved)
    except Exception:
        csv_path.unlink(missing_ok=True)
        raise
    digest = file_sha256(csv_path)
    write_fingerprint(fingerprint_path, csv_path, digest, int(summary["rows"]))
    return {
        "verification_code": VERIFICATION_CODE,
        "csv_path": str(csv_path),
        "fingerprint_path": str(fingerprint_path),
        "sha256": digest,
        "feature_count": len(FEATURE_COLUMNS),
        **summary,
    }
