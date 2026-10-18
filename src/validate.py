"""Schema, range, missing-value, duplicate, and leakage checks."""

from __future__ import annotations

import pandas as pd

from src.config import (
    ALL_COLUMNS,
    BASE_ROW_COUNT,
    CATEGORICAL_VALUES,
    CLASSIFICATION_TARGET,
    DUPLICATE_ROW_COUNT,
    FEATURE_COLUMNS,
    FINAL_ROW_COUNT,
    LEAKAGE_COLUMNS,
    LLLL,
    MISSING_COUNT,
    NUMERIC_RANGES,
    REGISTRATION_NUMBER,
    STUDENT_ID,
)


class DataValidationError(ValueError):
    """Raised when identity settings or the dataset fail a required check."""


def require_seed(seed: int) -> None:
    """Reject any seed other than the registration digits LLLL."""
    if int(seed) != LLLL:
        raise DataValidationError(
            f"Seed {seed} is not allowed. Use seed {LLLL}, the last four digits "
            f"of registration number {REGISTRATION_NUMBER}."
        )


def require_student_id(student_id: str) -> None:
    """Reject a student id that does not match this submission."""
    if student_id != STUDENT_ID:
        raise DataValidationError(
            f"Student id {student_id!r} does not match {STUDENT_ID}."
        )


def assert_no_target_leakage(feature_columns: list[str]) -> None:
    """Reject yield, yield per hectare, or high_risk used as model inputs."""
    leaked = [column for column in feature_columns if column in LEAKAGE_COLUMNS]
    if leaked:
        joined = ", ".join(leaked)
        raise DataValidationError(
            "Target leakage: these columns cannot be model inputs: " + joined
        )


def validate_dataset(frame: pd.DataFrame) -> dict[str, int]:
    """Check the saved table and return missing-value and duplicate counts.

    Missing values and the three appended copies are expected. The check
    confirms they appear only where the generator was required to put them.
    """
    errors: list[str] = []
    if list(frame.columns) != ALL_COLUMNS:
        errors.append(
            "Columns must be "
            + ", ".join(ALL_COLUMNS)
            + f". Found: {', '.join(map(str, frame.columns))}."
        )

    if len(frame) != FINAL_ROW_COUNT:
        errors.append(
            f"Expected {FINAL_ROW_COUNT} rows ({BASE_ROW_COUNT} base + "
            f"{DUPLICATE_ROW_COUNT} duplicates), found {len(frame)}."
        )

    extra_duplicates = 0
    if not errors:
        extra_duplicates = len(frame) - len(frame.drop_duplicates())
        if extra_duplicates != DUPLICATE_ROW_COUNT:
            errors.append(
                f"Expected {DUPLICATE_ROW_COUNT} extra duplicate rows, "
                f"found {extra_duplicates}."
            )

    missing_counts: dict[str, int] = {}
    for column in ALL_COLUMNS:
        if column not in frame.columns:
            continue
        missing_counts[column] = int(frame[column].isna().sum())

    if not errors:
        unique_rows = frame.drop_duplicates()
        for column in ("soil_moisture_pct", "fertilizer_kg"):
            unique_missing = int(unique_rows[column].isna().sum())
            if unique_missing != MISSING_COUNT:
                errors.append(
                    f"{column} should have {MISSING_COUNT} missing values "
                    f"before the duplicate rows, found {unique_missing}."
                )
        for column in ALL_COLUMNS:
            if column in ("soil_moisture_pct", "fertilizer_kg"):
                continue
            if missing_counts[column] != 0:
                errors.append(
                    f"{column} should have no missing values, "
                    f"found {missing_counts[column]}."
                )

    for column, (lower, upper) in NUMERIC_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        observed = values.dropna()
        if observed.empty:
            continue
        if (observed < lower).any():
            errors.append(f"{column} has a value below {lower}.")
        if upper is not None and (observed > upper).any():
            errors.append(f"{column} has a value above {upper}.")

    for column, allowed in CATEGORICAL_VALUES.items():
        if column not in frame.columns:
            continue
        present = frame[column].dropna()
        if column == CLASSIFICATION_TARGET or column == "cooperative_support":
            try:
                observed_values = set(present.astype(int).tolist())
            except (TypeError, ValueError):
                errors.append(f"{column} contains a non-integer value.")
                continue
        else:
            observed_values = set(present.astype(str).tolist())
        unexpected = observed_values - allowed
        if unexpected:
            errors.append(
                f"{column} has values outside {sorted(allowed)}: "
                f"{sorted(unexpected)}."
            )

    if "yield_per_ha" in frame.columns:
        errors.append("yield_per_ha must not be stored in the dataset.")

    try:
        assert_no_target_leakage(list(FEATURE_COLUMNS))
    except DataValidationError as exc:
        errors.append(str(exc))

    if errors:
        details = "\n- ".join(errors)
        raise DataValidationError("Dataset validation failed:\n- " + details)

    return {
        "rows": int(len(frame)),
        "base_rows": BASE_ROW_COUNT,
        "extra_duplicate_rows": extra_duplicates,
        "missing_soil_moisture_pct": missing_counts["soil_moisture_pct"],
        "missing_fertilizer_kg": missing_counts["fertilizer_kg"],
    }
