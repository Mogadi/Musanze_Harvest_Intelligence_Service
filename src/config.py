"""Identity constants and dataset rules for registration number 25/27014."""

from __future__ import annotations

STUDENT_NAME = "Mohammed Osama Hasan"
REGISTRATION_NUMBER = "25/27014"
REGISTRATION_DIGITS = "2527014"
STUDENT_ID = "ug2527014@ines.ac.rw"

# Last four digits of the registration number. Also the train/test random_state.
LLLL = 7014
VERIFICATION_CODE = f"AI-A2-{LLLL}"
TEST_SIZE = 0.2

# 240 + (7014 mod 41) = 243 base rows, then three exact duplicates.
BASE_ROW_COUNT = 240 + (LLLL % 41)
MISSING_COUNT = int(round(0.05 * BASE_ROW_COUNT))
DUPLICATE_ROW_COUNT = 3
FINAL_ROW_COUNT = BASE_ROW_COUNT + DUPLICATE_ROW_COUNT

CSV_NAME = f"AI_A2_{REGISTRATION_DIGITS}.csv"

FEATURE_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]
REGRESSION_TARGET = "yield_tons"
CLASSIFICATION_TARGET = "high_risk"
ALL_COLUMNS = FEATURE_COLUMNS + [REGRESSION_TARGET, CLASSIFICATION_TARGET]

# Never pass these to a classifier or clusterer. yield_per_ha is not stored,
# but it is listed so a later feature list cannot smuggle it in.
LEAKAGE_COLUMNS = [REGRESSION_TARGET, CLASSIFICATION_TARGET, "yield_per_ha"]

FLOAT_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    REGRESSION_TARGET,
]

# Inclusive bounds checked on the saved four-decimal CSV.
# Uniform draws use a half-open interval, and rounding can land on the endpoint.
NUMERIC_RANGES: dict[str, tuple[float, float | None]] = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
    REGRESSION_TARGET: (0.5, None),
}

CATEGORICAL_VALUES = {
    "season": {"A", "B"},
    "cooperative_support": {0, 1},
    CLASSIFICATION_TARGET: {0, 1},
}
