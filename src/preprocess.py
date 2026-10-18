"""Cleaning and train-only preprocessing for the harvest models.

Duplicate rows are removed before the split so the same farm cannot appear in
both sets. Imputation, scaling, and encoding are not fit here. They live inside
each model pipeline and are fit on the training split only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_COLUMNS,
    CLASSIFICATION_TARGET,
    DUPLICATE_ROW_COUNT,
    FEATURE_COLUMNS,
    LLLL,
    TEST_SIZE,
)
from src.validate import DataValidationError, assert_no_target_leakage

CATEGORICAL_FEATURES = ["season"]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]


def make_preprocessor() -> ColumnTransformer:
    """Return an unfitted transformer for the official feature columns."""
    assert_no_target_leakage(list(FEATURE_COLUMNS))
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def drop_duplicate_farms(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the appended copies and keep a stable row id from the CSV order."""
    if list(frame.columns) != ALL_COLUMNS:
        found = ", ".join(map(str, frame.columns))
        raise DataValidationError(f"Cannot clean a table with unexpected columns: {found}.")

    numbered = frame.copy()
    numbered.insert(0, "row_id", range(1, len(numbered) + 1))
    cleaned = numbered.drop_duplicates(subset=ALL_COLUMNS, keep="first").reset_index(drop=True)
    removed = len(numbered) - len(cleaned)
    if removed != DUPLICATE_ROW_COUNT:
        raise DataValidationError(
            f"Cleaning should remove {DUPLICATE_ROW_COUNT} duplicate rows, but removed {removed}."
        )
    return cleaned


def split_train_test(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split farms with the registration seed, stratified on high_risk."""
    from sklearn.model_selection import train_test_split

    if CLASSIFICATION_TARGET not in cleaned.columns:
        raise DataValidationError("The classification target is missing, so the split cannot be stratified.")

    train_frame, test_frame = train_test_split(
        cleaned,
        test_size=TEST_SIZE,
        random_state=LLLL,
        stratify=cleaned[CLASSIFICATION_TARGET],
    )
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)
