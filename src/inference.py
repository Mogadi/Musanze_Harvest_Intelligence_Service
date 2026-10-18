"""Validate one farm record and score it with the saved pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.config import FEATURE_COLUMNS, NUMERIC_RANGES, VERIFICATION_CODE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
NUMERIC_INPUTS = (
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
)


class PredictionInputError(ValueError):
    """Raised when a farm record cannot be scored."""


class FarmRecord(BaseModel):
    """One farm in the ranges the Musanze models were built to accept."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "rainfall_mm": 1100.0,
                    "avg_temp_c": 18.5,
                    "soil_moisture_pct": 55.0,
                    "fertilizer_kg": 120.0,
                    "field_size_ha": 1.5,
                    "season": "A",
                    "cooperative_support": 1,
                }
            ]
        },
    )

    rainfall_mm: float
    avg_temp_c: float
    soil_moisture_pct: float
    fertilizer_kg: float
    field_size_ha: float
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]

    @field_validator(*NUMERIC_INPUTS, mode="before")
    @classmethod
    def numeric_value(cls, value: object) -> object:
        """Reject nulls, booleans, and text before the range check."""
        if value is None:
            raise ValueError("is required and cannot be null")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return value

    @field_validator(*NUMERIC_INPUTS)
    @classmethod
    def within_training_range(cls, value: float, info) -> float:
        """Keep numeric inputs inside the generator bounds."""
        lower, upper = NUMERIC_RANGES[info.field_name]
        if value < lower or (upper is not None and value > upper):
            raise ValueError(f"must be between {lower:g} and {upper:g}")
        return value

    @field_validator("season", mode="before")
    @classmethod
    def season_text(cls, value: object) -> object:
        """Require a text season code before checking A or B."""
        if value is None:
            raise ValueError("is required and cannot be null")
        if not isinstance(value, str):
            raise ValueError("must be the text A or B")
        return value

    @field_validator("cooperative_support", mode="before")
    @classmethod
    def support_flag(cls, value: object) -> object:
        """Require the integer 0 or 1, not text or a boolean."""
        if value is None:
            raise ValueError("is required and cannot be null")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be the integer 0 or 1")
        return value


class Prediction(BaseModel):
    """Scores returned for one farm."""

    verification_code: str
    yield_tons: float
    high_risk: int = Field(description="1 means the model predicts high production risk")
    high_risk_probability: float
    cluster: int


def format_validation_error(exc: ValidationError) -> str:
    """Turn a pydantic error into one readable sentence."""
    parts = []
    for error in exc.errors():
        field = ".".join(str(item) for item in error["loc"]) or "record"
        parts.append(f"{field}: {error['msg']}")
    return "Invalid farm record. " + " ".join(parts)


def parse_farm(record: dict) -> FarmRecord:
    """Validate a farm dictionary or raise PredictionInputError."""
    try:
        return FarmRecord.model_validate(record)
    except ValidationError as exc:
        raise PredictionInputError(format_validation_error(exc)) from exc


def load_json_record(path: Path) -> dict:
    """Read one JSON object from disk."""
    if not path.is_file():
        raise PredictionInputError(f"JSON file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictionInputError(f"The file is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise PredictionInputError("The JSON file must contain one farm object.")
    return payload


def load_service() -> dict[str, object]:
    """Load the saved pipelines and the model manifest once."""
    if load_service.cache is not None:
        return load_service.cache
    required = {
        "regression": MODEL_DIR / "regression_pipeline.joblib",
        "classification": MODEL_DIR / "classification_pipeline.joblib",
        "clustering": MODEL_DIR / "clustering_pipeline.joblib",
        "manifest": MODEL_DIR / "manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Saved models are missing. Run run_all.py first. Missing: {joined}")
    load_service.cache = {
        "regression": joblib.load(required["regression"]),
        "classification": joblib.load(required["classification"]),
        "clustering": joblib.load(required["clustering"]),
        "manifest": json.loads(required["manifest"].read_text(encoding="utf-8")),
    }
    return load_service.cache


load_service.cache = None


def _positive_probability(pipeline, frame: pd.DataFrame) -> float:
    classes = list(pipeline.named_steps["model"].classes_)
    probabilities = pipeline.predict_proba(frame)[0]
    return float(probabilities[classes.index(1)])


def predict_record(record: dict) -> dict[str, object]:
    """Score one farm and return the yield, risk, probability, and cluster."""
    farm = parse_farm(record)
    frame = pd.DataFrame([farm.model_dump()])[FEATURE_COLUMNS]
    service = load_service()
    yield_estimate = float(service["regression"].predict(frame)[0])
    risk = int(service["classification"].predict(frame)[0])
    probability = _positive_probability(service["classification"], frame)
    cluster = int(service["clustering"].predict(frame)[0])
    prediction = Prediction(
        verification_code=VERIFICATION_CODE,
        yield_tons=round(yield_estimate, 4),
        high_risk=risk,
        high_risk_probability=round(probability, 4),
        cluster=cluster,
    )
    return prediction.model_dump()
