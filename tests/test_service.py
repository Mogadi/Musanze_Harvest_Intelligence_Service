"""Inference, validation, reproducibility, and API checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from predict import main
from src.config import VERIFICATION_CODE
from src.generate import DEFAULT_CSV_PATH, DEFAULT_FINGERPRINT_PATH, file_sha256, generate_frame
from src.inference import PredictionInputError, load_json_record, predict_record

ROOT = Path(__file__).resolve().parents[1]
VALID_PATH = ROOT / "samples" / "valid.json"
INVALID_PATH = ROOT / "samples" / "invalid.json"
client = TestClient(app)


def valid_record() -> dict:
    return json.loads(VALID_PATH.read_text(encoding="utf-8"))


def test_valid_inference_returns_the_verification_code(capsys) -> None:
    """A complete in-range farm returns yield, risk, probability, and cluster."""
    result = predict_record(valid_record())
    assert result["verification_code"] == VERIFICATION_CODE
    assert result["yield_tons"] > 0
    assert result["high_risk"] in (0, 1)
    assert 0.0 <= result["high_risk_probability"] <= 1.0
    assert isinstance(result["cluster"], int)

    main(["--json", str(VALID_PATH)])
    printed = json.loads(capsys.readouterr().out)
    assert printed == result


def test_missing_field_is_rejected() -> None:
    """Omitting a required field is a 422, not a prediction."""
    record = valid_record()
    del record["soil_moisture_pct"]
    with pytest.raises(PredictionInputError, match="soil_moisture_pct"):
        predict_record(record)

    response = client.post("/predict", json=record)
    assert response.status_code == 422
    assert "soil_moisture_pct" in response.text


def test_wrong_type_is_rejected() -> None:
    """A numeric season is a type error."""
    record = valid_record()
    record["season"] = 1
    with pytest.raises(PredictionInputError, match="season"):
        predict_record(record)

    response = client.post("/predict", json=record)
    assert response.status_code == 422
    assert "season" in response.text


def test_range_error_is_rejected() -> None:
    """Rainfall below 600 mm is rejected by the CLI and the API."""
    record = load_json_record(INVALID_PATH)
    with pytest.raises(PredictionInputError, match="rainfall_mm"):
        predict_record(record)

    response = client.post("/predict", json=record)
    assert response.status_code == 422
    body = response.json()
    assert body["verification_code"] == VERIFICATION_CODE
    assert any(item["field"] == "rainfall_mm" for item in body["detail"])


def test_dataset_and_prediction_are_reproducible() -> None:
    """The same seed and the same farm produce the same dataset and the same scores."""
    first = generate_frame(7014)
    second = generate_frame(7014)
    pd.testing.assert_frame_equal(first, second)
    digest = file_sha256(DEFAULT_CSV_PATH)
    assert digest in DEFAULT_FINGERPRINT_PATH.read_text(encoding="utf-8")
    assert predict_record(valid_record()) == predict_record(valid_record())


def test_api_health_model_info_and_predict() -> None:
    """The service reports its identity and scores a valid farm."""
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "verification_code": VERIFICATION_CODE}

    info = client.get("/model-info")
    assert info.status_code == 200
    payload = info.json()
    assert payload["verification_code"] == VERIFICATION_CODE
    assert payload["selected_regression"] == "linear_regression"
    assert payload["selected_k"] == 2

    prediction = client.post("/predict", json=valid_record())
    assert prediction.status_code == 200
    assert prediction.json()["verification_code"] == VERIFICATION_CODE
    assert set(prediction.json()) == {
        "verification_code",
        "yield_tons",
        "high_risk",
        "high_risk_probability",
        "cluster",
    }
