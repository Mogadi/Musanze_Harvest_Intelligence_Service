"""Local API for the Musanze Harvest Intelligence Service.

Run from the project root:

    uvicorn app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config import REGISTRATION_NUMBER, STUDENT_NAME, VERIFICATION_CODE
from src.inference import FarmRecord, Prediction, load_service, predict_record

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    summary="Classroom decision-support prototype for a Musanze potato cooperative.",
    version=VERIFICATION_CODE,
)


@app.exception_handler(RequestValidationError)
async def invalid_farm(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a readable 422 body for a missing field, wrong type, or range error."""
    details = []
    for error in exc.errors():
        location = [str(part) for part in error["loc"] if part != "body"]
        details.append(
            {
                "field": ".".join(location) or "request",
                "message": error["msg"],
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "verification_code": VERIFICATION_CODE,
            "detail": details,
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the local service is running."""
    return {"status": "ok", "verification_code": VERIFICATION_CODE}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Describe the saved models and the student verification code."""
    manifest = load_service()["manifest"]
    return {
        "verification_code": VERIFICATION_CODE,
        "student_name": STUDENT_NAME,
        "registration_number": REGISTRATION_NUMBER,
        "selected_regression": manifest["selected_regression"],
        "selected_classification": manifest["selected_classification"],
        "selected_k": manifest["selected_k"],
        "dataset_sha256": manifest["dataset_sha256"],
        "feature_columns": manifest["feature_columns"],
    }


@app.post("/predict", response_model=Prediction)
def predict(farm: FarmRecord) -> dict[str, object]:
    """Estimate yield, high-risk probability, and cluster for one farm."""
    return predict_record(farm.model_dump())
