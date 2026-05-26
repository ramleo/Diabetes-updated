"""
Diabetes Prediction API
Binary classification: 0 = no diabetes, 1 = diabetes
Model: LogisticRegression (final_pipeline.pkl)
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# ── App & model setup ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "final_pipeline.pkl")

app = FastAPI(
    title="Diabetes Prediction API",
    description="Binary classification: predict diabetes risk from clinical features.",
    version="1.0.0",
)

# Load pipeline once at startup
pipeline = joblib.load(PIPELINE_PATH)

# Columns that the pipeline expects (all numeric)
FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Columns where 0 is physiologically impossible — replace with NaN so the
# pipeline's median imputer handles them (mirrors src/preprocess.py logic)
ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# ── Pydantic models ───────────────────────────────────────────────────────────

class DiabetesFeatures(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    Glucose: Optional[float] = Field(
        None, ge=0, description="Plasma glucose concentration (mg/dL); omit or set 0 to impute"
    )
    BloodPressure: Optional[float] = Field(
        None, ge=0, description="Diastolic blood pressure (mm Hg); omit or set 0 to impute"
    )
    SkinThickness: Optional[float] = Field(
        None, ge=0, description="Triceps skin fold thickness (mm); omit or set 0 to impute"
    )
    Insulin: Optional[float] = Field(
        None, ge=0, description="2-Hour serum insulin (mu U/ml); omit or set 0 to impute"
    )
    BMI: Optional[float] = Field(
        None, ge=0, description="Body mass index (kg/m²); omit or set 0 to impute"
    )
    DiabetesPedigreeFunction: float = Field(
        ..., ge=0.0, description="Diabetes pedigree function score"
    )
    Age: int = Field(..., ge=1, description="Age in years")

    class Config:
        json_schema_extra = {
            "example": {
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50,
            }
        }


class PredictionResult(BaseModel):
    prediction: int = Field(..., description="0 = no diabetes, 1 = diabetes")
    label: str = Field(..., description="Human-readable label")
    probability_no_diabetes: float = Field(..., description="P(Outcome=0)")
    probability_diabetes: float = Field(..., description="P(Outcome=1)")


class BatchPredictionResult(BaseModel):
    predictions: List[PredictionResult]
    count: int


# ── Helper ────────────────────────────────────────────────────────────────────

def _features_to_dataframe(features: DiabetesFeatures) -> pd.DataFrame:
    """Convert a single DiabetesFeatures instance to a pipeline-ready DataFrame."""
    row = {
        "Pregnancies": float(features.Pregnancies),
        "Glucose": features.Glucose,
        "BloodPressure": features.BloodPressure,
        "SkinThickness": features.SkinThickness,
        "Insulin": features.Insulin,
        "BMI": features.BMI,
        "DiabetesPedigreeFunction": features.DiabetesPedigreeFunction,
        "Age": float(features.Age),
    }

    df = pd.DataFrame([row], columns=FEATURE_COLS)

    # Replicate zero→NaN logic from src/preprocess.py
    for col in ZERO_AS_NAN_COLS:
        df = df.assign(**{col: df[col].astype(float).replace(0, np.nan)})

    return df


def _predict_single(features: DiabetesFeatures) -> PredictionResult:
    df = _features_to_dataframe(features)
    pred = int(pipeline.predict(df)[0])
    proba = pipeline.predict_proba(df)[0]
    return PredictionResult(
        prediction=pred,
        label="diabetes" if pred == 1 else "no diabetes",
        probability_no_diabetes=round(float(proba[0]), 4),
        probability_diabetes=round(float(proba[1]), 4),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Health check")
def health():
    """Return service status and active model name."""
    model_name = type(pipeline.named_steps["model"]).__name__
    return {"status": "ok", "model": model_name}


@app.post("/predict", response_model=PredictionResult, summary="Single prediction")
def predict(features: DiabetesFeatures):
    """
    Predict diabetes risk for a single patient.

    - Optional fields (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
      can be omitted or set to 0 — they will be median-imputed by the pipeline.
    """
    try:
        return _predict_single(features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResult,
    summary="Batch prediction",
)
def predict_batch(patients: List[DiabetesFeatures]):
    """
    Predict diabetes risk for multiple patients in a single request.

    Returns a list of predictions in the same order as the input.
    """
    if not patients:
        raise HTTPException(status_code=422, detail="Request body must not be empty.")
    try:
        results = [_predict_single(p) for p in patients]
        return BatchPredictionResult(predictions=results, count=len(results))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
