import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Path Configuration (Points to Random Forest Model Files)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "dam_risk_rf.joblib")
SCALER_FILE = os.path.join(BASE_DIR, "scaler_rf.joblib")

# Load model and scaler at application startup
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    raise FileNotFoundError(
        f"Model files not found! Ensure '{MODEL_FILE}' and '{SCALER_FILE}' exist."
    )

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# Initialize FastAPI App
app = FastAPI(
    title="Dam Failure Risk Scoring Microservice (Random Forest)",
    version="2.0.0",
    description="Random Forest ML API for predicting structural dam failure risks.",
)


# Input Schema Validation
class RiskPredictionRequest(BaseModel):
    dam_id: str = Field(..., example="DAM-001")
    water_level_pct: float = Field(..., ge=0, le=100, example=88.5)
    rainfall_mm: float = Field(..., ge=0, example=120.4)
    dam_age_years: float = Field(..., ge=0, example=45.0)
    seepage_rate_lpm: float = Field(..., ge=0, example=8.2)
    gate_open_pct: float = Field(..., ge=0, le=100, example=20.0)


# Response Schema
class RiskPredictionResponse(BaseModel):
    dam_id: str
    risk_score_pct: float
    hazard_level: str
    status: str


def determine_hazard_level(risk_score: float) -> str:
    if risk_score >= 80.0:
        return "CRITICAL"
    elif risk_score >= 60.0:
        return "HIGH"
    elif risk_score >= 35.0:
        return "MEDIUM"
    else:
        return "LOW"


@app.get("/")
def read_root():
    return {
        "service": "Dam Failure Risk Scoring Microservice",
        "model_type": "Random Forest Regressor",
        "status": "Healthy",
    }


@app.post(
    "/api/v1/ml/predict-risk",
    response_model=RiskPredictionResponse,
    summary="Predict Dam Risk using Random Forest",
)
def predict_risk(payload: RiskPredictionRequest):
    try:
        # Extract features in exact trained order
        features = np.array(
            [
                [
                    payload.water_level_pct,
                    payload.rainfall_mm,
                    payload.dam_age_years,
                    payload.seepage_rate_lpm,
                    payload.gate_open_pct,
                ]
            ]
        )

        # Preprocess features using loaded scaler
        scaled_features = scaler.transform(features)

        # Run Random Forest prediction
        predicted_score = float(model.predict(scaled_features)[0])
        # Clip response to valid percentage boundaries [0, 100]
        predicted_score = float(np.clip(predicted_score, 0.0, 100.0))

        hazard = determine_hazard_level(predicted_score)

        return RiskPredictionResponse(
            dam_id=payload.dam_id,
            risk_score_pct=round(predicted_score, 2),
            hazard_level=hazard,
            status="SUCCESS",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        )