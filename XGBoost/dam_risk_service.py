import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_FILE = "dam_risk_xgboost.joblib"
SCALER_FILE = "scaler.joblib"

# Load Pre-trained Artifacts
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    raise RuntimeError(
        f"Model artifacts missing! Run 'python train.py' to build {MODEL_FILE} first."
    )

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

app = FastAPI(
    title="Dam Failure Risk Scoring Microservice",
    description="XGBoost ML API for predicting structural dam failure risks.",
    version="1.0.0",
)


class DamMetricsRequest(BaseModel):
    dam_id: str = Field(..., example="DAM-001")
    water_level_pct: float = Field(..., ge=0.0, le=100.0, example=88.5)
    rainfall_mm: float = Field(..., ge=0.0, example=120.4)
    dam_age_years: float = Field(..., ge=0.0, example=45.0)
    seepage_rate_lpm: float = Field(..., ge=0.0, example=8.2)
    gate_open_pct: float = Field(..., ge=0.0, le=100.0, example=20.0)


class RiskAssessmentResponse(BaseModel):
    dam_id: str
    risk_score_pct: float
    hazard_category: str
    recommended_action: str


def classify_risk(risk_score: float):
    if risk_score < 30.0:
        return ("LOW", "Normal operations. Routine sensor monitoring.")
    elif risk_score < 60.0:
        return (
            "MEDIUM",
            "Increased monitoring required. Inspect seepage and embankment.",
        )
    elif risk_score < 80.0:
        return (
            "HIGH",
            "Prepare emergency teams. Consider opening spillway gates.",
        )
    else:
        return (
            "CRITICAL",
            "IMMINENT BREACH RISK! Broadcast emergency evacuation alerts!",
        )


@app.post("/api/v1/ml/predict-risk", response_model=RiskAssessmentResponse)
def predict_dam_risk(data: DamMetricsRequest):
    try:
        input_features = np.array(
            [
                [
                    data.water_level_pct,
                    data.rainfall_mm,
                    data.dam_age_years,
                    data.seepage_rate_lpm,
                    data.gate_open_pct,
                ]
            ]
        )

        scaled_features = scaler.transform(input_features)
        predicted_risk = float(model.predict(scaled_features)[0])
        predicted_risk = max(0.0, min(100.0, predicted_risk))

        category, action = classify_risk(predicted_risk)

        return RiskAssessmentResponse(
            dam_id=data.dam_id,
            risk_score_pct=round(predicted_risk, 2),
            hazard_category=category,
            recommended_action=action,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))