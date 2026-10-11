# 🌊 Dam Failure Risk Scoring System

A machine learning-powered microservice designed to predict structural dam failure risks based on real-time sensor and environmental telemetry. The project includes comparative models (**XGBoost** and **Random Forest Regressor**) exposed through a high-performance **FastAPI** REST API.

---

## 📐 Project Architecture & Workflow

```text
[ Sensor Telemetry / Inputs ]
  ├── Water Level (%)
  ├── Rainfall (mm)
  ├── Dam Age (Years)
  ├── Seepage Rate (LPM)
  └── Spillway Gate Open (%)
              │
              ▼
   [ Preprocessing & Scaling ] (StandardScaler)
              │
              ▼
   [ Machine Learning Model ] (XGBoost / Random Forest)
              │
              ▼
    [ Risk Score Calculation ] (0% - 100%)
              │
              ▼
    [ Hazard Categorization ]
  ├── 0% - 34%:   LOW
  ├── 35% - 59%:  MEDIUM
  ├── 60% - 79%:  HIGH
  └── 80% - 100%: CRITICAL
📁 Repository Directory StructurePlaintextDam Failure Risk Scoring/
├── .gitignore
├── README.md
├── dam_risk_dataset.csv             # Centralized sensor dataset
├── XGBoost/                         # XGBoost Training & API Module
│   ├── train.py                     # GridSearchCV tuning for XGBoost
│   ├── test_model.py                # Standalone test script
│   ├── dam_risk_service.py          # FastAPI service using XGBoost
│   ├── dam_risk_xgboost.joblib      # Saved XGBoost model artifact
│   └── scaler.joblib                # Feature scaler artifact
└── Random Forest Model Training/    # Random Forest Training & API Module
    ├── train_rf.py                  # GridSearchCV tuning for Random Forest
    ├── test_rf.py                   # Standalone test script
    ├── dam_risk_service.py          # FastAPI service using Random Forest
    ├── dam_risk_rf.joblib           # Saved Random Forest model artifact
    └── scaler_rf.joblib             # Feature scaler artifact
📊 Dataset Features & TargetThe models evaluate five core feature parameters to derive a continuous risk score percentage:Feature NameTypeDescriptionwater_level_pctfloatCurrent reservoir water storage percentage (0% - 100%)rainfall_mmfloatCumulative precipitation level over the last 24 hoursdam_age_yearsfloatStructural operational age of the dam structureseepage_rate_lpmfloatMeasured foundation/internal seepage rate in Liters Per Minutegate_open_pctfloatCurrent spillway gate opening capacity percentageTarget: risk_score_pctfloatPredicted structural failure risk probability (0.0% - 100.0%)⚡ Quick Start & Installation1. PrerequisitesEnsure Python 3.9+ is installed on your system.2. Clone Repository & Install DependenciesBashgit clone [https://github.com/PiyumalSandeepa/dam-failure-risk-scoring.git](https://github.com/PiyumalSandeepa/dam-failure-risk-scoring.git)
cd dam-failure-risk-scoring

# Install required Python packages
pip install numpy pandas scikit-learn xgboost fastapi uvicorn joblib
🏋️ Model Training & EvaluationTraining XGBoost ModelBashcd "XGBoost"
python train.py
R² Score: 0.9974 (99.74% Accuracy)RMSE Error: 0.9090%Training Random Forest ModelBashcd "../Random Forest Model Training"
python train_rf.py
R² Score: ~0.9965RMSE Error: ~1.0200%🚀 Serving the FastAPI MicroserviceTo run the REST API service using the Random Forest implementation:Bashcd "Random Forest Model Training"
python -m uvicorn dam_risk_service:app --reload --port 8000
Once running, access the interactive API documentation (Swagger UI) at:Swagger UI: http://localhost:8000/docsOpenAPI Schema: http://localhost:8000/openapi.json📡 API Endpoint ReferencePOST /api/v1/ml/predict-riskPredicts the structural risk percentage and hazard level based on sensor telemetry.Request Body Payload (JSON)JSON{
  "dam_id": "DAM-001",
  "water_level_pct": 92.5,
  "rainfall_mm": 180.0,
  "dam_age_years": 45.0,
  "seepage_rate_lpm": 11.2,
  "gate_open_pct": 10.0
}
Response Payload (200 OK)JSON{
  "dam_id": "DAM-001",
  "risk_score_pct": 86.45,
  "hazard_level": "CRITICAL",
  "status": "SUCCESS"
}
🧪 Testing with PostmanOpen Postman and click Import.Paste the OpenAPI spec URL: http://localhost:8000/openapi.json.Select the imported POST /api/v1/ml/predict-risk request.Click Send to view live predictions.