import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Path pointing to the parent directory where 'dam_risk_dataset.csv' resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "dam_risk_dataset.csv")

MODEL_FILE = os.path.join(BASE_DIR, "dam_risk_xgboost.joblib")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.joblib")

FEATURE_COLUMNS = [
    "water_level_pct",
    "rainfall_mm",
    "dam_age_years",
    "seepage_rate_lpm",
    "gate_open_pct",
]
TARGET_COLUMN = "risk_score_pct"


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    # Resolve the absolute normalized path
    abs_csv_path = os.path.abspath(csv_path)

    if not os.path.exists(abs_csv_path):
        raise FileNotFoundError(f"CSV dataset not found at '{abs_csv_path}'")

    print(f"📂 Loading dataset from: {abs_csv_path}")
    df = pd.read_csv(abs_csv_path)

    # Clean missing values
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill().bfill()
    df = df.dropna(subset=[TARGET_COLUMN])
    return df


def train_with_grid_search():
    df = load_and_preprocess_data(CSV_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Base XGBoost Regressor
    xgb = XGBRegressor(random_state=42)

    # Hyperparameter Grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    print("🚀 Starting GridSearchCV to optimize XGBoost Model...")

    # Grid Search Configuration
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    # Fit Grid Search
    grid_search.fit(X_train_scaled, y_train)

    # Best Model Artifacts
    best_model = grid_search.best_estimator_
    print(f"\n🏆 Best Hyperparameters Found: {grid_search.best_params_}")

    # Evaluate Best Model
    predictions = best_model.predict(X_test_scaled)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n--- 📊 MODEL EVALUATION METRICS ---")
    print(f" R2 Score (Accuracy): {r2:.4f}")
    print(f" RMSE (Average Error): {rmse:.4f}%")

    # Save Best Model & Scaler into current working folder
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(
        f" Optimized model saved to '{MODEL_FILE}' & scaler saved to '{SCALER_FILE}'"
    )


if __name__ == "__main__":
    train_with_grid_search()