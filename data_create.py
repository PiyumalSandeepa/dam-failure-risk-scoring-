import numpy as np
import pandas as pd

# Data samples 10,000
np.random.seed(42)
n_samples = 10000

# Features generate
water_level_pct = np.random.uniform(30, 100, n_samples)  # 30% to 100%
rainfall_mm = np.random.uniform(0, 250, n_samples)  # 0 to 250mm
dam_age_years = np.random.uniform(5, 80, n_samples)  # 5 to 80 years
seepage_rate_lpm = np.random.uniform(0.1, 15.0, n_samples)  # Seepage rate
gate_open_pct = np.random.uniform(0, 100, n_samples)  # Gates open %

# Target Variable (Risk Score calculation logic based on domain knowledge)
risk_score = (
    (water_level_pct * 0.35)
    + (rainfall_mm * 0.25)
    + (seepage_rate_lpm * 2.5)
    + (dam_age_years * 0.15)
    - (gate_open_pct * 0.10)
)

# Normalize Risk Score between 0 and 100
risk_score = (risk_score - risk_score.min()) / (
    risk_score.max() - risk_score.min()
) * 100

# Create Dataframe
df = pd.DataFrame(
    {
        "water_level_pct": np.round(water_level_pct, 2),
        "rainfall_mm": np.round(rainfall_mm, 2),
        "dam_age_years": np.round(dam_age_years, 1),
        "seepage_rate_lpm": np.round(seepage_rate_lpm, 2),
        "gate_open_pct": np.round(gate_open_pct, 1),
        "risk_score_pct": np.round(risk_score, 2),
    }
)

# Save as CSV file
df.to_csv("dam_risk_dataset.csv", index=False)
print("Successfully generated 'dam_risk_dataset.csv' with 10,000 rows!")