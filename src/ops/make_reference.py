"""Build the drift-reference dataset: a fixed sample of the training data.

Evidently compares live traffic (logs/predictions.csv) against this reference,
which represents the feature distribution the model was trained on.
"""

from pathlib import Path

import pandas as pd

TRAIN_CSV = Path("data/raw/train.csv")
OUT_CSV = Path("monitoring/reference_data.csv")
N = 2000
SEED = 42

# the 19 raw features the API accepts/logs (same names as PredictRequest in app.py)
FEATURES = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    sample = df.sample(n=min(N, len(df)), random_state=SEED)[FEATURES]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(sample)} reference rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
