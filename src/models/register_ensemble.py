"""One-off: log the IrrigationEnsemble pyfunc and register it in the Model Registry."""

import os

import mlflow
import pandas as pd

from src.serving.IrrigationEnsemble import IrrigationEnsemble

#mlflow.set_tracking_uri("sqlite:///mlflow.db")          # the SQLite backend you set up
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("irrigation-need-prod")

# A realistic single row → lets MLflow infer & store the input/output schema
input_example = pd.DataFrame([{
"Soil_pH": 6.5, "Soil_Moisture": 25.0, "Organic_Carbon": 1.5,
"Electrical_Conductivity": 0.8, "Temperature_C": 35.0, "Humidity": 40.0,
"Rainfall_mm": 5.0, "Sunlight_Hours": 8.0, "Wind_Speed_kmh": 10.0,
"Field_Area_hectare": 2.5, "Previous_Irrigation_mm": 10.0,
"Soil_Type": "Loamy", "Crop_Type": "Rice", "Crop_Growth_Stage": "Flowering",
"Season": "Kharif", "Irrigation_Type": "Drip", "Water_Source": "Borewell",
"Mulching_Used": "No", "Region": "South",
}])

with mlflow.start_run(run_name="ensemble_v001"):
    info = mlflow.pyfunc.log_model(
    name="irrigation_ensemble",
    python_model=IrrigationEnsemble(),
    artifacts={
        "feature_artifacts": "data/processed/feature_artifacts.pkl",
        "label_encoders":    "saved_models/label_encoders.pkl",
        "models_dir":        "saved_models",
    },
    code_paths=["src"],
    input_example=input_example,
    registered_model_name="irrigation-need-classifier",
)
print("Logged model URI:", info.model_uri)
