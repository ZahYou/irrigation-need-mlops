"""FastAPI serving app for the Irrigation Need classifier.

Loads the full pipeline (feature engineering + encoders + 9-model ensemble)
from the MLflow Model Registry by alias. Swap models by reassigning @champion —
no code change, no logic redeploy.
"""

import csv
import os
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

MODEL_URI = os.getenv("MODEL_URI", "models:/irrigation-need-classifier@champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

_state: dict = {}
LOG_PATH = Path(os.getenv("PREDICTION_LOG", "logs/predictions.csv"))
_log_lock = threading.Lock()

def _log_prediction(features: dict, predicted_class: str, probs: dict[str, float]) -> None:
    """Append one request + prediction to the CSV log (thread-safe, best-effort)."""
    row = {
          "timestamp": datetime.now(UTC).isoformat(),
          **features,
          "predicted_class": predicted_class,
          **probs,
      }

    try :
        with _log_lock:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            write_header = not LOG_PATH.exists()
            with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    except Exception as exc:
        print(f"[warn] prediction logging failed: {exc}")


PREDICTIONS = Counter("predictions_total",
                    "Count of predictions by predicted class",
                    ["predicted_class"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"Loaded model: {MODEL_URI}")
    yield
    _state.clear()


app = FastAPI(
    title="Irrigation Need Classifier",
    description="3-class irrigation need prediction: Low / Medium / High",
    version="0.2.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


class PredictRequest(BaseModel):
    Soil_pH: float | None = None
    Soil_Moisture: float | None = None
    Organic_Carbon: float | None = None
    Electrical_Conductivity: float | None = None
    Temperature_C: float | None = None
    Humidity: float | None = None
    Rainfall_mm: float | None = None
    Sunlight_Hours: float | None = None
    Wind_Speed_kmh: float | None = None
    Field_Area_hectare: float | None = None
    Previous_Irrigation_mm: float | None = None
    Soil_Type: str | None = None
    Crop_Type: str | None = None
    Crop_Growth_Stage: str | None = None
    Season: str | None = None
    Irrigation_Type: str | None = None
    Water_Source: str | None = None
    Mulching_Used: str | None = None
    Region: str | None = None


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_uri": MODEL_URI, "loaded": "model" in _state}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([req.model_dump()])
    try:
        result = model.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    row = result.iloc[0]
    pred_class = str(row["predicted_class"])
    PREDICTIONS.labels(predicted_class=pred_class).inc()

    probs={
            "Low": round(float(row["Low"]), 6),
            "Medium": round(float(row["Medium"]), 6),
            "High": round(float(row["High"]), 6),
        }
    
    _log_prediction(req.model_dump(), pred_class, probs)

    return PredictResponse(
        predicted_class=pred_class,
        probabilities=probs,
    )

