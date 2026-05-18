"""FastAPI serving app for the Irrigation Need classifier."""

import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.feature_engineering import build_single_inference_features

SAVED_MODELS_DIR = Path("saved_models")
PROCESSED_DIR = Path("data/processed")

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

_MODEL_PREFIXES = {
    "xgb": "xgb_baseline_v001",
    "lgbm": "lgbm_baseline_v001",
    "catboost": "cat_baseline_v001",
    "logreg": "logreg_baseline_v001",
}

_state: dict = {}


def _load_fold_models() -> dict[str, list]:
    models: dict[str, list] = {k: [] for k in _MODEL_PREFIXES}
    for model_key, prefix in _MODEL_PREFIXES.items():
        for fold_path in sorted(SAVED_MODELS_DIR.glob(f"{prefix}_fold*.pkl")):
            with open(fold_path, "rb") as f:
                models[model_key].append(pickle.load(f))
    return models


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts_path = PROCESSED_DIR / "feature_artifacts.pkl"
    if not artifacts_path.exists():
        raise RuntimeError(
            f"Feature artifacts not found at {artifacts_path}. Run `dvc repro` first."
        )

    with open(artifacts_path, "rb") as f:
        _state["artifacts"] = pickle.load(f)

    label_enc_path = SAVED_MODELS_DIR / "label_encoders.pkl"
    _state["label_encoders"] = {}
    if label_enc_path.exists():
        with open(label_enc_path, "rb") as f:
            _state["label_encoders"] = pickle.load(f)

    _state["models"] = _load_fold_models()
    total = sum(len(v) for v in _state["models"].values())
    print(f"Loaded {total} fold model(s): { {k: len(v) for k, v in _state['models'].items()} }")

    yield
    _state.clear()


app = FastAPI(
    title="Irrigation Need Classifier",
    description="3-class irrigation need prediction: Low / Medium / High",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    Soil_pH: Optional[float] = None
    Soil_Moisture: Optional[float] = None
    Organic_Carbon: Optional[float] = None
    Electrical_Conductivity: Optional[float] = None
    Temperature_C: Optional[float] = None
    Humidity: Optional[float] = None
    Rainfall_mm: Optional[float] = None
    Sunlight_Hours: Optional[float] = None
    Wind_Speed_kmh: Optional[float] = None
    Field_Area_hectare: Optional[float] = None
    Previous_Irrigation_mm: Optional[float] = None
    Soil_Type: Optional[str] = None
    Crop_Type: Optional[str] = None
    Crop_Growth_Stage: Optional[str] = None
    Season: Optional[str] = None
    Irrigation_Type: Optional[str] = None
    Water_Source: Optional[str] = None
    Mulching_Used: Optional[str] = None
    Region: Optional[str] = None


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    models_used: int


def _label_encode(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    label_encoders = _state["label_encoders"]
    for col in _state["artifacts"]["cat_cols"]:
        if col not in df.columns:
            continue
        le = label_encoders.get(col)
        if le is None:
            df[col] = 0
            continue
        known = set(le.classes_)
        fallback = len(le.classes_)
        df[col] = df[col].astype(str).apply(
            lambda v, _le=le, _k=known, _fb=fallback: int(_le.transform([v])[0]) if v in _k else _fb
        )
    return df


def _to_category_dtype(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    for col in _state["artifacts"]["cat_cols"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


@app.get("/health")
def health() -> dict:
    models = _state.get("models", {})
    return {
        "status": "ok",
        "models": {k: len(v) for k, v in models.items()},
        "total_fold_models": sum(len(v) for v in models.values()),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    artifacts = _state.get("artifacts")
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    row = req.model_dump()

    try:
        features = build_single_inference_features(row, artifacts)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {exc}") from exc

    le_features = _label_encode(features)
    cat_features = _to_category_dtype(features)

    all_probas: list[np.ndarray] = []
    models = _state["models"]

    for model in models["xgb"] + models["logreg"]:
        all_probas.append(model.predict_proba(le_features))

    for model in models["lgbm"]:
        all_probas.append(model.predict_proba(cat_features))

    for model in models["catboost"]:
        all_probas.append(model.predict_proba(features))

    if not all_probas:
        raise HTTPException(status_code=503, detail="No trained models found in saved_models/")

    ensemble_proba: np.ndarray = np.mean(all_probas, axis=0)[0]
    predicted_idx = int(np.argmax(ensemble_proba))

    return PredictResponse(
        predicted_class=LABEL_MAP[predicted_idx],
        probabilities={
            "Low": round(float(ensemble_proba[0]), 6),
            "Medium": round(float(ensemble_proba[1]), 6),
            "High": round(float(ensemble_proba[2]), 6),
        },
        models_used=len(all_probas),
    )
