
"""API tests for src/serving/app.py.

The app loads its model from the MLflow registry in `lifespan`. That registry
is not available in CI, so we patch `mlflow.pyfunc.load_model` with a FakeModel
and test our own code: routing, request validation, response shape, errors.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.serving.app as app_module
from src.serving.app import app

# A complete, valid request (mirrors register_ensemble.py's input_example).
VALID_REQUEST = {
    "Soil_pH": 6.5, "Soil_Moisture": 25.0, "Organic_Carbon": 1.5,
    "Electrical_Conductivity": 0.8, "Temperature_C": 35.0, "Humidity": 40.0,
    "Rainfall_mm": 5.0, "Sunlight_Hours": 8.0, "Wind_Speed_kmh": 10.0,
    "Field_Area_hectare": 2.5, "Previous_Irrigation_mm": 10.0,
    "Soil_Type": "Loamy", "Crop_Type": "Rice", "Crop_Growth_Stage": "Flowering",
    "Season": "Kharif", "Irrigation_Type": "Drip", "Water_Source": "Borewell",
    "Mulching_Used": "No", "Region": "South",
}


class FakeModel:
    """Stand-in for the registry pyfunc: returns a fixed prediction frame."""

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame)  # app must pass a DataFrame
        return pd.DataFrame([{
            "predicted_class": "High",
            "Low": 0.004464, "Medium": 0.426935, "High": 0.568601,
        }])


@pytest.fixture
def client():
    """TestClient whose lifespan loads a FakeModel instead of the real registry."""
    with patch("mlflow.pyfunc.load_model", return_value=FakeModel()):
        with TestClient(app) as c:   # entering the context triggers lifespan
            yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["loaded"] is True
    assert body["model_uri"] == "models:/irrigation-need-classifier@champion"


def test_predict_happy_path(client):
    resp = client.post("/predict", json=VALID_REQUEST)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] == "High"
    assert set(body["probabilities"]) == {"Low", "Medium", "High"}
    assert body["probabilities"]["High"] == 0.568601          # passed through, rounded
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_predict_rejects_bad_type(client):
    bad = {**VALID_REQUEST, "Soil_Moisture": "not-a-number"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422   # pydantic rejects an uncoercible float


def test_predict_accepts_partial_input(client):
    # All fields are Optional by design, so a sparse body is still valid.
    resp = client.post("/predict", json={"Soil_Moisture": 25.0})
    assert resp.status_code == 200


def test_predict_handles_model_error(client):
    class Boom:
        def predict(self, df):
            raise RuntimeError("kaboom")

    app_module._state["model"] = Boom()   # swap in a model that fails
    resp = client.post("/predict", json=VALID_REQUEST)
    assert resp.status_code == 422
    assert "Prediction failed" in resp.json()["detail"]


def test_predict_503_when_model_not_loaded():
    # No `with` => lifespan never runs => _state has no model.
    app_module._state.clear()
    plain = TestClient(app)
    resp = plain.post("/predict", json=VALID_REQUEST)
    assert resp.status_code == 503
