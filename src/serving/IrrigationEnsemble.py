"""Custom MLflow pyfunc model wrapping the full irrigation ensemble.

Packages feature engineering + label encoding + all fold models + probability
averaging into ONE registry artifact, so serving can load the whole prediction
pipeline by alias: models:/irrigation-need-classifier@champion
"""

import pickle
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.data.feature_engineering import build_single_inference_features

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

_MODEL_PREFIXES = {
"xgb": "xgb_baseline_v001",
"lgbm": "lgbm_baseline_v001",
"catboost": "cat_baseline_v001",
"logreg": "logreg_baseline_v001",
}


class IrrigationEnsemble(mlflow.pyfunc.PythonModel):

    """Averages predict_proba across every fold model of every enabled algorithm."""

    def load_context(self, context):
        """Runs ONCE when MLflow loads the model. Restores everything from artifacts."""
        artifacts = context.artifacts

        # Feature-engineering lookup tables (built during the features stage)
        with open(artifacts["feature_artifacts"], "rb") as f:
            self.artifacts = pickle.load(f)

        # Label encoders for the XGB/LogReg integer-encoded variant
        self.label_encoders = {}
        le_path = artifacts.get("label_encoders")
        if le_path and Path(le_path).exists():
            with open(le_path, "rb") as f:
                self.label_encoders = pickle.load(f)

        self.cat_cols = self.artifacts["cat_cols"]

        # Load all fold models, grouped by algorithm
        models_dir = Path(artifacts["models_dir"])
        self.models = {k: [] for k in _MODEL_PREFIXES}
        for key, prefix in _MODEL_PREFIXES.items():
            for fold_path in sorted(models_dir.glob(f"{prefix}_fold*.pkl")):
                with open(fold_path, "rb") as f:
                    self.models[key].append(pickle.load(f))

    # ── the 3 feature variants (identical to app.py) ──────────────
    def _label_encode(self, features):
        df = features.copy()
        for col in self.cat_cols:
            if col not in df.columns:
                continue
            le = self.label_encoders.get(col)
            if le is None:
                df[col] = 0
                continue
            known = set(le.classes_)
            fallback = len(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda v, _le=le, _k=known, _fb=fallback:
                    int(_le.transform([v])[0]) if v in _k else _fb
            )
        return df

    def _to_category_dtype(self, features):
        df = features.copy()
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def _predict_one(self, row: dict) -> np.ndarray:
        """Run the full ensemble for a single raw input row → 3-class proba vector."""
        features = build_single_inference_features(row, self.artifacts)
        le_features = self._label_encode(features)
        cat_features = self._to_category_dtype(features)

        all_probas = []
        for model in self.models["xgb"] + self.models["logreg"]:
            all_probas.append(model.predict_proba(le_features))
        for model in self.models["lgbm"]:
            all_probas.append(model.predict_proba(cat_features))
        for model in self.models["catboost"]:
            all_probas.append(model.predict_proba(features))

        if not all_probas:
            raise RuntimeError("No fold models loaded")
        return np.mean(all_probas, axis=0)[0]  # shape (3,)

    def predict(self, context, model_input, params=None):
        """MLflow's entry point. model_input = DataFrame, one raw request per row."""
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])

        out = []
        for row in model_input.to_dict(orient="records"):
            proba = self._predict_one(row)
            out.append({
                "predicted_class": LABEL_MAP[int(np.argmax(proba))],
                "Low": float(proba[0]),
                "Medium": float(proba[1]),
                "High": float(proba[2]),
            })
        return pd.DataFrame(out)


