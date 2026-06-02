"""Champion vs Challenger comparison gate.

Loads both registry aliases, predicts on a stratified holdout of the training
set, computes metrics for each, and writes a decision report. Run before
promoting @challenger to @champion.
"""

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "irrigation-need-classifier"
TARGET = "Irrigation_Need"
#HOLDOUT_FRAC = 0.2
HOLDOUT_N = 500  # fixed-size holdout, for more stable comparisons across runs
                    # model predicts per-row (FE pipeline is per-request);
                    # batch eval scales linearly. 500 is plenty for the gate.
SEED = 42
PROMOTE_MARGIN = 0.001  # min BA improvement to recommend promotion
REPORTS_DIR = Path("reports")


def _load_holdout() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("data/raw/train.csv")
    _, holdout = train_test_split(
        df, test_size=HOLDOUT_N, stratify=df[TARGET], random_state=SEED
    )
    X = holdout.drop(columns=[TARGET, "id"], errors="ignore")
    y = holdout[TARGET].reset_index(drop=True)
    return X.reset_index(drop=True), y


def _predict(alias: str, X: pd.DataFrame) -> np.ndarray:
    uri = f"models:/{MODEL_NAME}@{alias}"
    print(f"Loading {uri} ...")
    model = mlflow.pyfunc.load_model(uri)
    return model.predict(X)["predicted_class"].to_numpy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = ["Low", "Medium", "High"]
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "high_recall": float(per_class_recall[2]),
    }


def main() -> None:
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )

    X, y = _load_holdout()
    print(f"Holdout: {len(X)} rows (stratified {HOLDOUT_N}, seed={SEED})")

    champ_pred = _predict("champion", X)
    chall_pred = _predict("challenger", X)

    y_arr = y.to_numpy()
    champ = _metrics(y_arr, champ_pred)
    chall = _metrics(y_arr, chall_pred)

    delta_ba = chall["balanced_accuracy"] - champ["balanced_accuracy"]
    agreement = float(np.mean(champ_pred == chall_pred))
    promote = delta_ba > PROMOTE_MARGIN

    report = {
        "model_name": MODEL_NAME,
        "holdout_rows": int(len(X)),
        "champion": champ,
        "challenger": chall,
        "delta_balanced_accuracy": float(delta_ba),
        "agreement_rate": agreement,
        "promote_margin": PROMOTE_MARGIN,
        "decision": "PROMOTE challenger" if promote else "HOLD champion",
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "champion_vs_challenger.json"
    out_path.write_text(json.dumps(report, indent=2))

    print("\n--- COMPARISON ---")
    print(f"Champion   BA={champ['balanced_accuracy']:.4f}  macroF1={champ['macro_f1']:.4f}  HighRec={champ['high_recall']:.4f}")
    print(f"Challenger BA={chall['balanced_accuracy']:.4f}  macroF1={chall['macro_f1']:.4f}  HighRec={chall['high_recall']:.4f}")
    print(f"Delta BA:       {delta_ba:+.4f}")
    print(f"Agreement rate: {agreement:.2%}")
    print(f"Decision:       {report['decision']}")
    print(f"Report saved:   {out_path}")


if __name__ == "__main__":
    main()
