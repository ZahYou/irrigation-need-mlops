"""
Evaluate stage — measures model quality on OOF predictions and writes reports.

Why this stage exists
---------------------
After the train stage we have raw numpy OOF arrays on disk, but no structured
answer to "how good is this pipeline?"  This stage turns those arrays into:

  1. reports/metrics.json  — machine-readable, DVC-tracked so `dvc metrics diff`
                             can compare any two commits side-by-side.
  2. reports/cm_<name>.png — normalized confusion matrices so we can see *which*
                             classes are confused, not just the headline number.

Design decisions
----------------
- We evaluate on OOF predictions, not on test.  Each training row was scored by
  a fold model that never saw it, so OOF BA is an unbiased generalization estimate.

- Balanced accuracy is the primary metric because the High class is rare (~5 %).
  Plain accuracy would reward a model that always predicts Low.

- We also report per-class precision, recall, and F1 because the three irrigation
  levels have different real-world costs.  Missing a High label is more expensive
  than missing a Low one.

- The ensemble row averages probabilities across all models before argmax.  It is
  almost always the best single number and is what the /predict endpoint uses.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in CI and headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = ["Low", "Medium", "High"]
OOF_DIR     = Path("saved_oof")
REPORTS_DIR = Path("reports")


def _load_oof_probas() -> dict[str, np.ndarray]:
    """
    Discover every *_proba.npy file in saved_oof/ and return {model_name: array}.

    We discover files dynamically so the evaluate stage does not need to know
    which models were enabled in the config — it just evaluates whatever was
    produced by the train stage.
    """
    files = sorted(OOF_DIR.glob("*_proba.npy"))
    if not files:
        raise FileNotFoundError(
            f"No OOF files found in {OOF_DIR}.  Run the train stage first:\n"
            "  dvc repro train"
        )
    return {f.stem.replace("_proba", ""): np.load(f) for f in files}


def _compute_metrics(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Compute all scalar metrics for one set of OOF probabilities.

    Returns a flat dict so every key maps directly to a number — this is what
    DVC expects in metrics.json for `dvc metrics show` and `dvc metrics diff`.
    """
    y_pred = np.argmax(proba, axis=1)

    # classification_report with output_dict=True gives precision/recall/F1
    # for every class plus macro and weighted averages.
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics: dict = {
        "oof_balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 6),
        "macro_f1":              round(report["macro avg"]["f1-score"], 6),
        "weighted_f1":           round(report["weighted avg"]["f1-score"], 6),
    }

    # Per-class breakdown — crucial for spotting the rare High class being missed.
    for cls in CLASS_NAMES:
        key = cls.lower()
        metrics[f"{key}_precision"] = round(report[cls]["precision"], 6)
        metrics[f"{key}_recall"]    = round(report[cls]["recall"],    6)
        metrics[f"{key}_f1"]        = round(report[cls]["f1-score"],  6)

    return metrics


def _save_confusion_matrix(proba: np.ndarray, y_true: np.ndarray, name: str) -> None:
    """
    Save a row-normalized confusion matrix as a PNG.

    Normalization (normalize='true') shows recall per class — each cell is the
    fraction of actual-class rows predicted as each column class.  This makes the
    plot scale-invariant even when class sizes differ a lot.
    """
    y_pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(f"{name}  —  normalized confusion matrix", fontsize=11)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / f"cm_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # y_train is integer-encoded (0=Low 1=Medium 2=High) — produced by run_features.py
    y_true: np.ndarray = pd.read_pickle("data/processed/y_train.pkl").to_numpy()

    oof_probas = _load_oof_probas()
    log.info("Evaluating %d model(s): %s", len(oof_probas), list(oof_probas))

    all_metrics: dict[str, dict] = {}

    for name, proba in oof_probas.items():
        metrics = _compute_metrics(proba, y_true)
        all_metrics[name] = metrics
        _save_confusion_matrix(proba, y_true, name)

        log.info(
            "%-40s  BA=%.4f  macro_F1=%.4f  high_recall=%.4f",
            name,
            metrics["oof_balanced_accuracy"],
            metrics["macro_f1"],
            metrics["high_recall"],
        )

    # ── Ensemble ────────────────────────────────────────────────────────────────
    # Average the probability matrices across all models, then take argmax.
    # This is what the /predict endpoint does at serve time, so the ensemble row
    # in metrics.json is the most honest estimate of production performance.
    ensemble_proba = np.mean(list(oof_probas.values()), axis=0)
    ensemble_metrics = _compute_metrics(ensemble_proba, y_true)
    all_metrics["ensemble"] = ensemble_metrics
    _save_confusion_matrix(ensemble_proba, y_true, "ensemble")

    log.info(
        "%-40s  BA=%.4f  macro_F1=%.4f  high_recall=%.4f",
        "ensemble",
        ensemble_metrics["oof_balanced_accuracy"],
        ensemble_metrics["macro_f1"],
        ensemble_metrics["high_recall"],
    )

    # ── Write metrics.json ──────────────────────────────────────────────────────
    # DVC treats this file specially: `dvc metrics show` pretty-prints it and
    # `dvc metrics diff <branch>` compares it against any other git ref.
    # cache: false in dvc.yaml means the file is stored in git (as plain text),
    # not in the DVC content-addressable cache.
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    log.info("Wrote %s", metrics_path)
    log.info("Wrote confusion matrix PNGs to %s/", REPORTS_DIR)

    # Print a compact summary to stdout so it shows in `dvc repro` output.
    print("\n--- EVALUATION SUMMARY ---")
    header = f"  {'model':<40}  {'BA':>8}  {'macro_F1':>10}  {'high_recall':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in all_metrics.items():
        print(
            f"  {name:<40}  {m['oof_balanced_accuracy']:>8.4f}"
            f"  {m['macro_f1']:>10.4f}  {m['high_recall']:>12.4f}"
        )


if __name__ == "__main__":
    main()
