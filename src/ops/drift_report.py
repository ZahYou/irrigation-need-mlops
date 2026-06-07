"""Evidently data-drift report: reference (training sample) vs current (live traffic).

Compares monitoring/reference_data.csv against logs/predictions.csv and writes
an HTML report + JSON summary to reports/.

Exit code is the machine-readable verdict (so a script/cron can branch on it):
    0 = no drift          2 = dataset drift detected
    1 = the job itself failed (Python's default on an uncaught exception)

Run:  python -m src.ops.drift_report
"""

import sys
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

REFERENCE_CSV = Path("monitoring/reference_data.csv")
CURRENT_CSV = Path("logs/predictions.csv")
OUT_HTML = Path("reports/drift_report.html")
OUT_JSON = Path("reports/drift_report.json")

# Fraction of columns that must drift before we call the WHOLE dataset drifted.
# Evidently flags each column individually; this is the aggregate-verdict threshold.
DRIFT_SHARE_THRESHOLD = 0.5

# Exit codes — the orchestrator (Fork 1) branches on these. We deliberately do NOT
# use 1 for drift: an uncaught exception already exits 1, so reserving it for
# "the job broke" keeps "drift" (2) distinct from "tool failure" (1).
EXIT_NO_DRIFT = 0
EXIT_DRIFT = 2

NUMERIC = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
CATEGORICAL = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
FEATURES = NUMERIC + CATEGORICAL


def main() -> int:
    """Run the drift report and RETURN an exit code (0 = no drift, 2 = drift)."""
    reference = pd.read_csv(REFERENCE_CSV)[FEATURES]
    current = pd.read_csv(CURRENT_CSV)[FEATURES]

    # tell Evidently which columns are numeric vs categorical
    definition = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL)
    ref_ds = Dataset.from_pandas(reference, data_definition=definition)
    cur_ds = Dataset.from_pandas(current, data_definition=definition)

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(OUT_HTML))
    snapshot.save_json(str(OUT_JSON))

    # pull the dataset-level drift metric out of the result dict
    result = snapshot.dict()
    summary = next(
        (m for m in result["metrics"] if m["metric_name"].startswith("DriftedColumnsCount")),
        None,
    )
    if summary is None:
        # We couldn't find the metric -> we cannot make a decision. Raising here
        # gives Python's exit code 1 ("job failed"), which is correctly NOT 2 (drift).
        raise RuntimeError("DriftedColumnsCount metric not found in Evidently result")

    count = int(summary["value"]["count"])
    share = summary["value"]["share"]
    drift = share >= DRIFT_SHARE_THRESHOLD

    print(f"Drifted columns: {count} / {len(FEATURES)}  (share={share:.1%})")
    print("==> DATASET DRIFT DETECTED" if drift else "==> No dataset-level drift")
    print(f"Report written to: {OUT_HTML}")

    # hand the verdict back to the shell / orchestrator as the process exit code
    return EXIT_DRIFT if drift else EXIT_NO_DRIFT


if __name__ == "__main__":
    sys.exit(main())  # propagate main()'s return value as the process exit code
