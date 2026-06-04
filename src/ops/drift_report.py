"""Evidently data-drift report: reference (training sample) vs current (live traffic).

Compares monitoring/reference_data.csv against logs/predictions.csv and writes
an HTML report + JSON summary to reports/.

Run:  python -m src.ops.drift_report
"""

from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

REFERENCE_CSV = Path("monitoring/reference_data.csv")
CURRENT_CSV = Path("logs/predictions.csv")
OUT_HTML = Path("reports/drift_report.html")
OUT_JSON = Path("reports/drift_report.json")

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


def main() -> None:
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

    # short text summary from the result dict
    result = snapshot.dict()
    summary = next(
        (m for m in result["metrics"] if m["metric_name"].startswith("DriftedColumnsCount")),
        None,
    )
    if summary:
        count = int(summary["value"]["count"])
        share = summary["value"]["share"]
        print(f"Drifted columns: {count} / {len(FEATURES)}  (share={share:.1%})")
        print("==> DATASET DRIFT DETECTED" if share >= 0.5 else "==> No dataset-level drift")
    print(f"Report written to: {OUT_HTML}")


if __name__ == "__main__":
    main()