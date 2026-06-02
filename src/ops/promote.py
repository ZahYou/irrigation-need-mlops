"""Champion/Challenger promotion CLI.

Manages @champion and @previous aliases on the irrigation-need-classifier
registered model. Run after compare_models.py recommends PROMOTE.

Examples:
    python -m src.ops.promote --status                # show current aliases
    python -m src.ops.promote --to-champion 7         # V7 (must be @challenger) -> @champion
    python -m src.ops.promote --rollback              # swap @champion <-> @previous
"""

import argparse
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

MODEL_NAME = "irrigation-need-classifier"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    return MlflowClient()


def _aliases(client: MlflowClient) -> dict[str, str]:
    #info = client.get_registered_model(MODEL_NAME)
    #return {a.alias: a.version for a in info.aliases}

    return dict(client.get_registered_model(MODEL_NAME).aliases)  # alias -> version


def _smoke_predict(version: str) -> None:
    uri = f"models:/{MODEL_NAME}/{version}"
    print(f"  smoke loading {uri} ...")
    model = mlflow.pyfunc.load_model(uri)
    row = pd.DataFrame([{
        "Soil_pH": 6.5, "Soil_Moisture": 25.0, "Organic_Carbon": 1.5,
        "Electrical_Conductivity": 0.8, "Temperature_C": 35.0, "Humidity": 40.0,
        "Rainfall_mm": 5.0, "Sunlight_Hours": 8.0, "Wind_Speed_kmh": 10.0,
        "Field_Area_hectare": 2.5, "Previous_Irrigation_mm": 10.0,
        "Soil_Type": "Loamy", "Crop_Type": "Rice", "Crop_Growth_Stage": "Flowering",
        "Season": "Kharif", "Irrigation_Type": "Drip", "Water_Source": "Borewell",
        "Mulching_Used": "No", "Region": "South",
    }])
    out = model.predict(row)
    print(f"  smoke prediction OK: {out.iloc[0]['predicted_class']}")


def status() -> None:
    aliases = _aliases(_client())
    print(f"Registered model: {MODEL_NAME}")
    for alias in ("champion", "challenger", "previous"):
        v = aliases.get(alias, "(unset)")
        print(f"  @{alias:<10} -> V{v}")


def promote(target_version: str, force: bool = False) -> None:
    client = _client()
    aliases = _aliases(client)
    current_champ = aliases.get("champion")
    current_chall = aliases.get("challenger")

    if target_version == current_champ:
        raise SystemExit(f"V{target_version} is already @champion. Nothing to do.")
    if not force and target_version != current_chall:
        raise SystemExit(
            f"Refusing: V{target_version} is not the current @challenger "
            f"(@challenger = V{current_chall}). Use --force to override."
        )

    _smoke_predict(target_version)

    if current_champ:
        client.set_registered_model_alias(MODEL_NAME, "previous", current_champ)
        print(f"  @previous  -> V{current_champ}  (old champion preserved for rollback)")
    client.set_registered_model_alias(MODEL_NAME, "champion", target_version)
    print(f"  @champion  -> V{target_version}")
    print("Promotion complete. Restart the serving app to pick up the new champion.")


def rollback() -> None:
    client = _client()
    aliases = _aliases(client)
    current_champ = aliases.get("champion")
    current_prev = aliases.get("previous")
    if not current_prev:
        raise SystemExit("No @previous alias set. Nothing to roll back to.")
    if not current_champ:
        raise SystemExit("No @champion alias set. Inconsistent state.")

    _smoke_predict(current_prev)

    client.set_registered_model_alias(MODEL_NAME, "champion", current_prev)
    client.set_registered_model_alias(MODEL_NAME, "previous", current_champ)
    print(f"  @champion  -> V{current_prev}  (was @previous)")
    print(f"  @previous  -> V{current_champ}  (was @champion)")
    print("Restart the serving app to pick up the reverted champion.")


def main() -> None:
    p = argparse.ArgumentParser(description="Champion/Challenger promotion CLI")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--to-champion", type=str, metavar="VERSION", help="Promote VERSION to @champion")
    g.add_argument("--rollback", action="store_true", help="Swap @champion <-> @previous")
    g.add_argument("--status", action="store_true", help="Show current aliases")
    p.add_argument("--force", action="store_true", help="Skip 'must be @challenger' check")
    args = p.parse_args()

    if args.status:
        status()
    elif args.rollback:
        rollback()
    else:
        promote(args.to_champion, force=args.force)


if __name__ == "__main__":
    main()
