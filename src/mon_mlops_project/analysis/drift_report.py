from pathlib import Path
import pandas as pd


import numpy as np
from sklearn.datasets import fetch_california_housing

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

def main() -> None:
    # 1) Données "train" (référence)
    data = fetch_california_housing(as_frame=True)
    ref_df = data.frame.rename(columns={"MedHouseVal": "target"})  
    feature_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

    ref = ref_df[feature_cols].copy()

    # 2) Données "prod" simulées (on crée un drift artificiel)
    prod = ref.sample(frac=0.5, random_state=42).copy()
    prod["MedInc"] = prod["MedInc"] * 1.20          # drift (revenu médian plus élevé)
    prod["HouseAge"] = prod["HouseAge"] + 5         # drift (maisons plus vieilles)
    prod["Latitude"] = prod["Latitude"] + 0.2       # drift (géographie décalée)
    prod["Longitude"] = prod["Longitude"] - 0.2

    # 3) Schéma Evidently
    schema = DataDefinition(
        numerical_columns=feature_cols,
        categorical_columns=[],
    )

    ref_ds = Dataset.from_pandas(ref, data_definition=schema)
    prod_ds = Dataset.from_pandas(prod, data_definition=schema)

    # 4) Report drift
    report = Report([DataDriftPreset()])
    evaluation = report.run(prod_ds, ref_ds)

    # 5) Export HTML
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "drift_report.html"
    evaluation.save_html(str(out_path))
    print(f"Drift report généré: {out_path.resolve()}")


if __name__ == "__main__":
    main()