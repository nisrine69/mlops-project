import os
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "california_housing_price_prediction")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "CaliforniaHousingModel")
METRIC_NAME = os.getenv("MLFLOW_SELECTION_METRIC", "rmse")  # we want MIN rmse
TARGET_STAGE = os.getenv("MLFLOW_TARGET_STAGE", "Production")  # "Staging" or "Production"


def get_experiment_id(client: MlflowClient, experiment_name: str) -> str:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")
    return exp.experiment_id


def find_best_run_id(client: MlflowClient, experiment_id: str, metric_name: str) -> str:
    # order_by: ascending for rmse/mae, descending for r2
    if metric_name.lower() in {"r2", "r2_score"}:
        order_by = [f"metrics.{metric_name} DESC"]
    else:
        order_by = [f"metrics.{metric_name} ASC"]

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=order_by,
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No finished runs found in experiment_id={experiment_id}.")
    return runs[0].info.run_id


def get_model_version_from_run(client: MlflowClient, model_name: str, run_id: str) -> Optional[str]:
    # Search model versions and match by run_id
    versions = client.search_model_versions(f"name='{model_name}'")
    for mv in versions:
        if mv.run_id == run_id:
            return mv.version
    return None


def promote_version(client: MlflowClient, model_name: str, version: str, stage: str) -> None:
    # archive existing versions already in that stage
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )


def main() -> None:
    # Set tracking URI if you use a custom one; otherwise default local
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    experiment_id = get_experiment_id(client, EXPERIMENT_NAME)
    best_run_id = find_best_run_id(client, experiment_id, METRIC_NAME)

    best_version = get_model_version_from_run(client, MODEL_NAME, best_run_id)
    if best_version is None:
        raise RuntimeError(
            f"Could not find a model version for model='{MODEL_NAME}' linked to run_id='{best_run_id}'.\n"
            f"Tip: ensure your training used registered_model_name='{MODEL_NAME}'."
        )

    promote_version(client, MODEL_NAME, best_version, TARGET_STAGE)

    best_run = client.get_run(best_run_id)
    best_metric_value = best_run.data.metrics.get(METRIC_NAME)

    print(
        f" Promoted model '{MODEL_NAME}' version {best_version} to stage '{TARGET_STAGE}'.\n"
        f"Best run_id: {best_run_id}\n"
        f"{METRIC_NAME}={best_metric_value}"
    )


if __name__ == "__main__":
    main()
