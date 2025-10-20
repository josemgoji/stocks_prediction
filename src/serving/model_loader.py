"""Carga modelos registrados en MLflow para uso en batch o APIs."""

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


@dataclass(slots=True)
class LoadedModel:
    """Representa un modelo cargado desde MLflow junto con su metadata básica."""

    pipeline: Any
    model_uri: str
    version: str
    run_id: str | None


def load_latest_model_version(
    model_name: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> LoadedModel:
    """Carga la última versión registrada de un modelo."""

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No hay versiones registradas para el modelo '{model_name}'.")

    latest = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{model_name}/{latest.version}"
    pipeline = mlflow.pyfunc.load_model(model_uri)
    run_id = getattr(latest, "run_id", None)
    return LoadedModel(
        pipeline=pipeline,
        model_uri=model_uri,
        version=str(latest.version),
        run_id=run_id,
    )
