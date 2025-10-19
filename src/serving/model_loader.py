"""Carga modelos registrados en MLflow para uso en batch o APIs."""

from typing import Any

import mlflow


def load_registered_model(
    model_name: str,
    *,
    stage: str | None = "Latest",
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> Any:
    """Devuelve el pipeline almacenado en el Model Registry.

    Args:
        model_name: Nombre registrado en MLflow (p. ej. `StocksPredictionModel`).
        stage: Versión a resolver (`None` para versión numérica, `"Latest"` o un stage como `"Staging"`).
        tracking_uri: URI del tracking server si no está configurado globalmente.
        registry_uri: URI del registry (usualmente igual al tracking server).

    Returns:
        Instancia del modelo/pipeline listo para predecir.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    if stage is None:
        raise ValueError(
            "Debes especificar un `stage` ('Latest', 'Staging', 'Production', etc.) "
            "o reemplazar el método para cargar por versión explícita."
        )

    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)


def load_model_version(
    model_name: str,
    version: str | int,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> Any:
    """Carga una versión específica (`models:/name/version`)."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    model_uri = f"models:/{model_name}/{version}"
    return mlflow.pyfunc.load_model(model_uri)

