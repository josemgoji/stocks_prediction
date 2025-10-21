"""Clientes auxiliares para interactuar con MLflow."""

import os
from collections.abc import Mapping
from typing import Any

import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient


class MLflowTracker:
    """Cliente MLflow compatible con los métodos usados en el entrenamiento."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        default_tags: Mapping[str, Any] | None = None,
        run_name: str | None = None,
    ) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_registry_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        self._default_tags = dict(default_tags or {})
        self._parent_run = None
        self._run_name = run_name or "model_selection"
        self._client = MlflowClient()

    def start_parent_run(self) -> mlflow.ActiveRun:
        """Abre un run principal para agrupar los candidatos."""
        if self._parent_run is not None:
            return self._parent_run

        self._parent_run = mlflow.start_run(run_name=self._run_name)
        if self._default_tags:
            mlflow.set_tags(self._default_tags)
        return self._parent_run

    def end_parent_run(self) -> None:
        """Cierra el run principal."""
        if self._parent_run is None:
            return
        mlflow.end_run()
        self._parent_run = None

    def start_run(self, run_name: str | None = None) -> mlflow.ActiveRun:
        """Inicia un run (anidado si existe run principal)."""
        active = self._parent_run is not None
        run = mlflow.start_run(run_name=run_name, nested=active)
        if self._default_tags:
            mlflow.set_tags(self._default_tags)
        return run

    def end_run(self) -> None:
        mlflow.end_run()

    def log_params(self, params: Mapping[str, Any]) -> None:
        if params:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        if metrics:
            mlflow.log_metrics(metrics)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if tags:
            mlflow.set_tags(tags)

    def log_dict(self, dictionary: dict, artifact_file: str) -> None:
        mlflow.log_dict(dictionary, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        mlflow.log_text(text, artifact_file)

    def log_model(
        self,
        model: Any,
        name: str,
        *,
        signature: Any | None = None,
        input_example: Any | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> str:
        """Loguea un modelo sklearn y retorna el URI resultante."""
        model_info = mlflow_sklearn.log_model(
            model,
            signature=signature,
            input_example=input_example,
            params=dict(params) if params else None,
            tags=dict(tags) if tags else None,
            name=name,
        )
        return model_info.model_uri

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Mapping[str, Any] | None = None,
    ) -> Any:
        """Registra un modelo en el Model Registry y aplica tags opcionales."""
        result = mlflow.register_model(model_uri, name)
        if tags:
            for key, value in tags.items():
                self._client.set_registered_model_tag(name, key, str(value))
        return result

    def ensure_artifact_bucket(self, bucket_name: str, config: Mapping[str, Any] | None = None) -> None:
        """Asegura que el bucket S3 existe para artefactos."""
        if not bucket_name:
            return
            
        try:
            import boto3
            from botocore.config import Config
            
            # Usar configuración centralizada
            s3_config = _get_s3_config(config)
            
            s3_resource = boto3.resource("s3", **s3_config)
            bucket = s3_resource.Bucket(str(bucket_name))
            try:
                bucket.load()
            except Exception:
                bucket.create()
        except Exception as exc:
            print(f"[mlflow] No se pudo asegurar el bucket '{bucket_name}': {exc}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MLflowTracker":
        """Factory method para crear tracker desde configuración YAML."""
        if not config:
            raise ValueError("Configuración MLflow requerida")
        if not config.get("enabled", True):
            raise ValueError("MLflow debe estar habilitado")
        
        tracking_uri = config.get("tracking_uri")
        if not tracking_uri:
            raise ValueError("tracking_uri es requerido")
        
        configure_mlflow_environment(config)
        
        tracker = cls(
            tracking_uri=tracking_uri,
            experiment_name=config.get("experiment_name"),
            default_tags=config.get("tags") or {},
            run_name=config.get("run_name", "model_selection"),
        )
        
        if artifact_bucket := config.get("artifact_bucket"):
            tracker.ensure_artifact_bucket(artifact_bucket, config)
        
        return tracker


def _get_s3_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extrae configuración S3 de manera centralizada."""
    if not config:
        return {}
    
    s3_config = {}
    if s3_endpoint := config.get("s3_endpoint_url"):
        s3_config["endpoint_url"] = str(s3_endpoint)
    if region_name := config.get("aws_default_region"):
        s3_config["region_name"] = str(region_name)
    if config.get("aws_s3_force_path_style"):
        from botocore.config import Config
        s3_config["config"] = Config(s3={"addressing_style": "path"})
    
    return s3_config


def configure_mlflow_environment(config: Mapping[str, Any] | None) -> None:
    """Aplica variables de entorno para MLflow/S3 a partir de configuración declarativa."""
    if not config:
        return

    # Mapeo de configuración a variables de entorno
    env_mapping = {
        "s3_endpoint_url": "MLFLOW_S3_ENDPOINT_URL",
        "aws_access_key_id": "AWS_ACCESS_KEY_ID", 
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_session_token": "AWS_SESSION_TOKEN",
        "aws_default_region": "AWS_DEFAULT_REGION",
        "aws_s3_force_path_style": "AWS_S3_FORCE_PATH_STYLE",
    }
    
    # Aplicar configuraciones
    for config_key, env_var in env_mapping.items():
        value = config.get(config_key)
        if value is not None:
            if config_key == "aws_s3_force_path_style":
                os.environ[env_var] = "true"
            elif config_key == "aws_default_region":
                os.environ.setdefault(env_var, str(value))
            else:
                os.environ[env_var] = str(value)
