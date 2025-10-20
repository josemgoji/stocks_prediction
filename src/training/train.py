"""Punto de entrada para entrenar modelos utilizando configuración YAML."""

from pathlib import Path
from typing import Any, Mapping, Sequence

from mlflow.models.signature import infer_signature

from src.models.estimators import create_estimator
from src.models.pipelines import PipelineConfig
from src.registry.mlflow_client import MLflowTracker, configure_mlflow_environment
from src.training.dataset import load_dataset, temporal_train_val_test_split
from src.training.model_selection import (
    CandidateDefinition,
    FeatureSelectionConfig,
    ModelSelectionOutcome,
    run_model_selection,
)
from src.training.hyperparameter_tuning import HyperparameterTuningConfig
from src.utils.io import load_yaml

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def run_training(
    *,
    data_config_path: Path,
    training_config_path: Path,
    tracker: Any | None = None,
) -> ModelSelectionOutcome:
    """Ejecuta la competencia de modelos completa."""
    data_cfg = load_yaml(data_config_path)
    training_cfg = load_yaml(training_config_path)

    dataset = load_dataset(data_cfg)

    target_column = training_cfg.get("target_column", "Close")
    split_cfg = training_cfg.get("split", {})
    horizon = int(training_cfg.get("horizon", 1))

    splits = temporal_train_val_test_split(
        dataset,
        target_column=target_column,
        train_size=float(split_cfg.get("train_size", 0.7)),
        val_size=float(split_cfg.get("val_size", 0.15)),
        test_size=float(split_cfg.get("test_size", 0.15)),
        gap=int(split_cfg.get("gap", 0)),
        horizon=horizon,
    )

    assembler_params = training_cfg.get("assembler", {})

    pipeline_config = PipelineConfig(
        target_column=target_column,
        assembler_params=assembler_params,
        extra_steps=_build_extra_steps(training_cfg.get("extra_steps")),
    )

    selection_cfg = training_cfg.get("feature_selection", {})
    selection_enabled = selection_cfg.get("enabled", True)

    selection_config = FeatureSelectionConfig(
        estimator=create_estimator(
            selection_cfg.get("estimator", "lasso"),
            selection_cfg.get("params"),
        ),
        n_splits=int(selection_cfg.get("n_splits", 5)),
        min_features_to_select=int(selection_cfg.get("min_features_to_select", 5)),
        step=selection_cfg.get("step", 1),
        scoring=selection_cfg.get("scoring"),
        gap=int(selection_cfg.get("gap", split_cfg.get("gap", 0))),
        min_samples_required=selection_cfg.get("min_samples_required"),
        correlation_threshold=selection_cfg.get("correlation_threshold"),
        artifact_dir=selection_cfg.get("artifact_dir"),
        artifact_name=selection_cfg.get("artifact_name", "selected_features.json"),
        tracking_artifact_path=selection_cfg.get("tracking_artifact_path", "feature_selection"),
    )

    candidates_cfg = training_cfg.get("candidates") or []
    if not candidates_cfg:
        candidates_cfg = [
            {"name": "linear_regression", "estimator": "linear_regression"},
        ]

    candidates = [
        _build_candidate_definition(candidate)
        for candidate in candidates_cfg
    ]

    tracker_cfg = training_cfg.get("mlflow")
    if tracker is None:
        tracker = _build_mlflow_tracker(tracker_cfg)

    parent_run_active = False
    if tracker is not None and hasattr(tracker, "start_parent_run"):
        tracker.start_parent_run()
        parent_run_active = True
        tracker.set_tags(
            {
                "pipeline_target": target_column,
                "selection_estimator": selection_config.estimator.__class__.__name__,
                "label_column": splits.label_column,
            }
        )
        tracker.log_params(
            {
                "train_size": split_cfg.get("train_size", 0.7),
                "val_size": split_cfg.get("val_size", 0.15),
                "test_size": split_cfg.get("test_size", 0.15),
                "split_gap": split_cfg.get("gap", 0),
                "horizon": horizon,
            }
        )

    outcome = run_model_selection(
        splits=splits,
        pipeline_config=pipeline_config,
        selection_config=selection_config if selection_enabled else FeatureSelectionConfig(
            estimator=create_estimator("linear_regression")
        ),
        candidates=candidates,
        tracker=tracker,
        primary_metric=training_cfg.get("primary_metric", "rmse"),
    )

    if tracker is not None:
        best = outcome.best_candidate
        signature = None
        input_example = None
        try:
            sample_input = splits.train_frame.tail(50)
            if sample_input.empty:
                sample_input = splits.train_frame.head(1)
            sample_input = sample_input.copy()

            label_column = getattr(splits, "label_column", None)

            sample_for_signature = sample_input.head(min(len(sample_input), 5)).copy()
            if label_column:
                sample_for_signature = sample_for_signature.drop(
                    columns=[label_column],
                    errors="ignore",
                )
            int_columns = sample_for_signature.select_dtypes(
                include=("int", "int64", "int32", "Int64")
            ).columns
            if len(int_columns) > 0:
                sample_for_signature = sample_for_signature.astype(
                    {column: "float64" for column in int_columns}
                )
            predictions_sample = best.pipeline.predict(sample_for_signature)
            signature = infer_signature(sample_for_signature, predictions_sample)
            input_example = sample_for_signature.head(1)
        except Exception as exc:  # noqa: BLE001
            print("[mlflow] No se pudo generar signature del modelo:", exc)

        tracker.set_tags({"best_candidate": best.name})
        tracker.log_params({"best_candidate": best.name})
        tracker.log_metrics(
            {f"best_val_{metric}": value for metric, value in best.metrics_val.items()}
        )
        tracker.log_metrics(
            {f"best_test_{metric}": value for metric, value in best.metrics_test.items()}
        )
        tracker.log_params(
            {"selected_features_count": len(outcome.feature_selection.selected_features)}
        )
        tracker.log_params({"prediction_horizon": horizon})
        try:
            tracker.log_dict(
                {"selected_features": list(outcome.feature_selection.selected_features)},
                "feature_selection/selected_features.json",
            )
        except Exception as exc:  # noqa: BLE001
            print("[mlflow] No se pudo registrar el artefacto de features:", exc)

        registry_cfg = training_cfg.get("model_registry", {})
        artifact_path_cfg = registry_cfg.get("artifact_path")
        model_name = registry_cfg.get("model_name")

        if artifact_path_cfg and hasattr(tracker, "log_model"):
            try:
                model_info = tracker.log_model(
                    best.pipeline,
                    artifact_path_cfg,
                    signature=signature,
                    input_example=input_example,
                )
                if model_name and hasattr(tracker, "register_model"):
                    tracker.register_model(
                        model_info,
                        model_name,
                        registry_cfg.get("tags"),
                    )
            except Exception as exc:  # noqa: BLE001
                print("[mlflow] No se pudo registrar el modelo automáticamente:", exc)

    if parent_run_active and tracker is not None and hasattr(tracker, "end_parent_run"):
        tracker.end_parent_run()

    return outcome


def _build_extra_steps(config: Sequence[Mapping[str, Any]] | None):
    """Convierte configuración declarativa en transformadores adicionales."""
    if not config:
        return None

    steps = []
    for idx, step_cfg in enumerate(config):
        kind = step_cfg.get("type")
        name = step_cfg.get("name", f"step_{idx}")
        if kind == "standard_scaler":
            steps.append((name, StandardScaler()))
        elif kind == "minmax_scaler":
            steps.append((name, MinMaxScaler()))
        else:
            raise ValueError(f"Tipo de paso extra no soportado: {kind}")
    return steps


def _build_candidate_definition(config: Mapping[str, Any]) -> CandidateDefinition:
    """Construye un candidato a partir de la configuración declarativa."""
    tuning_cfg = config.get("tuning")
    tuning = None
    if tuning_cfg:
        tuning = HyperparameterTuningConfig.from_mapping(tuning_cfg)
        if not tuning.param_grid:
            raise ValueError(
                f"El candidato '{config.get('name')}' tiene tuning habilitado sin 'param_grid'."
            )

    estimator_name = config.get("estimator", config["name"])
    params = config.get("params")
    return CandidateDefinition(
        name=str(config["name"]),
        estimator=create_estimator(
            estimator_name,
            params,
        ),
        extra_steps=_build_extra_steps(config.get("extra_steps")),
        tags=config.get("tags"),
        params=params,
        tuning=tuning,
    )


def _build_mlflow_tracker(config: Mapping[str, Any] | None) -> MLflowTracker:
    """Construye un tracker de MLflow; requiere configuración explícita."""
    if not config:
        raise ValueError(
            "Debes definir la sección 'mlflow' en training.yaml para ejecutar el entrenamiento."
        )
    if not config.get("enabled", True):
        raise ValueError("La sección 'mlflow' debe tener 'enabled: true'.")

    tracking_uri = config.get("tracking_uri")
    if not tracking_uri:
        raise ValueError("Define `mlflow.tracking_uri` en training.yaml para habilitar el tracking.")
    experiment_name = config.get("experiment_name")
    default_tags = config.get("tags") or {}
    run_name = config.get("run_name", "model_selection")

    configure_mlflow_environment(config)

    artifact_bucket = config.get("artifact_bucket")
    s3_endpoint = config.get("s3_endpoint_url")
    region_name = config.get("aws_default_region")
    force_path_style = config.get("aws_s3_force_path_style")

    if artifact_bucket:
        try:
            import boto3
            from botocore.config import Config

            resource_kwargs: dict[str, Any] = {}
            if s3_endpoint:
                resource_kwargs["endpoint_url"] = str(s3_endpoint)
            if region_name:
                resource_kwargs["region_name"] = str(region_name)
            if force_path_style:
                resource_kwargs["config"] = Config(s3={"addressing_style": "path"})

            s3_resource = boto3.resource("s3", **resource_kwargs)
            bucket = s3_resource.Bucket(str(artifact_bucket))
            try:
                bucket.load()
            except Exception:
                bucket.create()
        except Exception as exc:
            print(
                f"[mlflow] No se pudo asegurar el bucket de artefactos '{artifact_bucket}': {exc}"
            )

    return MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        default_tags=default_tags,
        run_name=run_name,
    )
