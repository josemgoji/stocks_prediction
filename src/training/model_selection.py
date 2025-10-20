"""Orquestación de selección de features y competencia de modelos."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.features.transformers import FeatureFrameAssembler
from src.models.pipelines import PipelineConfig, build_training_pipeline
from src.training.dataset import TemporalSplitResult
from src.training.feature_selection import (
    CorrelatedFeatureDropper,
    FeatureSelectionResult,
    TrackingClient,
    run_rfecv_selection,
)
from src.training.hyperparameter_tuning import (
    HyperparameterTuningConfig,
    run_hyperparameter_search,
)


@dataclass(slots=True)
class FeatureSelectionConfig:
    """Parámetros necesarios para ejecutar RFECV con TimeSeriesSplit."""

    estimator: BaseEstimator
    n_splits: int = 5
    min_features_to_select: int = 1
    step: int | float = 1
    scoring: str | None = None
    gap: int = 0
    artifact_dir: str | Path | None = None
    artifact_name: str = "selected_features.json"
    tracking_artifact_path: str = "feature_selection"
    min_samples_required: int | None = None
    correlation_threshold: float | None = None


@dataclass(slots=True)
class CandidateDefinition:
    """Representa un modelo candidato en la competencia."""

    name: str
    estimator: BaseEstimator
    extra_steps: Iterable[tuple[str, TransformerMixin]] | None = None
    tags: Mapping[str, Any] | None = None
    params: Mapping[str, Any] | None = None
    tuning: HyperparameterTuningConfig | None = None


@dataclass(slots=True)
class CandidateResult:
    """Resultado de entrenamiento y evaluación de un candidato."""

    name: str
    pipeline: Pipeline
    metrics_val: Mapping[str, float]
    metrics_test: Mapping[str, float]
    selected_features: tuple[str, ...]
    run_id: str | None = None
    params: Mapping[str, Any] | None = None


@dataclass(slots=True)
class ModelSelectionOutcome:
    """Resumen del proceso de selección de modelo."""

    feature_selection: FeatureSelectionResult
    candidates: list[CandidateResult]
    primary_metric: str = "rmse"

    @property
    def best_candidate(self) -> CandidateResult:
        """Retorna el candidato con mejor desempeño según `primary_metric`."""
        if not self.candidates:
            raise ValueError("No hay candidatos evaluados.")
        return min(self.candidates, key=lambda c: c.metrics_val[self.primary_metric])


def run_model_selection(
    splits: TemporalSplitResult,
    *,
    pipeline_config: PipelineConfig,
    selection_config: FeatureSelectionConfig,
    candidates: Sequence[CandidateDefinition],
    tracker: Any | None = None,
    primary_metric: str = "rmse",
) -> ModelSelectionOutcome:
    """Ejecuta selección de features y evalúa múltiples modelos candidatos."""
    if not candidates:
        raise ValueError("Debes proporcionar al menos un modelo candidato.")

    assembler_params = _build_assembler_params(pipeline_config)
    train_frame = splits.train_frame
    horizon = getattr(splits, "horizon", 1)
    label_column = getattr(splits, "label_column", None)

    assembler = FeatureFrameAssembler(**assembler_params)
    feature_matrix = assembler.fit_transform(train_frame).dropna()
    aligned_target = splits.y_train.loc[feature_matrix.index]
    train_subset = train_frame.loc[feature_matrix.index]

    dropped_correlated: tuple[str, ...] = ()
    if selection_config.correlation_threshold is not None:
        dropper = CorrelatedFeatureDropper(
            threshold=float(selection_config.correlation_threshold)
        )
        base_columns_in_matrix = tuple(
            column
            for column in train_subset.columns
            if column in feature_matrix.columns
        )
        feature_matrix, dropped_correlated = dropper.transform(
            feature_matrix,
            columns=base_columns_in_matrix,
        )
        if dropped_correlated:
            print(
                "[feature-selection] Columnas eliminadas por alta correlación "
                f"(>{selection_config.correlation_threshold}): {list(dropped_correlated)}"
            )

    usable_rows = len(feature_matrix)
    min_samples_required = selection_config.min_samples_required or (
        selection_config.n_splits + 1
    )

    if usable_rows <= selection_config.n_splits or usable_rows < min_samples_required:
        selection_result = _create_passthrough_selection_result(
            feature_matrix,
            selection_config.estimator.__class__.__name__,
            dropped_features=dropped_correlated,
        )
        print(
            "[feature-selection] RFECV omitido: muestras disponibles "
            f"{usable_rows} < requerido {min_samples_required}. "
            "Se usarán todas las features disponibles."
        )
    else:
        min_features = min(
            selection_config.min_features_to_select,
            feature_matrix.shape[1],
        )

        selection_result = run_rfecv_selection(
            feature_matrix,
            aligned_target,
            estimator=selection_config.estimator,
            n_splits=selection_config.n_splits,
            min_features_to_select=min_features,
            step=selection_config.step,
            scoring=selection_config.scoring,
            gap=selection_config.gap,
            artifact_dir=_to_path(selection_config.artifact_dir),
            artifact_name=selection_config.artifact_name,
            tracking_client=None,
            tracking_artifact_path=selection_config.tracking_artifact_path,
            pre_dropped_features=dropped_correlated,
        )

    if label_column is not None:
        train_features = train_subset.drop(columns=[label_column], errors="ignore")
    else:
        train_features = train_subset

    candidate_results: list[CandidateResult] = []
    for definition in candidates:
        run_id = None
        best_params: Mapping[str, Any] | None = definition.params
        if tracker is not None:
            active_run = tracker.start_run(run_name=definition.name)
            run_id = getattr(active_run, "info", getattr(active_run, "run_id", None))
            if hasattr(active_run, "info") and hasattr(active_run.info, "run_id"):
                run_id = active_run.info.run_id

            tracker.set_tags(
                {"candidate": definition.name, "estimator": definition.estimator.__class__.__name__}
            )
            if definition.tags:
                tracker.set_tags(definition.tags)
            if definition.params:
                tracker.log_params(definition.params)
            tracker.log_params(
                {
                    "feature_selection_estimator": selection_config.estimator.__class__.__name__,
                    "selected_features_count": len(selection_result.selected_features),
                    "horizon": horizon,
                }
            )
            if label_column is not None:
                tracker.set_tags({"label_column": label_column})

        candidate_pipeline = _build_candidate_pipeline(
            pipeline_config,
            definition,
            selection_result.selected_features,
        )

        if definition.tuning is not None:
            tuning_result = run_hyperparameter_search(
                candidate_pipeline,
                X=train_features,
                y=aligned_target,
                config=definition.tuning,
                tracker=tracker,
            )
            candidate_pipeline = tuning_result.best_estimator
            best_params = dict(tuning_result.best_params)
            if tracker is not None:
                if tuning_result.best_score is not None:
                    tracker.log_metrics({"tuning_best_score": float(tuning_result.best_score)})
                tracker.log_params(
                    {f"tuning__{k}": v for k, v in tuning_result.best_params.items()}
                )
        else:
            candidate_pipeline.fit(train_features, aligned_target)

        val_metrics = _evaluate_candidate(
            candidate_pipeline,
            splits.val_frame,
            splits.y_val,
            label_column=label_column,
        )
        test_metrics = _evaluate_candidate(
            candidate_pipeline,
            splits.test_frame,
            splits.y_test,
            label_column=label_column,
        )

        if tracker is not None:
            tracker.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            tracker.end_run()

        candidate_results.append(
            CandidateResult(
                name=definition.name,
                pipeline=candidate_pipeline,
                metrics_val=val_metrics,
                metrics_test=test_metrics,
                selected_features=selection_result.selected_features,
                run_id=run_id,
                params=best_params,
            )
        )

    outcome = ModelSelectionOutcome(
        feature_selection=selection_result,
        candidates=candidate_results,
        primary_metric=primary_metric,
    )

    # Validar que la métrica primaria exista.
    if primary_metric not in outcome.best_candidate.metrics_val:
        raise KeyError(
            f"La métrica primaria '{primary_metric}' no está en los resultados. "
            f"Métricas disponibles: {list(outcome.best_candidate.metrics_val)}"
        )

    return outcome


def _build_assembler_params(config: PipelineConfig) -> dict[str, Any]:
    """Prepara los parámetros del FeatureFrameAssembler."""
    params = dict(config.assembler_params or {})
    params.setdefault("target_column", config.target_column)
    return params


def _build_candidate_pipeline(
    base_config: PipelineConfig,
    definition: CandidateDefinition,
    selected_features: tuple[str, ...],
) -> Pipeline:
    """Compone un pipeline por candidato utilizando los features seleccionados."""
    extra_steps = _merge_extra_steps(base_config.extra_steps, definition.extra_steps)

    config = PipelineConfig(
        target_column=base_config.target_column,
        assembler_params=base_config.assembler_params,
        selected_features=selected_features,
        selected_features_path=None,
        extra_steps=extra_steps,
    )

    return build_training_pipeline(
        estimator=definition.estimator,
        config=config,
    )


def _merge_extra_steps(
    base_steps: Iterable[tuple[str, TransformerMixin]] | None,
    new_steps: Iterable[tuple[str, TransformerMixin]] | None,
) -> list[tuple[str, TransformerMixin]]:
    """Combina pasos adicionales para el pipeline preservando el orden."""
    steps: list[tuple[str, TransformerMixin]] = []
    if base_steps:
        steps.extend(base_steps)
    if new_steps:
        steps.extend(new_steps)
    return steps


def _evaluate_candidate(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    label_column: str | None = None,
) -> dict[str, float]:
    """Calcula métricas estándar de regresión en el dataset proporcionado."""
    if label_column is None and target.name:
        label_column = target.name

    features_frame = frame
    if label_column:
        features_frame = frame.drop(columns=[label_column], errors="ignore")

    predictions = _predict_with_index(pipeline, features_frame)

    aligned_target = target.loc[predictions.index]
    mask = aligned_target.notna()
    aligned_target = aligned_target[mask]
    predictions = predictions[mask]

    if aligned_target.empty:
        raise ValueError("No hay observaciones válidas para evaluar el candidato.")

    errors = aligned_target - predictions
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))

    epsilon = 1e-8
    mape = float(np.mean(np.abs(errors / (aligned_target + epsilon)))) * 100
    ss_res = float(np.sum(np.square(errors)))
    ss_tot = float(np.sum(np.square(aligned_target - aligned_target.mean())))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "mse": mse,
        "r2": r2,
    }


def _predict_with_index(pipeline: Pipeline, frame: pd.DataFrame) -> pd.Series:
    """Genera predicciones preservando el índice del DataFrame original."""
    preds = pipeline.predict(frame)
    if isinstance(preds, pd.Series):
        return preds
    return pd.Series(preds, index=frame.index, name="prediction")


def _to_path(path: str | Path | None) -> Path | None:
    """Convierte un string o Path a Path (o None)."""
    if path is None:
        return None
    return Path(path)


def _create_passthrough_selection_result(
    feature_matrix: pd.DataFrame,
    estimator_name: str,
    *,
    dropped_features: Sequence[str] | None = None,
) -> FeatureSelectionResult:
    """Genera un resultado de selección trivial usando todas las columnas."""
    columns = list(feature_matrix.columns)
    count = len(columns)
    return FeatureSelectionResult(
        selected_features=tuple(columns),
        ranking=tuple([1] * count),
        support=tuple([True] * count),
        validation_scores=tuple(),
        estimator_name=estimator_name,
        artifact_path=None,
        dropped_features=tuple(dropped_features) if dropped_features else (),
    )
