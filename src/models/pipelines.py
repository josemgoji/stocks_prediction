"""Constructores de pipelines reutilizables para entrenamiento y scoring."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.features.transformers import (
    FeatureFrameAssembler,
    FeatureSubsetTransformer,
    FillNaTransformer,
)
from src.training.feature_selection import load_selected_features


@dataclass(slots=True)
class PipelineConfig:
    """Configuración mínima para generar un pipeline de entrenamiento."""

    target_column: str
    assembler_params: Mapping[str, Any] | None = None
    selected_features: Sequence[str] | None = None
    selected_features_path: str | None = None
    extra_steps: Iterable[tuple[str, TransformerMixin]] | None = None


def build_training_pipeline(
    estimator: BaseEstimator,
    config: PipelineConfig,
) -> Pipeline:
    """Arma un Pipeline de sklearn con ingeniería de features y estimador final.

    El pipeline aplica:
        1. `FeatureFrameAssembler` para generar lags, retornos, etc.
        2. `FeatureSubsetTransformer` (opcional) con la lista proveniente de RFECV.
        3. Pasos extra declarados (p. ej. escaladores) antes del estimador.
        4. Estimador final.

    Args:
        estimator: Modelo final (regresión/clasificación) a entrenar.
        config: Parámetros de armado, incluyendo la ruta/lista de features seleccionadas.

    Returns:
        Pipeline listo para `fit` y `predict`.
    """
    if config.assembler_params is None:
        assembler_params = {"target_column": config.target_column}
    else:
        assembler_params = dict(config.assembler_params)
        assembler_params.setdefault("target_column", config.target_column)

    steps: list[tuple[str, TransformerMixin | BaseEstimator]] = [
        ("feature_assembler", FeatureFrameAssembler(**assembler_params)),
        ("feature_fillna", FillNaTransformer()),
    ]

    selected_features = _resolve_selected_features(config)
    if selected_features is not None:
        steps.append(
            ("feature_subset", FeatureSubsetTransformer(selected_features, fail_on_missing=False))
        )

    if config.extra_steps:
        steps.extend(config.extra_steps)

    steps.append(("estimator", estimator))
    return Pipeline(steps)


def _resolve_selected_features(config: PipelineConfig) -> Sequence[str] | None:
    """Obtiene la lista de features a usar según el config."""
    if config.selected_features is not None:
        return config.selected_features
    if config.selected_features_path:
        path = config.selected_features_path
        return load_selected_features(Path(path))
    return None
