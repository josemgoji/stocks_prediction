"""Rutinas para búsqueda de hiperparámetros con validación temporal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.training.feature_selection import TrackingClient


@dataclass(slots=True)
class HyperparameterTuningConfig:
    """Configuración serializable para la búsqueda de hiperparámetros."""

    strategy: str = "grid"
    param_grid: Mapping[str, Sequence[Any]] | None = None
    n_iter: int | None = None
    scoring: str | None = None
    n_splits: int = 3
    gap: int = 0
    refit: str | bool | None = None
    random_state: int | None = None
    artifact_dir: str | Path | None = None
    artifact_name: str | None = None
    tracking_artifact_path: str = "hyperparameter_tuning"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HyperparameterTuningConfig":
        """Construye la configuración asegurando tipos por defecto."""
        params = dict(data or {})
        return cls(
            strategy=str(params.get("strategy", "grid")).lower(),
            param_grid=params.get("param_grid"),
            n_iter=params.get("n_iter"),
            scoring=params.get("scoring"),
            n_splits=int(params.get("n_splits", 3)),
            gap=int(params.get("gap", 0)),
            refit=params.get("refit"),
            random_state=params.get("random_state"),
            artifact_dir=params.get("artifact_dir"),
            artifact_name=params.get("artifact_name"),
            tracking_artifact_path=params.get(
                "tracking_artifact_path",
                "hyperparameter_tuning",
            ),
        )


@dataclass(slots=True)
class HyperparameterTuningResult:
    """Resultado de la búsqueda de hiperparámetros."""

    best_estimator: Pipeline
    best_params: Mapping[str, Any]
    best_score: float | None
    artifact_path: Path | None = None


def run_hyperparameter_search(
    pipeline: Pipeline,
    *,
    X,
    y,
    config: HyperparameterTuningConfig,
    tracker: TrackingClient | None = None,
) -> HyperparameterTuningResult:
    """Ejecuta GridSearchCV/RandomizedSearchCV con TimeSeriesSplit.

    El pipeline recibido debe incluir toda la ingeniería de features para que los
    folds se mantengan consistentes con el entrenamiento final.
    """
    if not config.param_grid:
        raise ValueError("La configuración de tuning requiere 'param_grid'.")

    cv = TimeSeriesSplit(n_splits=config.n_splits, gap=config.gap)
    refit = True if config.refit is None else config.refit

    if config.strategy == "random":
        search = RandomizedSearchCV(
            pipeline,
            config.param_grid,
            n_iter=config.n_iter or 10,
            scoring=config.scoring,
            cv=cv,
            refit=refit,
            random_state=config.random_state,
            n_jobs=None,
        )
    elif config.strategy == "grid":
        search = GridSearchCV(
            pipeline,
            config.param_grid,
            scoring=config.scoring,
            cv=cv,
            refit=refit,
            n_jobs=None,
        )
    else:
        raise ValueError(f"Estrategia de tuning no soportada: {config.strategy}")

    search.fit(X, y)

    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_score = float(search.best_score_) if hasattr(search, "best_score_") else None

    payload = {
        "strategy": config.strategy,
        "param_grid": config.param_grid,
        "best_params": best_params,
        "best_score": best_score,
        "n_splits": config.n_splits,
        "gap": config.gap,
        "scoring": config.scoring,
        "refit": refit,
        "cv_results": _sanitize_cv_results(search.cv_results_),
    }

    artifact_path: Path | None = None
    if config.artifact_dir:
        directory = Path(config.artifact_dir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = config.artifact_name or "tuning_results.json"
        artifact_path = directory / filename
        with artifact_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    if tracker is not None:
        tracker.log_dict(
            payload,
            f"{config.tracking_artifact_path.rstrip('/')}/tuning_results.json",
        )

    return HyperparameterTuningResult(
        best_estimator=best_estimator,
        best_params=best_params,
        best_score=best_score,
        artifact_path=artifact_path,
    )


def _sanitize_cv_results(results: Mapping[str, Any]) -> Mapping[str, Any]:
    """Convierte los arrays de resultados de CV a listas serializables."""
    sanitized: dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        else:
            sanitized[key] = value
    return sanitized

