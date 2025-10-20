"""Herramientas para ejecutar selección de características reproducible."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit


class TrackingClient(Protocol):
    """Interfaz mínima para clientes de tracking como MLflow."""

    def log_dict(self, dictionary: dict, artifact_file: str) -> None:
        """Registra un diccionario como artefacto."""


@dataclass(slots=True)
class FeatureSelectionResult:
    """Resultado serializable de un proceso de selección de features."""

    selected_features: tuple[str, ...]
    ranking: tuple[int, ...]
    support: tuple[bool, ...]
    validation_scores: tuple[float, ...]
    estimator_name: str
    artifact_path: Path | None = None
    dropped_features: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Representación en diccionario lista para ser persistida."""
        return {
            "selected_features": list(self.selected_features),
            "ranking": list(self.ranking),
            "support": list(self.support),
            "validation_scores": list(self.validation_scores),
            "estimator_name": self.estimator_name,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "dropped_features": list(self.dropped_features),
        }


@dataclass(slots=True)
class CorrelatedFeatureDropper:
    """Remueve columnas altamente correlacionadas dentro de un subconjunto."""

    threshold: float

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """Devuelve un DataFrame sin columnas con correlación > threshold."""
        if self.threshold <= 0:
            return frame, ()

        if columns is None:
            candidate_columns = frame.select_dtypes(include="number").columns
        else:
            candidate_columns = [column for column in columns if column in frame.columns]

        if not candidate_columns:
            return frame, ()

        numeric_columns = frame.loc[:, candidate_columns].select_dtypes(include="number").columns
        if numeric_columns.empty:
            return frame, ()

        corr_matrix = frame.loc[:, numeric_columns].corr().abs()
        if corr_matrix.empty:
            return frame, ()

        upper_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.where(upper_mask)

        to_drop: list[str] = []
        for column in upper_triangle.columns:
            column_corr = upper_triangle[column]
            if column_corr.dropna().gt(self.threshold).any():
                to_drop.append(column)

        if not to_drop:
            return frame, ()

        pruned_frame = frame.drop(columns=to_drop, errors="ignore")
        return pruned_frame, tuple(to_drop)


def run_rfecv_selection(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator,
    *,
    n_splits: int = 5,
    min_features_to_select: int = 1,
    step: int | float = 1,
    scoring: str | None = None,
    gap: int = 0,
    artifact_dir: Path | None = None,
    artifact_name: str = "selected_features.json",
    tracking_client: TrackingClient | None = None,
    tracking_artifact_path: str = "feature_selection",
    pre_dropped_features: Sequence[str] | None = None,
) -> FeatureSelectionResult:
    """Ejecuta RFECV con TimeSeriesSplit y persiste la lista de features.

    Args:
        X: Matriz de características en formato `DataFrame`.
        y: Variable objetivo alineada con `X`.
        estimator: Estimador base para evaluar la importancia de los features.
        n_splits: Número de particiones para `TimeSeriesSplit`.
        min_features_to_select: Límite inferior de columnas a mantener.
        step: Tamaño del paso de eliminación usado por RFECV.
        scoring: Métrica de validación; usa la configuración del estimador si es `None`.
        gap: Tamaño del hueco entre train/test en el split temporal.
        artifact_dir: Directorio donde se almacenará el JSON con resultados.
        artifact_name: Nombre del archivo JSON generado.
        tracking_client: Cliente opcional (p. ej. MLflow) para registrar el JSON.
        tracking_artifact_path: Carpeta destino relativa en el cliente de tracking.
        pre_dropped_features: Columnas eliminadas previamente (p. ej. por alta correlación).

    Returns:
        `FeatureSelectionResult` con los datos esenciales de la ejecución.

    Raises:
        ValueError: si `X` no tiene columnas o `n_splits` no es válido.
    """
    if X.empty:
        raise ValueError("El DataFrame de entrada `X` no contiene columnas.")
    if n_splits < 2:
        raise ValueError("`n_splits` para TimeSeriesSplit debe ser al menos 2.")

    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=None,
    )
    selector.fit(X, y)

    support = selector.support_.tolist()
    ranking = selector.ranking_.tolist()
    selected_features = tuple(X.columns[support])
    scores = _extract_cv_scores(selector)

    dropped_features = tuple(pre_dropped_features) if pre_dropped_features else ()

    payload = {
        "selected_features": list(selected_features),
        "ranking": ranking,
        "support": support,
        "validation_scores": scores,
        "estimator_name": estimator.__class__.__name__,
        "n_splits": n_splits,
        "step": step,
        "min_features_to_select": min_features_to_select,
        "scoring": scoring,
        "gap": gap,
        "dropped_features": list(dropped_features),
    }

    artifact_path: Path | None = None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / artifact_name
        with artifact_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    if tracking_client is not None:
        tracking_file = f"{tracking_artifact_path.rstrip('/')}/{artifact_name}"
        tracking_client.log_dict(payload, tracking_file)

    return FeatureSelectionResult(
        selected_features=selected_features,
        ranking=tuple(ranking),
        support=tuple(support),
        validation_scores=tuple(scores),
        estimator_name=estimator.__class__.__name__,
        artifact_path=artifact_path,
        dropped_features=dropped_features,
    )


def load_selected_features(path: Path) -> list[str]:
    """Carga la lista de features previamente persistida.

    Args:
        path: Ruta al archivo JSON generado por `run_rfecv_selection`.

    Returns:
        Lista de nombres de columnas seleccionadas.
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    features = data.get("selected_features")
    if not isinstance(features, Sequence):
        raise ValueError(
            f"Formato inesperado en {path}. Se esperaba la clave 'selected_features'."
        )
    return list(features)


def _extract_cv_scores(selector: RFECV) -> list[float]:
    """Obtiene los puntajes de validación independientemente de la versión de sklearn."""
    if hasattr(selector, "cv_results_"):
        results = selector.cv_results_
        if isinstance(results, dict) and "mean_test_score" in results:
            return np.array(results["mean_test_score"], dtype=float).tolist()
    if hasattr(selector, "grid_scores_"):
        return np.array(selector.grid_scores_, dtype=float).tolist()
    return []
