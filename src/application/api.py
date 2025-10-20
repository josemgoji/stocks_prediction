"""API REST mínima para descargar datos, entrenar modelos y generar predicciones."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.application.use_cases.fetch_market_data import fetch_market_data
from src.serving.generate_prediction import generate_prediction_from_configs
from src.training.train import run_training
from src.utils.io import load_yaml

app = FastAPI(title="Stocks Prediction API", version="1.0.0")


class FetchDataRequest(BaseModel):
    data_config_path: Path = Field(default=Path("conf/base/data.yaml"))
    skip_macro: bool = False


class TrainRequest(BaseModel):
    data_config_path: Path = Field(default=Path("conf/base/data.yaml"))
    training_config_path: Path = Field(default=Path("conf/base/training.yaml"))


class PredictionRequest(BaseModel):
    data_config_path: Path = Field(default=Path("conf/base/data.yaml"))
    training_config_path: Path = Field(default=Path("conf/base/training.yaml"))
    prediction_date: str | None = None


@app.post("/data/fetch")
def fetch_data(request: FetchDataRequest) -> dict[str, Any]:
    """Descarga datos históricos (y macro opcional) según la configuración."""
    try:
        params = _build_fetch_params(request.data_config_path, skip_macro=request.skip_macro)
        dataset = fetch_market_data(**params).sort_index()
    except Exception as exc:  
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if dataset.empty:
        raise HTTPException(status_code=404, detail="La descarga no devolvió registros.")

    return {
        "ticker": params["ticker"],
        "rows": len(dataset),
        "start": _to_serializable(dataset.index.min()),
        "end": _to_serializable(dataset.index.max()),
        "head": _serialize_head(dataset),
        "saved_path": str(params.get("save_path")) if params.get("save_path") else None,
    }


@app.post("/training/run")
def train_model(request: TrainRequest) -> dict[str, Any]:
    """Ejecución de entrenamiento completa tomando rutas declaradas en YAML."""
    try:
        outcome = run_training(
            data_config_path=request.data_config_path,
            training_config_path=request.training_config_path,
        )
    except Exception as exc:  
        raise HTTPException(status_code=500, detail=f"No se pudo completar el entrenamiento: {exc}") from exc

    best = outcome.best_candidate
    metrics_val = _float_map(best.metrics_val)
    metrics_test = _float_map(best.metrics_test)

    return {
        "best_candidate": best.name,
        "primary_metric": outcome.primary_metric,
        "primary_metric_value": metrics_val.get(outcome.primary_metric),
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "selected_features": list(outcome.feature_selection.selected_features),
        "selected_features_artifact": str(outcome.feature_selection.artifact_path)
        if outcome.feature_selection.artifact_path
        else None,
    }


@app.post("/prediction/run")
def generate_prediction(request: PredictionRequest) -> dict[str, Any]:
    """Genera la predicción más reciente aprovechando el modelo registrado."""
    try:
        return generate_prediction_from_configs(
            data_config_path=request.data_config_path,
            training_config_path=request.training_config_path,
            prediction_date=request.prediction_date,
        )
    except Exception as exc:  
        raise HTTPException(status_code=500, detail=f"No se pudo generar la predicción: {exc}") from exc


def _build_fetch_params(path: Path, *, skip_macro: bool) -> dict[str, Any]:
    """Arma los parámetros de descarga de mercado a partir del YAML."""
    config = load_yaml(path)
    for key in ("ticker", "start", "end"):
        if key not in config:
            raise ValueError(f"'{key}' es obligatorio en el YAML de datos.")
    save_cfg = config.get("save", {}) if isinstance(config.get("save"), dict) else {}
    macro_cfg = config.get("macro") if isinstance(config.get("macro"), dict) else None

    params: dict[str, Any] = {
        "ticker": config["ticker"],
        "start": datetime.fromisoformat(str(config["start"])),
        "end": datetime.fromisoformat(str(config["end"])),
        "interval": config.get("interval"),
        "save_path": Path(save_cfg.get("path")) if save_cfg.get("path") else None,
        "save_format": save_cfg.get("format", "parquet"),
        "auto_adjust": config.get("auto_adjust"),
    }

    if macro_cfg and not skip_macro:
        series = macro_cfg.get("series")
        if isinstance(series, dict) and series:
            params["macro_series"] = series
            if macro_cfg.get("api_key"):
                params["fred_api_key"] = str(macro_cfg["api_key"])
            if macro_cfg.get("fill_method"):
                params["macro_fill_method"] = str(macro_cfg["fill_method"])

    return params


def _serialize_head(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    """Convierte las primeras filas del dataframe en registros serializables."""
    head = frame.head(limit).reset_index()
    head = head.applymap(_to_serializable)
    return head.to_dict(orient="records")


def _float_map(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Normaliza diccionarios de métricas a flotantes JSON-friendly."""
    return {key: _to_float(value) for key, value in metrics.items()}


def _to_float(value: Any) -> float | None:
    """Cast seguro a float devolviendo None cuando no aplica."""
    if value is None:
        return None
    if isinstance(value, (float, int)):
        value_float = float(value)
    elif isinstance(value, (np.floating, np.integer)):
        value_float = float(value)
    else:
        return None
    return None if np.isnan(value_float) else value_float


def _to_serializable(value: Any) -> Any:
    """Homogeneiza tipos numpy/pandas a objetos compatibles con JSON."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if pd.isna(value):
        return None
    return value


__all__ = ["app"]
