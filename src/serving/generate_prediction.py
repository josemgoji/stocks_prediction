import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import mlflow
import pandas as pd

from src.application.use_cases.fetch_market_data import fetch_market_data
from src.registry.mlflow_client import configure_mlflow_environment
from src.serving.model_loader import load_latest_model_version
from src.utils.io import load_yaml


def generate_prediction_from_configs(
    *,
    data_config_path: Path,
    training_config_path: Path,
    prediction_date: str | None = None,
) -> dict[str, Any]:
    """Carga configuraciones, obtiene datos recientes y retorna la predicción."""
    data_cfg: Mapping[str, Any] = load_yaml(data_config_path)
    training_cfg: Mapping[str, Any] = load_yaml(training_config_path)
    ticker = str(data_cfg["ticker"]).strip()

    horizon = int(training_cfg.get("horizon", 1))
    mlflow_cfg = training_cfg.get("mlflow") or {}
    registry_cfg = training_cfg.get("model_registry") or {}

    tracking_uri = mlflow_cfg["tracking_uri"]
    registry_uri = registry_cfg.get("registry_uri") or tracking_uri
    model_name = registry_cfg["model_name"]

    configure_mlflow_environment(mlflow_cfg)

    loaded_model = load_latest_model_version(
        model_name=model_name,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    pipeline = loaded_model.pipeline
    model_uri = loaded_model.model_uri
    model_version = loaded_model.version
    run_id = loaded_model.run_id

    required_history = _infer_history_window(pipeline, run_id=run_id)

    if prediction_date:
        try:
            end_ts = pd.Timestamp(prediction_date)
        except Exception:  
            end_ts = pd.Timestamp.now(tz="UTC")
    else:
        end_ts = pd.Timestamp.now(tz="UTC")

    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    start_ts = end_ts - pd.tseries.offsets.BDay(required_history)

    config_start = data_cfg.get("start")
    if config_start is not None:
        configured_start = pd.Timestamp(datetime.fromisoformat(str(config_start))).tz_localize("UTC")
        start_ts = max(start_ts, configured_start)

    interval = data_cfg.get("interval")
    auto_adjust = data_cfg.get("auto_adjust")
    macro_raw = data_cfg.get("macro")
    macro_cfg = macro_raw if isinstance(macro_raw, Mapping) else {}

    dataset = fetch_market_data(
        ticker=ticker,
        start=start_ts.to_pydatetime(),
        end=end_ts.to_pydatetime(),
        interval=interval,
        auto_adjust=auto_adjust,
        macro_series=macro_cfg.get("series"),
        macro_fill_method=macro_cfg.get("fill_method"),
        fred_api_key=os.getenv("FRED_API_KEY"),
    ).sort_index()
    dataset = dataset.loc[~dataset.index.duplicated(keep="last")]

    if dataset.empty:
        raise ValueError("No hay datos suficientes para generar la predicción.")

    numeric_columns = dataset.select_dtypes(include=["number"]).columns
    if len(numeric_columns) != len(dataset.columns):
        dataset = dataset.copy()
    dataset = dataset.astype({column: "float64" for column in numeric_columns})

    predictions = pipeline.predict(dataset)
    if isinstance(predictions, pd.Series):
        prediction_series = predictions
    elif isinstance(predictions, pd.DataFrame):
        prediction_series = predictions.iloc[:, 0]
    else:
        prediction_series = pd.Series(
            predictions,
            index=dataset.index[-len(predictions) :],
            dtype=float,
            name="prediction",
        )

    latest_timestamp = pd.Timestamp(prediction_series.index[-1])
    latest_timestamp = latest_timestamp.tz_convert(None) if latest_timestamp.tzinfo else latest_timestamp
    forecast_timestamp = latest_timestamp + pd.tseries.offsets.BDay(horizon)

    result = {
        "ticker": ticker,
        "generated_at": datetime.utcnow().isoformat(),
        "lookback_start": start_ts.isoformat(),
        "lookback_end": end_ts.isoformat(),
        "horizon": horizon,
        "prediction_date": str(forecast_timestamp.date()),
        "prediction_value": float(prediction_series.iloc[-1]),
        "model_name": model_name,
        "model_version": model_version,
        "model_uri": model_uri,
    }

    return result


def _infer_history_window(pipeline: Any, *, run_id: str | None) -> int:
    """Estimación del lookback necesario usando metadatos MLflow o el assembler."""
    if run_id:
        try:
            run = mlflow.get_run(run_id)
        except Exception:  
            run = None
        if run:
            raw_values = [
                str(value)
                for source in (run.data.params, run.data.tags)
                for key, value in source.items()
                if "lag" in key.lower()
            ]
            matches = [int(match) for value in raw_values for match in re.findall(r"\d+", value)]
            if matches:
                return max(matches)

    assembler = getattr(pipeline, "named_steps", {}).get("feature_assembler")
    if assembler is None:
        return 60

    def _values_as_ints(value: Any) -> list[int]:
        if value is None:
            return []
        iterable = value if isinstance(value, (list, tuple, set)) else [value]
        ints: list[int] = []
        for item in iterable:
            if item is None:
                continue
            try:
                ints.append(int(item))
            except Exception:  
                continue
        return ints

    lag_values = _values_as_ints(getattr(assembler, "_lag_values", None))
    if not lag_values:
        lag_values = _values_as_ints(getattr(assembler, "lags", 1))

    candidates = [1, *lag_values]
    candidates.extend(_values_as_ints(getattr(assembler, "returns", 1)))
    candidates.extend(_values_as_ints(getattr(assembler, "rolling_windows", 1)))

    return max(candidates)
