import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import mlflow
import pandas as pd

from src.application.use_cases.fetch_market_data import fetch_market_data
from src.registry.mlflow_client import configure_mlflow_environment
from src.serving.model_loader import load_latest_model_version
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga datos recientes y genera una predicción con el último modelo registrado en MLflow."
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("conf/base/data.yaml"),
        help="Ruta al archivo de configuración de datos.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("conf/base/training.yaml"),
        help="Ruta al archivo de entrenamiento.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Ruta opcional para guardar la predicción en formato JSON.",
    )
    parser.add_argument(
        "--prediction-date",
        type=str,
        default=None,
        help=(
            "Fecha base (ISO 8601) para la predicción. Si no se indica se usa la fecha actual."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_prediction_from_configs(
        data_config_path=args.data_config,
        training_config_path=args.training_config,
        prediction_date=args.prediction_date,
    )
    print(json.dumps(result, indent=2))
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def generate_prediction_from_configs(
    *,
    data_config_path: Path,
    training_config_path: Path,
    prediction_date: str | None = None,
) -> dict[str, Any]:
    data_cfg = load_yaml(data_config_path)
    training_cfg = load_yaml(training_config_path)
    return _generate_prediction_from_configs(
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        prediction_date=prediction_date,
    )


def _generate_prediction_from_configs(
    *,
    data_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    prediction_date: str | None,
) -> dict[str, Any]:
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
    end_ts = _resolve_prediction_date(prediction_date)
    start_ts = end_ts - pd.tseries.offsets.BDay(required_history)

    config_start = data_cfg.get("start")
    if config_start is not None:
        configured_start = pd.Timestamp(datetime.fromisoformat(str(config_start))).tz_localize("UTC")
        start_ts = max(start_ts, configured_start)

    interval = data_cfg.get("interval")
    macro_cfg = data_cfg.get("macro") or {}
    macro_series = (macro_cfg.get("series") or None) if isinstance(macro_cfg, Mapping) else None
    macro_fill_method = macro_cfg.get("fill_method") if isinstance(macro_cfg, Mapping) else None

    dataset = fetch_market_data(
        ticker=ticker,
        start=start_ts.to_pydatetime(),
        end=end_ts.to_pydatetime(),
        interval=interval,
        macro_series=macro_series,
        macro_fill_method=macro_fill_method,
        fred_api_key=os.getenv("FRED_API_KEY"),
    ).sort_index()
    dataset = dataset.loc[~dataset.index.duplicated(keep="last")]

    dataset = _align_with_model_signature(dataset, pipeline)
    dataset = _ensure_numeric_compatibility(dataset)

    predictions = pipeline.predict(dataset)
    prediction_series = _to_series(predictions, dataset.index)

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
    history_from_metadata = _history_from_mlflow(run_id) if run_id else None
    if history_from_metadata:
        return history_from_metadata

    assembler = getattr(pipeline, "named_steps", {}).get("feature_assembler")
    if assembler is None:
        return 60

    lag_values = getattr(assembler, "_lag_values", None)
    if lag_values:
        max_lag = max(int(value) for value in lag_values if value is not None)
    else:
        max_lag = _max_config_value(getattr(assembler, "lags", 1))

    max_return = _max_config_value(getattr(assembler, "returns", 1))
    max_rolling = _max_config_value(getattr(assembler, "rolling_windows", 1))

    return max(max_lag, max_return, max_rolling)


def _to_series(predictions: Any, index: pd.Index) -> pd.Series:
    if isinstance(predictions, pd.Series):
        return predictions
    if isinstance(predictions, pd.DataFrame):
        first_column = predictions.columns[0]
        return predictions[first_column]
    series = pd.Series(predictions).astype(float)
    series.index = index[-len(series) :]
    series.name = "prediction"
    return series


def _align_with_model_signature(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    metadata = getattr(model, "metadata", None)
    if metadata is None or not hasattr(metadata, "get_input_schema"):
        return frame

    try:
        schema = metadata.get_input_schema()
    except Exception: 
        return frame

    if not schema or not getattr(schema, "inputs", None):
        return frame

    required_columns = [col.name for col in schema.inputs]
    dtype_mapping: dict[str, str | None] = {
        col.name: _mlflow_dtype_to_numpy(getattr(col, "type", None))
        for col in schema.inputs
    }

    aligned = frame.copy()
    for column in required_columns:
        if column not in aligned.columns:
            aligned[column] = float("nan")

    for column, dtype in dtype_mapping.items():
        if dtype and column in aligned.columns:
            try:
                aligned[column] = aligned[column].astype(dtype)
            except Exception: 
                pass

    extra = [column for column in aligned.columns if column not in required_columns]
    ordered_columns = required_columns + extra
    return aligned.loc[:, ordered_columns]


def _ensure_numeric_compatibility(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = frame.select_dtypes(include=["int", "uint", "int64", "uint64"]).columns
    if len(numeric_columns) == 0:
        return frame

    converted = frame.copy()
    converted.loc[:, numeric_columns] = converted.loc[:, numeric_columns].astype("float64")
    return converted


def _mlflow_dtype_to_numpy(dtype: Any) -> str | None:
    if dtype is None:
        return None
    name = getattr(dtype, "name", str(dtype)).lower()
    mapping = {
        "double": "float64",
        "float": "float32",
        "integer": "int64",
        "long": "int64",
        "short": "int32",
        "byte": "int8",
    }
    return mapping.get(name)


def _resolve_prediction_date(raw: str | None) -> pd.Timestamp:
    if raw is None:
        return pd.Timestamp.now(tz="UTC")
    try:
        ts = pd.Timestamp(raw)
    except Exception:  # noqa: BLE001
        return pd.Timestamp.now(tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _history_from_mlflow(run_id: str) -> int | None:
    try:
        run = mlflow.get_run(run_id)
    except Exception: 
        return None

    candidates: list[int] = []
    for source in (run.data.params, run.data.tags):
        for key, value in source.items():
            if "lag" not in key.lower():
                continue
            candidates.extend(_extract_ints(value))

    if not candidates:
        return None
    return max(candidates)


def _extract_ints(raw: str) -> list[int]:
    values: list[int] = []
    current = ""
    for char in raw:
        if char.isdigit():
            current += char
        elif current:
            values.append(int(current))
            current = ""
    if current:
        values.append(int(current))
    return values


def _max_config_value(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, int):
        return max(value, 1)
    if isinstance(value, (list, tuple, set)):
        ints = [int(item) for item in value if item is not None]
        return max(ints) if ints else 1
    try:
        return max(int(value), 1)
    except Exception: 
        return 1


if __name__ == "__main__":
    main()
