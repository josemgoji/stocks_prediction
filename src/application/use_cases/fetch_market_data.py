"""Funciones básicas para descargar y opcionalmente guardar datos de mercado."""

from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
import numpy as np

from src.resources.yahoo_finance import YahooFinanceResource
from src.resources.fred import FredResource


def fetch_market_data(
    *,
    ticker: str,
    start: datetime,
    end: datetime,
    interval: Optional[str] = None,
    save_path: Optional[Path] = None,
    save_format: str = "parquet",
    auto_adjust: Optional[bool] = None,
    resource: Optional[YahooFinanceResource] = None,
    macro_series: Optional[Mapping[str, str]] = None,
    fred_api_key: Optional[str] = None,
    fred_resource: Optional[FredResource] = None,
    macro_fill_method: Optional[str] = "ffill",
) -> pd.DataFrame:
    """Descarga datos y los persiste si se solicita."""
    data_resource = resource or YahooFinanceResource()

    dataset = data_resource.fetch_history(
        ticker=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
    )
    
    original_columns = list(dataset.columns)

    if macro_series:
        macro_client = fred_resource or FredResource(api_key=fred_api_key)
        macro_data = macro_client.fetch_series(
            macro_series,
            index=dataset.index,
            start=start,
            end=end,
            fill_method=macro_fill_method,
        )
        dataset = dataset.join(macro_data)

        macro_columns = list(macro_data.columns)
        if macro_columns:
            dataset.loc[:, macro_columns] = dataset[macro_columns].ffill()
            dataset.loc[:, macro_columns] = dataset[macro_columns].bfill()

            empty_macros = [
                column for column in macro_columns if dataset[column].isna().all()
            ]
            if empty_macros:
                dataset = dataset.drop(columns=empty_macros)

    if original_columns:
        dataset = dataset.dropna(subset=original_columns)

    numeric_columns = dataset.select_dtypes(include=["number"]).columns
    if len(numeric_columns) > 0:
        dataset = dataset.astype({column: "float64" for column in numeric_columns})

    if save_path:
        _persist(dataset, save_path, save_format)

    return dataset


def _persist(dataset: pd.DataFrame, destination: Path, fmt: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        dataset.to_parquet(destination)
        return

    if fmt == "csv":
        dataset.to_csv(destination)
        return

    raise ValueError(
        f"Formato de guardado no soportado: {fmt}. Usa 'parquet' o 'csv'."
    )
