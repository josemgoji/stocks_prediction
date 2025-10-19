"""Descarga series históricas desde Yahoo Finance usando `yfinance`."""

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


class YahooFinanceResource:
    """Envuelve la llamada a `yfinance.download` con configuraciones por defecto."""

    def __init__(self, interval: str = "1d", auto_adjust: bool = True) -> None:
        self.interval = interval
        self.auto_adjust = auto_adjust

    def fetch_history(
        self,
        *,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Optional[str] = None,
        auto_adjust: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Descarga precios históricos para un ticker y regresa un DataFrame OHLCV."""
        if not ticker:
            raise ValueError("Se requiere un ticker no vacío.")

        if start >= end:
            raise ValueError("`start` debe ser anterior a `end`.")

        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval or self.interval,
            auto_adjust=self.auto_adjust if auto_adjust is None else auto_adjust,
            progress=False,
        )

        if data.empty:
            raise ValueError(
                f"Yahoo Finance regresó un DataFrame vacío para {ticker} "
                f"entre {start} y {end}."
            )

        if isinstance(data.columns, pd.MultiIndex):
            ticker_levels = data.columns.get_level_values(-1).unique()
            if len(ticker_levels) == 1:
                data.columns = data.columns.get_level_values(0)
            else:
                data.columns = [
                    "_".join(str(level) for level in col if level not in ("", None))
                    for col in data.columns.to_flat_index()
                ]
            data.columns.name = None

        data.index = (
            data.index.tz_convert("UTC")
            if data.index.tz is not None
            else data.index.tz_localize("UTC")
        )
        data.attrs["ticker"] = ticker
        return data
