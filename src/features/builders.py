"""Funciones utilitarias para crear features sobre series de precios."""

import numpy as np
import pandas as pd


def create_lag(series: pd.Series, lag: int) -> pd.Series:
    """Genera una serie desplazada `lag` periodos hacia atrás."""
    if lag <= 0:
        raise ValueError("`lag` debe ser positivo.")

    lagged = series.shift(lag)
    lagged.name = f"{series.name}_lag_{lag}"
    return lagged


def compute_return(series: pd.Series, periods: int = 1, log: bool = False) -> pd.Series:
    """Calcula el retorno porcentual o logarítmico."""
    if periods <= 0:
        raise ValueError("`periods` debe ser positivo.")

    if log:
        returns = np.log(series / series.shift(periods))
        suffix = "logret"
    else:
        returns = series.pct_change(periods=periods)
        suffix = "ret"

    returns.name = f"{series.name}_{suffix}_{periods}"
    return returns


def rolling_mean(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Media móvil centrada a la derecha."""
    if window <= 0:
        raise ValueError("`window` debe ser positivo.")
    rolling = series.rolling(window=window, min_periods=min_periods or window).mean()
    rolling.name = f"{series.name}_sma_{window}"
    return rolling


def rolling_std(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Volatilidad móvil (desviación estándar)."""
    if window <= 0:
        raise ValueError("`window` debe ser positivo.")
    rolling = series.rolling(window=window, min_periods=min_periods or window).std()
    rolling.name = f"{series.name}_std_{window}"
    return rolling


def compute_amplitude(
    frame: pd.DataFrame, high_column: str, low_column: str, *, suffix: str = "amplitude"
) -> pd.Series:
    """Calcula la amplitud diaria como `high - low`."""
    missing = [column for column in (high_column, low_column) if column not in frame]
    if missing:
        raise KeyError(f"Columnas faltantes para amplitud: {missing}")

    amplitude = frame[high_column] - frame[low_column]
    amplitude.name = f"{suffix}"
    return amplitude


def compute_nominal_value(
    frame: pd.DataFrame,
    price_column: str,
    volume_column: str,
    *,
    suffix: str = "nominal_value",
) -> pd.Series:
    """Calcula el valor nominal negociado como `price * volume`."""
    missing = [
        column for column in (price_column, volume_column) if column not in frame
    ]
    if missing:
        raise KeyError(f"Columnas faltantes para nominal value: {missing}")

    nominal = frame[price_column] * frame[volume_column]
    nominal.name = suffix
    return nominal


def assemble_feature_frame(
    base: pd.DataFrame,
    target_column: str,
    *,
    lags: tuple[int, ...] = (1, 5, 10),
    returns: tuple[int, ...] = (1,),
    rolling_windows: tuple[int, ...] = (5, 20),
    high_column: str | None = "High",
    low_column: str | None = "Low",
    volume_column: str | None = "Volume",
    price_for_nominal: str | None = None,
) -> pd.DataFrame:
    """Construye un `DataFrame` con un set básico de features financieras.

    Args:
        base: Datos originales (debe contener `target_column`).
        target_column: Columna sobre la que calcular features.
        lags: Conjunto de lags simples a generar.
        returns: Periodos para retornos porcentuales.
        rolling_windows: Ventanas para medias y desviaciones móviles.
    """
    if target_column not in base.columns:
        raise KeyError(f"{target_column} no existe en el DataFrame base.")

    series = base[target_column]
    features = {}

    for lag in lags:
        features[f"lag_{lag}"] = create_lag(series, lag)

    for period in returns:
        features[f"ret_{period}"] = compute_return(series, periods=period)

    for window in rolling_windows:
        features[f"sma_{window}"] = rolling_mean(series, window)
        features[f"std_{window}"] = rolling_std(series, window)

    price_column = price_for_nominal or target_column

    if high_column and low_column:
        features["amplitude"] = compute_amplitude(
            base, high_column=high_column, low_column=low_column
        )

    if price_column and volume_column:
        features["nominal_value"] = compute_nominal_value(
            base,
            price_column=price_column,
            volume_column=volume_column,
        )

    feature_frame = pd.concat(features.values(), axis=1)
    return feature_frame
