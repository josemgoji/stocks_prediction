"""Transformadores compatibles con scikit-learn para features temporales."""

from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.builders import assemble_feature_frame


class FeatureFrameAssembler(BaseEstimator, TransformerMixin):
    """Genera un paquete completo de features financieros a partir de una columna objetivo."""

    def __init__(
        self,
        *,
        target_column: str,
        lags: Iterable[int] = (1, 5, 10),
        returns: Iterable[int] = (1,),
        rolling_windows: Iterable[int] = (5, 20),
        high_column: str | None = "High",
        low_column: str | None = "Low",
        volume_column: str | None = "Volume",
        price_for_nominal: str | None = None,
        attach_base: bool = False,
        base_columns: Iterable[str] | None = None,
    ) -> None:
        self.target_column = target_column
        self.lags = tuple(lags)
        self.returns = tuple(returns)
        self.rolling_windows = tuple(rolling_windows)
        self.high_column = high_column
        self.low_column = low_column
        self.volume_column = volume_column
        self.price_for_nominal = price_for_nominal
        self.attach_base = attach_base
        self.base_columns = tuple(base_columns) if base_columns is not None else None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # noqa: N803
        self._validate_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        self._validate_columns(X)
        feature_frame = assemble_feature_frame(
            X,
            self.target_column,
            lags=self.lags,
            returns=self.returns,
            rolling_windows=self.rolling_windows,
            high_column=self.high_column,
            low_column=self.low_column,
            volume_column=self.volume_column,
            price_for_nominal=self.price_for_nominal,
        )
        feature_frame = feature_frame.dropna(axis=1, how="all")
        if self.attach_base:
            if self.base_columns is not None:
                base = X.loc[:, list(self.base_columns)]
            else:
                base = X.copy()
            feature_frame = pd.concat([base, feature_frame], axis=1)
        return feature_frame

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        required = [self.target_column]
        if self.high_column is not None:
            required.append(self.high_column)
        if self.low_column is not None:
            required.append(self.low_column)
        if self.volume_column is not None:
            required.append(self.volume_column)
        if self.price_for_nominal is not None:
            required.append(self.price_for_nominal)

        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(
                f"Columnas faltantes para FeatureFrameAssembler: {missing}"
            )

        if self.attach_base and self.base_columns is not None:
            missing_base = [
                column for column in self.base_columns if column not in frame.columns
            ]
            if missing_base:
                raise KeyError(
                    f"Columnas faltantes para adjuntar en FeatureFrameAssembler: {missing_base}"
                )


class FeatureSubsetTransformer(BaseEstimator, TransformerMixin):
    """Selecciona un subconjunto fijo de columnas, fijado tras feature selection."""

    def __init__(self, features: Iterable[str], *, fail_on_missing: bool = True) -> None:
        self.features = tuple(features)
        self.fail_on_missing = fail_on_missing
        self._missing: tuple[str, ...] = ()

    @property
    def missing_(self) -> tuple[str, ...]:
        """Columnas faltantes detectadas en el último `fit` o `transform`."""
        return self._missing

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # noqa: N803
        self._missing = self._compute_missing(X)
        if self.fail_on_missing and self._missing:
            raise KeyError(
                f"Columnas faltantes para FeatureSubsetTransformer: {self._missing}"
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        self._missing = self._compute_missing(X)
        frame = X.copy()
        if self._missing:
            if self.fail_on_missing:
                raise KeyError(
                    f"Columnas faltantes para FeatureSubsetTransformer: {self._missing}"
                )
            for column in self._missing:
                frame[column] = 0
        available = [column for column in self.features if column in frame.columns]
        return frame.loc[:, available]

    def _compute_missing(self, frame: pd.DataFrame) -> tuple[str, ...]:
        return tuple(column for column in self.features if column not in frame.columns)


class FillNaTransformer(BaseEstimator, TransformerMixin):
    """Rellena valores faltantes con un valor constante preservando DataFrames."""

    def __init__(self, value: float | int = 0) -> None:
        self.value = value

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        if isinstance(X, pd.DataFrame):
            return X.fillna(self.value)
        if isinstance(X, pd.Series):
            return X.fillna(self.value)
        raise TypeError(
            "FillNaTransformer solo soporta entradas tipo pandas.DataFrame o pandas.Series."
        )
