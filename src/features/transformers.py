"""Transformadores compatibles con scikit-learn para features temporales."""

from collections.abc import Iterable
from numbers import Integral

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.features.builders import assemble_feature_frame

_SCALER_REGISTRY = {
    "standard": StandardScaler,
    "standard_scaler": StandardScaler,
    "minmax": MinMaxScaler,
    "minmax_scaler": MinMaxScaler,
    "robust": RobustScaler,
    "robust_scaler": RobustScaler,
}


def build_feature_matrix(
    frame: pd.DataFrame,
    *,
    target_column: str,
    lags: Iterable[int],
    returns: Iterable[int],
    rolling_windows: Iterable[int],
    high_column: str | None,
    low_column: str | None,
    volume_column: str | None,
    price_for_nominal: str | None,
    attach_base: bool,
    base_columns: Iterable[str] | None,
) -> pd.DataFrame:
    """Genera el DataFrame de features aplicando `assemble_feature_frame` y reglas de adjuntos."""
    feature_frame = assemble_feature_frame(
        frame,
        target_column,
        lags=tuple(lags),
        returns=tuple(returns),
        rolling_windows=tuple(rolling_windows),
        high_column=high_column,
        low_column=low_column,
        volume_column=volume_column,
        price_for_nominal=price_for_nominal,
    )
    feature_frame = feature_frame.dropna(axis=1, how="all")

    if not attach_base:
        return feature_frame

    if base_columns is not None:
        base = frame.loc[:, list(base_columns)]
    else:
        base = frame.drop(columns=[target_column], errors="ignore")
        label_prefix = f"{target_column}_target"
        label_columns = [col for col in base.columns if col.startswith(label_prefix)]
        if label_columns:
            base = base.drop(columns=label_columns)

    return pd.concat([base, feature_frame], axis=1)


class FeatureFrameAssembler(BaseEstimator, TransformerMixin):
    """Genera un paquete completo de features financieros a partir de una columna objetivo."""

    def __init__(
        self,
        *,
        target_column: str,
        lags: Iterable[int] | int = (1, 5, 10),
        returns: Iterable[int] = (1,),
        rolling_windows: Iterable[int] = (5, 20),
        high_column: str | None = "High",
        low_column: str | None = "Low",
        volume_column: str | None = "Volume",
        price_for_nominal: str | None = None,
        attach_base: bool = False,
        base_columns: Iterable[str] | None = None,
        scaler: str | BaseEstimator | None = None,
    ) -> None:
        self.target_column = target_column
        self.lags = lags
        self._lag_values = self._normalize_lags(lags)
        self.returns = tuple(returns)
        self.rolling_windows = tuple(rolling_windows)
        self.high_column = high_column
        self.low_column = low_column
        self.volume_column = volume_column
        self.price_for_nominal = price_for_nominal
        self.attach_base = attach_base
        self.base_columns = tuple(base_columns) if base_columns is not None else None
        self.scaler = scaler

        self._scaler: BaseEstimator | None = None
        self._feature_columns: tuple[str, ...] | None = None
        self._scaled_columns: tuple[str, ...] = ()
        self._scale_fill_values: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # noqa: N803
        self._validate_columns(X)
        self._lag_values = self._normalize_lags(self.lags)
        feature_frame = build_feature_matrix(
            X,
            target_column=self.target_column,
            lags=self._lag_values,
            returns=self.returns,
            rolling_windows=self.rolling_windows,
            high_column=self.high_column,
            low_column=self.low_column,
            volume_column=self.volume_column,
            price_for_nominal=self.price_for_nominal,
            attach_base=self.attach_base,
            base_columns=self.base_columns,
        )
        self._feature_columns = tuple(feature_frame.columns)

        if self.scaler is None:
            self._scaler = None
            self._scaled_columns = ()
            self._scale_fill_values = None
            return self

        scaled_columns = feature_frame.select_dtypes(include="number").columns
        self._scaled_columns = tuple(scaled_columns)
        if not self._scaled_columns:
            self._scaler = None
            self._scale_fill_values = None
            return self

        float_cast = {column: "float64" for column in self._scaled_columns}
        feature_frame = feature_frame.astype(float_cast, copy=False)

        scaler = self._build_scaler()
        fill_values = feature_frame.loc[:, self._scaled_columns].mean(skipna=True)
        fill_values = fill_values.fillna(0.0)
        data_for_fit = feature_frame.loc[:, self._scaled_columns].fillna(fill_values)

        scaler.fit(data_for_fit)

        self._scaler = scaler
        self._scale_fill_values = fill_values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        self._validate_columns(X)
        self._lag_values = self._normalize_lags(self.lags)
        feature_frame = build_feature_matrix(
            X,
            target_column=self.target_column,
            lags=self._lag_values,
            returns=self.returns,
            rolling_windows=self.rolling_windows,
            high_column=self.high_column,
            low_column=self.low_column,
            volume_column=self.volume_column,
            price_for_nominal=self.price_for_nominal,
            attach_base=self.attach_base,
            base_columns=self.base_columns,
        )

        if self._feature_columns is not None:
            for column in self._feature_columns:
                if column not in feature_frame.columns:
                    feature_frame[column] = 0.0
            extra_columns = [
                column for column in feature_frame.columns if column not in self._feature_columns
            ]
            if extra_columns:
                feature_frame = feature_frame.drop(columns=extra_columns)
            feature_frame = feature_frame.loc[:, self._feature_columns]

        if self._scaler is not None and self._scaled_columns:
            float_cast = {column: "float64" for column in self._scaled_columns}
            feature_frame = feature_frame.astype(float_cast, copy=False)
            if self._scale_fill_values is not None:
                fill_values = self._scale_fill_values.reindex(self._scaled_columns).fillna(0.0)
            else:
                fill_values = pd.Series(0.0, index=self._scaled_columns)

            data_to_scale = feature_frame.loc[:, self._scaled_columns].copy()
            mask = data_to_scale.isna()
            data_to_scale = data_to_scale.fillna(fill_values)

            scaled = self._scaler.transform(data_to_scale)
            scaled_df = pd.DataFrame(
                scaled,
                columns=self._scaled_columns,
                index=feature_frame.index,
            )
            scaled_df = scaled_df.mask(mask)
            feature_frame.loc[:, self._scaled_columns] = scaled_df

        return feature_frame

    def _normalize_lags(self, lags: Iterable[int] | int) -> tuple[int, ...]:
        """Convierte `lags` declarativos en una tupla ordenada."""
        if isinstance(lags, Integral):
            if lags <= 0:
                raise ValueError("`lags` debe ser un entero positivo.")
            return tuple(range(1, int(lags) + 1))

        try:
            lag_values = tuple(int(lag) for lag in lags)
        except TypeError as exc:  # noqa: PERF203
            raise ValueError("`lags` debe ser un entero positivo o un iterable de enteros.") from exc

        if not lag_values:
            raise ValueError("`lags` no puede ser un iterable vacío.")

        if any(lag <= 0 for lag in lag_values):
            raise ValueError("Todos los `lags` deben ser enteros positivos.")

        return lag_values

    def _build_scaler(self) -> BaseEstimator:
        if isinstance(self.scaler, str):
            key = self.scaler.lower()
            if key not in _SCALER_REGISTRY:
                raise ValueError(
                    f"Scaler '{self.scaler}' no soportado. Opciones: {list(_SCALER_REGISTRY)}"
                )
            return _SCALER_REGISTRY[key]()

        if isinstance(self.scaler, BaseEstimator):
            return clone(self.scaler)

        raise ValueError("`scaler` debe ser un string soportado o un estimador de scikit-learn.")

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
