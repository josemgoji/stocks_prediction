"""Carga y particiona datasets para entrenamiento siguiendo configuración temporal."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


@dataclass(slots=True)
class TemporalSplitResult:
    """Representa los splits train/validation/test para features y target."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    label_column: str
    horizon: int = 1

    @property
    def train_frame(self) -> pd.DataFrame:
        """DataFrame de entrenamiento con la columna objetivo incluida."""
        return _join_features_target(self.X_train, self.y_train)

    @property
    def val_frame(self) -> pd.DataFrame:
        """DataFrame de validación con la columna objetivo incluida."""
        return _join_features_target(self.X_val, self.y_val)

    @property
    def test_frame(self) -> pd.DataFrame:
        """DataFrame de test con la columna objetivo incluida."""
        return _join_features_target(self.X_test, self.y_test)


def load_dataset(config: Mapping[str, object]) -> pd.DataFrame:
    """Carga un dataset a partir de configuración declarativa.

    La configuración acepta cualquiera de las siguientes claves:
        - `path`: ruta directa al archivo a cargar.
        - `save.path`: ruta definida en `conf/base/data.yaml` para los datos crudos.

    Args:
        config: Diccionario de configuración (p. ej. `data.yaml`).

    Returns:
        DataFrame con el dataset completo, indexado por fecha si la columna existe.
    """
    path = _resolve_path(config)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {path}")

    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path, parse_dates=True, index_col=0)
    else:
        raise ValueError(
            f"Formato de dataset no soportado para {path.suffix}. Usa parquet o csv."
        )

    frame = frame.sort_index()
    return frame


def temporal_train_val_test_split(
    frame: pd.DataFrame,
    *,
    target_column: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    gap: int = 0,
    horizon: int = 1,
) -> TemporalSplitResult:
    """Genera splits temporales manteniendo el orden cronológico.

    Args:
        frame: DataFrame original ordenado por tiempo.
        target_column: Columna objetivo.
        train_size: Proporción de datos para entrenamiento.
        val_size: Proporción para validación.
        test_size: Proporción para test (se ajusta si no suma 1).
        gap: Observaciones a omitir entre train/val o val/test para evitar fuga.
        horizon: Pasos futuros a predecir (t+h). `h=1` corresponde al siguiente periodo.

    Returns:
        `TemporalSplitResult` con `X` y `y` para cada partición.
    """
    if target_column not in frame.columns:
        raise KeyError(f"Columna objetivo '{target_column}' no está en el DataFrame.")

    if horizon < 1:
        raise ValueError("`horizon` debe ser un entero positivo.")

    label_name = f"{target_column}_target_h{horizon}"
    shifted_target = frame[target_column].shift(-horizon)
    shifted_target.name = label_name

    valid_mask = shifted_target.notna()
    frame_aligned = frame.loc[valid_mask].copy()
    shifted_target = shifted_target.loc[valid_mask]

    total = len(frame_aligned)
    train_end_idx = int(total * train_size)
    val_end_idx = train_end_idx + int(total * val_size)

    if train_end_idx <= 0 or val_end_idx <= train_end_idx:
        raise ValueError(
            "Los tamaños de split producen particiones inválidas. Ajusta train/val/test."
        )

    train_slice_end = max(train_end_idx - gap, 0)
    val_slice_start = min(train_end_idx + gap, total)
    val_slice_end = min(val_end_idx, total)
    test_slice_start = min(val_end_idx + gap, total)

    train_frame = frame_aligned.iloc[:train_slice_end]
    val_frame = frame_aligned.iloc[val_slice_start:val_slice_end]
    test_frame = frame_aligned.iloc[test_slice_start:]

    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError(
            "Alguna de las particiones resultó vacía. Revisa los porcentajes o el tamaño del dataset."
        )

    y_train = shifted_target.iloc[:train_slice_end]
    y_val = shifted_target.iloc[val_slice_start:val_slice_end]
    y_test = shifted_target.iloc[test_slice_start:]

    X_train = train_frame
    X_val = val_frame
    X_test = test_frame

    return TemporalSplitResult(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        label_column=label_name,
        horizon=horizon,
    )


def _resolve_path(config: Mapping[str, object]) -> Path:
    """Obtiene la ruta del dataset a partir de un diccionario de configuración."""
    if "path" in config and config["path"]:
        return Path(str(config["path"]))

    save_cfg = config.get("save")
    if isinstance(save_cfg, Mapping) and "path" in save_cfg:
        return Path(str(save_cfg["path"]))

    raise KeyError(
        "No se pudo resolver la ruta del dataset. Define `path` o `save.path` en la configuración."
    )


def _join_features_target(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Combina features y target garantizando alineación por índice."""
    frame = pd.concat([X, y], axis=1)
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    return frame
