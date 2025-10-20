"""Cliente sencillo para obtener series macroeconómicas desde FRED."""

from datetime import datetime
from typing import Mapping, Optional

import pandas as pd
from fredapi import Fred

from src.settings.env import settings


class FredResource:
    """Envuelve `fredapi.Fred` para descargar y alinear series macroeconómicas."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or settings.FRED_API_KEY
        if not key:
            raise ValueError(
                "Proporciona una API key para FRED (argumento `api_key` "
                "o variable de entorno FRED_API_KEY)."
            )
        self._client = Fred(api_key=key)

    def fetch_series(
        self,
        series: Mapping[str, str] | list[str],
        *,
        index: Optional[pd.DatetimeIndex] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fill_method: Optional[str] = "ffill",
    ) -> pd.DataFrame:
        """Descarga varias series y opcionalmente las alinea a un índice dado."""
        if isinstance(series, Mapping):
            items = list(series.items())
        else:
            items = [(sid, sid) for sid in series]

        frames: list[pd.DataFrame] = []
        for fred_id, column_name in items:
            data = self._client.get_series(
                fred_id,
                observation_start=start,
                observation_end=end,
            )
            if data is None:
                raise ValueError(f"No se obtuvo información para la serie {fred_id}.")

            series_df = pd.DataFrame(data, columns=[column_name])
            if series_df.empty:
                raise ValueError(f"La serie {fred_id} regresó datos vacíos.")

            # FRED regresa un índice naïve; lo localizamos en UTC para combinar con datos de mercado.
            series_df.index = pd.DatetimeIndex(series_df.index).tz_localize("UTC")
            frames.append(series_df)

        macro_df = pd.concat(frames, axis=1)

        if index is None:
            return macro_df

        target_index = pd.DatetimeIndex(index)
        if target_index.tz is None:
            target_index = target_index.tz_localize("UTC")

        aligned = macro_df.reindex(target_index)
        if fill_method == "ffill":
            aligned = aligned.ffill()
        elif fill_method == "bfill":
            aligned = aligned.bfill()
        elif fill_method:
            aligned = aligned.fillna(method=fill_method)

        return aligned
