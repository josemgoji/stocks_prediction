"""Configuración centralizada basada en Pydantic Settings."""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Carga las variables de entorno requeridas por el proyecto."""

    FRED_API_KEY: str

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
