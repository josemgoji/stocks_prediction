"""Utilidades de entrada/salida compartidas entre módulos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Carga un archivo YAML, devolviendo un diccionario vacío si está vacío."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró la configuración en {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"El contenido de {path} debe ser un mapeo YAML.")

    return data


__all__ = ["load_yaml"]
