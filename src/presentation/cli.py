"""CLI simple para ejecutar casos de uso del proyecto."""

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from src.application.use_cases.fetch_market_data import fetch_market_data
from src.training.feature_selection import load_selected_features
from src.training.train import run_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI para flujo NVIDIA stocks.")
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser("fetch-data", help="Descargar datos de mercado")
    fetch_parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/base/data.yaml"),
        help="Ruta al archivo de configuración YAML.",
    )
    fetch_parser.add_argument(
        "--sin-macro",
        action="store_true",
        help="Descargar solo datos de mercado, omitiendo series macroeconómicas.",
    )

    features_parser = subparsers.add_parser(
        "show-selected-features",
        help="Muestra la lista de features guardada tras la selección.",
    )
    features_parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Ruta al JSON generado por run_rfecv_selection.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Ejecuta la competencia de modelos definida en los YAML.",
    )
    train_parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("conf/base/data.yaml"),
        help="Ruta al YAML con configuración de datos.",
    )
    train_parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("conf/base/training.yaml"),
        help="Ruta al YAML con configuración de entrenamiento.",
    )

    return parser.parse_args()


def _load_fetch_config(path: Path, *, skip_macro: bool = False) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    save_cfg = config.get("save", {}) if isinstance(config.get("save"), dict) else {}
    save_path = save_cfg.get("path")
    macro_cfg = config.get("macro") if isinstance(config.get("macro"), dict) else None

    params: dict = {
        "ticker": config["ticker"],
        "start": datetime.fromisoformat(config["start"]),
        "end": datetime.fromisoformat(config["end"]),
        "interval": config.get("interval"),
        "save_path": Path(save_path) if save_path else None,
        "save_format": save_cfg.get("format", "parquet"),
        "auto_adjust": config.get("auto_adjust"),
    }

    if macro_cfg and not skip_macro:
        series = macro_cfg.get("series")
        if isinstance(series, dict) and series:
            params["macro_series"] = series
            if macro_cfg.get("api_key"):
                params["fred_api_key"] = str(macro_cfg["api_key"])
            if macro_cfg.get("fill_method"):
                params["macro_fill_method"] = str(macro_cfg["fill_method"])

    return params


def main() -> None:
    args = _parse_args()

    if args.command == "fetch-data":
        params = _load_fetch_config(args.config, skip_macro=args.sin_macro)
        dataset = fetch_market_data(**params)
        print(
            f"Descargados {len(dataset)} registros para {params['ticker']}. "
            f"Head:\n{dataset.head()}"
        )
    elif args.command == "show-selected-features":
        features = load_selected_features(args.artifact)
        print(
            "Features seleccionadas:\n"
            + "\n".join(f"- {feature}" for feature in features)
        )
    elif args.command == "train":
        outcome = run_training(
            data_config_path=args.data_config,
            training_config_path=args.training_config,
        )
        best = outcome.best_candidate
        print(
            "Mejor candidato: "
            f"{best.name} con {outcome.primary_metric}={best.metrics_val[outcome.primary_metric]:.4f}"
        )
        print("Métricas de validación:", best.metrics_val)
        print("Métricas de test:", best.metrics_test)
        if outcome.feature_selection.artifact_path:
            print(f"Features seleccionadas guardadas en: {outcome.feature_selection.artifact_path}")
    else:
        raise SystemExit("Comando no proporcionado. Usa --help para más detalle.")


if __name__ == "__main__":
    main()
