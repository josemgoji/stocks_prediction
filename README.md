# Stocks Prediction

Repositorio de una prueba técnica centrada en un pipeline de series financieras con MLflow. El objetivo es disponer de un flujo reproducible para descargar datos, entrenar modelos, persistir artefactos y servir predicciones con una API mínima.

## Estructura principal

```
conf/                   # Configuraciones YAML (datasets, entrenamiento, modelo)
docker/                 # Dockerfiles para MLflow y API
notebooks/              # Exploración adicional (EDA)
src/
  application/          # API FastAPI (`api.py`) y casos de uso de negocio
  features/             # Transformadores y builders de features
  models/               # Factorías de estimadores y pipelines sklearn
  registry/             # Cliente MLflow
  resources/            # Adapters hacia fuentes externas (Yahoo Finance, FRED)
  serving/              # Scripts de predicción y carga de modelos
  training/             # Dataset split, selección de modelos y entrenamiento
  utils/                # utilidades compartidas (`load_yaml`)
artifacts/              # Resultados locales (feature selection, tuning, etc.)
docker-compose.yml      # Orquestación de MLflow + MinIO + Postgres + API
```

## Dependencias locales

1. Crear y activar un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Instalar dependencias de la API (incluye sklearn, mlflow, etc.):
   ```bash
   pip install fastapi uvicorn[standard] numpy pandas scikit-learn lightgbm mlflow boto3 pyyaml yfinance fredapi
   ```

## Entrenamiento

Ejecuta el pipeline declarativo utilizando los YAML de `conf/base`:

```bash
python -m src.training.train \
  --data-config conf/base/data.yaml \
  --training-config conf/base/training.yaml
```

Al terminar, se registrará el mejor modelo en MLflow (según `primary_metric`), se guardarán métricas y se crearán artefactos en `artifacts/`.

## Serving sin Docker

### API FastAPI

1. Arranca la API en local:
   ```bash
   uvicorn src.application.api:app --host 0.0.0.0 --port 8000 --reload
   ```
2. Documentación interactiva: `http://127.0.0.1:8000/docs`
3. Endpoints disponibles:
   - `POST /data/fetch`: descarga datos según `data.yaml`.
   - `POST /training/run`: lanza el entrenamiento con `training.yaml`.
   - `POST /prediction/run`: carga el último modelo en MLflow y devuelve la predicción más reciente.

### Script CLI

Si prefieres un script directo:

```bash
python src/serving/generate_prediction.py \
  --data-config conf/base/data.yaml \
  --training-config conf/base/training.yaml \
  --prediction-date 2025-01-15
```

## Infraestructura con Docker Compose

El `docker-compose.yml` levanta:

- `postgres` (backend MLflow)
- `minio` (artifacts S3)
- `mlflow` (servidor MLflow)
- `api` (FastAPI)

Comandos útiles:

```bash
# Construir y levantar todos los servicios
docker compose up --build -d

# Listar estado de los contenedores
docker compose ps

# Detener todo
docker compose down
```

Endpoints:
- MLflow UI: `http://127.0.0.1:8888`
- API: `http://127.0.0.1:8000/docs`

