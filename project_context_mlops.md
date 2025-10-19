# 🧠 Project Context — Clean ML/MLOps Pipeline (NVIDIA Example)

## 🎯 Objective
Build a clean, modular, and reproducible **machine learning pipeline** (starting with NVIDIA data) under a **Clean Architecture / MLOps** approach.

The project aims to:
- Structure a proper **ML development flow** with clear separation of concerns.
- Use **on-the-fly feature engineering** (features are generated at runtime during training and inference).
- Apply **feature selection inside the model pipeline** (relearned in each retrain).
- Support **model competition** (multiple candidates with hyperparameter tuning via Optuna).
- Register and track everything with **MLflow**.
- Serve models through a simple **batch or FastAPI endpoint** (optional).
- Use **Docker only for MLflow** — no full containerization or orchestration yet.

---

## 🏗️ General Architecture

### Layers Overview
| Layer | Purpose |
|-------|----------|
| **domain/** | Core business logic and abstractions (entities & ports) — no dependencies. |
| **infrastructure/** | Adapters to external systems (Yahoo Finance, MLflow, file storage, logging). |
| **features/** | Feature engineering functions (lags, rolling, transformations). |
| **models/** | Model factory and pipeline definitions (preprocessing + estimator + selector). |
| **training/** | Dataset handling, training loops, hyperparameter tuning, and model competition logic. |
| **evaluation/** | Metrics and validation (e.g., RMSE, backtesting). |
| **registry/** | MLflow experiment and model registry integration. |
| **serving/** | Model inference logic (batch and optional API). |
| **presentation/** | CLI interface for executing flows (training, tuning, inference, etc.). |
| **conf/** | YAML configuration (data, model, training, candidates). |

---

## 🧩 Configuration (YAML-based)

All parameters (paths, hyperparameters, model definitions, etc.) are stored as versionable YAMLs:

- **data.yaml** → paths to train/test datasets.  
- **model.yaml** → model type, target column, feature policy (`__AUTO__` or list), and hyperparameters.  
- **training.yaml** → experiment name, metric, validation split, and seed.  
- **model_candidates.yaml** (optional) → defines multiple models + search spaces for competition.

Configuration is declarative and externalized (no hard-coded params in Python), ensuring reproducibility and easier experiment tracking.

---

## ⚙️ Design Decisions

### 1. Feature Engineering
- Performed **on-the-fly** during training and inference.
- Implemented in `features/builders.py` (lags, rolling windows, etc.).
- No materialization for now (no feature store, no parquet “gold layer”).
- Optionally, in future phases, features can be persisted using a **raw → stage → analytics** approach.

### 2. Feature Selection
- Implemented **inside the pipeline** using steps like `SelectKBest`, `RFECV`, or model-based selection.
- This makes selection reproducible, avoids data leakage, and adapts to every retraining.
- No static feature list stored yet; the YAML will use `features: "__AUTO__"` and exclusions.

### 3. Model Competition
- Multiple candidates (e.g., LightGBM, XGBoost, RandomForest) are defined in YAML.
- Each candidate runs Optuna optimization with `MLflowCallback`.
- MLflow records parameters, metrics, and artifacts for each trial.
- A `model_selection.py` module compares the best trial of each candidate and picks the top performer based on the primary metric.

### 4. MLflow Integration
- Used for:
  - Tracking experiments (params, metrics, artifacts).
  - Storing models (`mlflow.sklearn.log_model`).
  - Registering the best model in the **Model Registry**.
- Dockerized MLflow server used locally for UI and tracking.

### 5. Serving
- **Batch serving** preferred initially (simpler, reproducible).
- **API serving** (FastAPI) can be added later for online predictions.
- Both load the “Production” model from the MLflow Model Registry.

### 6. Orchestration
- No orchestration or scheduling yet.
- Retraining or scoring can be triggered manually, via CLI, or a basic cron/CI job.
- In the future, can migrate to Prefect/Airflow or Azure pipelines.

---

## 🧱 Directory Structure (Light & Scalable)

```
conf/
└─ base/
   ├─ data.yaml
   ├─ model.yaml
   └─ training.yaml

src/
├─ domain/
│  ├─ entities.py
│  └─ ports.py
├─ infrastructure/
│  ├─ adapters/
│  │  └─ yahoo_finance_adapter.py
│  ├─ config.py
│  └─ logging_setup.py
├─ features/
│  ├─ builders.py
│  └─ transformers.py
├─ models/
│  ├─ estimators.py
│  └─ pipelines.py
├─ training/
│  ├─ dataset.py
│  ├─ trainer.py
│  ├─ tuning.py
│  ├─ model_selection.py
│  └─ train.py
├─ evaluation/
│  ├─ metrics.py
│  └─ backtesting.py
├─ registry/
│  └─ mlflow_client.py
├─ serving/
│  ├─ batch.py
│  ├─ api.py
│  └─ schemas.py
└─ presentation/
   └─ cli.py
```

---

## 🔁 Workflow Overview

1. **Train pipeline (`train.py`):**
   - Load YAML configs (`data`, `model`, `training`).
   - Load training dataset.
   - Apply **feature builders** dynamically.
   - Resolve features (`__AUTO__` + exclusions).
   - Split data into train/validation.
   - Execute model training or full competition.
   - Log everything to MLflow and (optionally) register the best model.

2. **Model selection (`model_selection.py`):**
   - Runs candidates with Optuna and MLflow tracking.
   - Chooses the best run based on validation metric.
   - Returns `(best_model_name, best_run_id)`.

3. **MLflow registry (`mlflow_client.py`):**
   - Registers the best model from the experiment.
   - Handles promotion to `Staging` or `Production`.

4. **Serving (batch or API):**
   - Loads the latest “Production” model.
   - Applies the same preprocessing pipeline (since it’s inside the pipeline).
   - Outputs predictions.

---

## 🔍 Key Principles and Best Practices

- **On-the-fly feature engineering:** recalculated every retrain; keeps pipeline consistent.
- **Feature selection inside pipeline:** reproducible and leakage-safe.
- **Declarative YAML configs:** parameters, datasets, and model types externalized.
- **MLflow everywhere:** full experiment traceability (metrics, params, artifacts, models).
- **Simple structure:** no orchestration or containers except for MLflow.
- **CLI-first interface:** commands for fetch, train, tune, predict, serve.
- **Scalable foundation:** can evolve later into Prefect/Airflow and feature store layers.

---

## 🚀 Next Steps

1. Create the folder structure and base YAMLs.  
2. Implement basic feature builders (lags, rolling stats).  
3. Define initial pipeline (imputer → scaler → selector → model).  
4. Build training flow with MLflow logging.  
5. Add Optuna + model competition.  
6. Add batch serving.  
7. Optionally extend to API serving.

---

## 🐳 Docker & MLflow

- `docker-compose.yml` levanta `mlflow-db` (Postgres) y `mlflow` (imagen personalizada con `psycopg2`).
- Artefactos se persisten en `./docker/mlflow/artifacts` y la base en `./docker/postgres/data`.
- Ejecuta `docker compose up -d` para iniciar ambos servicios y `docker compose down` para detenerlos.
- El `tracking_uri` en `conf/base/training.yaml` apunta a `http://127.0.0.1:8888` y el flujo de entrenamiento registra automáticamente el mejor pipeline en MLflow Registry; además se inyectan las credenciales de MinIO para que los artefactos lleguen al bucket `mlflow` sin variables de entorno manuales.
- El módulo `src/serving/model_loader.py` expone helpers para cargar `models:/StocksPredictionModel/Latest` (o cualquier versión/stage) directamente desde el registry antes de servir.

---

✅ **Outcome:**  
A clean, modular ML project ready to train, track, and compare multiple models end-to-end, with MLflow integration and extendable toward full MLOps maturity later (orchestration, versioned features, CI/CD, etc.).


# TO DO:
Configuraciones vivas: completa conf/base/data.yaml, model.yaml y training.yaml con rutas, features y parámetros reales; suma conf/base/model_candidates.yaml si vas a correr competición.
Ingesta y validación: crea un adaptador (src/infrastructure/adapters/yahoo_finance_adapter.py) que descargue precios y guárdalos versionados (p.ej. DVC o al menos un timestamp en data/raw/). Añade validaciones con pandera o great_expectations antes de entrenar.
Dataset + features: implementa src/training/dataset.py para orquestar splits con walk-forward/backtesting y construye los transformadores en src/features/ (lags, retornos, rolling stats).
Pipelines y modelos: en src/models/pipelines.py arma un Pipeline de sklearn (imputer → scaler → selector → estimador). Define estimadores y espacios de hiperparámetros en estimators.py.
Entrenamiento y tuning: rellena src/training/trainer.py, tuning.py y model_selection.py para entrenar, loggear a MLflow, y comparar candidatos (Optuna + MLflowCallback).
Evaluación: implementa src/evaluation/metrics.py (RMSE, MAPE, Sharpe, max drawdown) y backtesting.py con un esquema walk-forward realista.
Registro y serving: completa src/registry/mlflow_client.py para promover modelos, y src/serving/batch.py/api.py para cargar el modelo “Production” desde MLflow y predecir.
Observabilidad: establece logging estructurado en src/infrastructure/logging_setup.py, y prepara métricas de monitoreo (latencia, drift, performance financiera) para cuando sirvas.
Operatividad: añade un pyproject.toml o requirements.txt, scripts (Makefile o invoke) para tareas comunes y un CLI en src/presentation/cli.py. Considera un flujo CI (GitHub Actions) que ejecute linters/tests y registre modelos de prueba.
Pruebas: crea tests/ con unit tests de features, pipelines y adaptadores; usa pytest con fixtures de datos sintéticos.
