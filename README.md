# ml-tabular-classification-pipeline

> Production-grade ML pipeline from Kaggle to deployment — DVC · MLflow · FastAPI · Docker

**Task**: 3-class irrigation need prediction — `Low / Medium / High`  
**Data**: Kaggle Playground S6E4 — tabular agricultural sensor data  
**Best dev CV (3-fold, 10k sample)**: CatBoost OOF BA = 0.9566

---

## Tech stack

| Tool | Role |
|------|------|
| DVC | Reproducible pipeline — tracks data + artifact versions without storing binaries in git |
| MLflow | Experiment tracking — one run per model, logs params / per-fold metrics / artifacts |
| XGBoost / LightGBM / CatBoost | GBDT ensemble — best out-of-box on tabular data |
| FastAPI + Pydantic | Typed async REST API — auto-docs at `/docs`, input validation for free |
| Docker | Reproducible runtime — multi-stage build, slim final image |
| GitHub Actions | CI on every push — lint → test → docker build |
| ruff | Lint + format in one tool |

---

## Repository layout

```
Project_One/
│
├── configs/
│   └── train_config.yaml        # single source of truth for all pipeline params
│
├── data/
│   ├── raw/                     # train.csv, test.csv, sample_submission.csv  (DVC-tracked)
│   └── processed/               # pickled features + artifacts               (DVC-tracked)
│
├── src/
│   ├── data/
│   │   ├── feature_engineering.py  # full feature pipeline (795 features)
│   │   └── run_features.py         # DVC "features" stage entry-point
│   ├── models/
│   │   └── baseline_runner.py      # DVC "train" stage — CV + MLflow logging
│   └── serving/
│       └── app.py                  # FastAPI app — /predict + /health
│
├── tests/                       # pytest suite (API + features + preprocess)
├── saved_models/                # per-fold .pkl files (git-ignored)
├── saved_oof/                   # OOF numpy arrays   (git-ignored)
├── submissions/                 # Kaggle submission CSVs
├── feature_importance/          # FI plots per model
├── mlruns/                      # MLflow tracking store (git-ignored)
│
├── dvc.yaml                     # pipeline DAG definition
├── Dockerfile                   # multi-stage build
├── docker-compose.yaml
└── pyproject.toml               # dependencies + ruff + pytest config
```

---

## Pipeline architecture

```
data/raw/train.csv ──┐
data/raw/test.csv  ──┤
configs/           ──┤
                     ▼
              [ features stage ]          python -m src.data.run_features
              feature_engineering.py
                     │
                     │  data/processed/
                     │    X_train.pkl  X_test.pkl  y_train.pkl
                     │    cat_cols.json  num_cols.json
                     │    feature_artifacts.pkl   ← inference lookup tables
                     ▼
              [ train stage ]             python -m src.models.baseline_runner
              baseline_runner.py
                     │
                     │  saved_models/  ← fold .pkl files + label_encoders.pkl
                     │  saved_oof/     ← OOF probabilities + classes
                     │  submissions/   ← Kaggle CSV per model
                     │  mlruns/        ← MLflow experiment store
                     ▼
              [ serve ]                   uvicorn src.serving.app:app
              app.py loads all artifacts at startup,
              ensembles all fold models per request
```

---

## Feature engineering summary (`src/data/feature_engineering.py`)

795 features are built in the following order (order matters — ordinals feed domain features):

| Step | What it creates |
|------|----------------|
| Bucket features | `Soil_pH_Band`, `Moisture_Band`, `Rainfall_Band`, `Temperature_Band` |
| Ordinal encodings | `Crop_Growth_Stage_Ordinal`, `Season_Ordinal` |
| Interaction features | 15 ratios/products — `Dryness_Index`, `Water_Availability_Index`, etc. |
| Irrigation domain features | `Net_Water_Balance`, `Water_Stress_Index`, `High_Stress_Flag`, etc. |
| Cross-category features | 7 combos — `Soil_Crop_Combo`, `Season_Region_Combo`, etc. |
| Pairwise categoricals | All C(8,2) = 28 category pairs |
| Missingness features | Per-column missing flags + `Missing_Pattern_Code` |
| Rounding / digit features | Decimal-precision signals on 4 numeric cols |
| Anchor combos | `{cat}_factorized × {anchor_numeric}` dense hybrid features |
| Groupby numeric stats | mean/std/count/nunique/min/max per 11 group keys |
| Groupby quantiles | p10/p25/p50/p75/p90 per (Season, Region, Crop, Soil) × 2 cols |
| Groupby ratio features | count_per_nunique, std_per_count |
| OOF target encoding | mean/std/count + per-class rate, leakage-safe, per 50+ group keys |

**Inference path**: `build_feature_artifacts()` serialises all groupby lookup tables and TE maps to `feature_artifacts.pkl` after training. `build_single_inference_features(row, artifacts)` applies the full pipeline to a single request dict using those pre-computed lookups — no training data needed at serve time.

---

## Config reference (`configs/train_config.yaml`)

```yaml
dev_mode: true        # human-readable flag — set false for production run

data:
  dev_sample: 10000   # rows sampled from train (stratified). null = full data
  dev_test_sample: 5000  # rows sampled from test. null = full data

training:
  seed: 42
  n_splits: 3         # CV folds. use 5 for full run
  models:             # comment out any model to skip it
    - xgb
    - lgbm
    - catboost
    # - logreg

mlflow:
  experiment_name: irrigation-need
  tracking_uri: mlruns
```

**To switch from dev to full run**, change these four fields:
```yaml
dev_mode: false
data:
  dev_sample: null
  dev_test_sample: null
training:
  n_splits: 5
  models: [xgb, lgbm, catboost, logreg]
```

---

## Setup

```bash
# 1. Clone and install
git clone <repo-url>
cd Project_One
pip install -e ".[dev]"

# 2. Verify data is present
ls data/raw/          # should show train.csv  test.csv  sample_submission.csv
```

---

## Workflow commands

### Run the full pipeline (DVC)

```bash
# Run features stage + train stage in dependency order
dvc repro

# Force re-run even if nothing changed
dvc repro --force

# Run only the features stage
dvc repro features

# Run only the train stage
dvc repro train

# Check what would run without executing
dvc status
```

> DVC skips stages whose inputs haven't changed (config, source files, data).
> Changing `dev_sample`, `n_splits`, or any model param in `train_config.yaml`
> invalidates the downstream stage and triggers a re-run.

### Run stages manually (without DVC)

```bash
# Features stage — builds processed/ artifacts + feature_artifacts.pkl
python -m src.data.run_features

# Train stage — trains all enabled models, logs to MLflow
python -m src.models.baseline_runner
```

### MLflow experiment tracking

```bash
# Start the UI (keep this terminal open, or run in background)
mlflow ui --backend-store-uri mlruns --port 5000

# Open in browser
# http://127.0.0.1:5000
```

You will see the `irrigation-need` experiment with one run per model:

| Run | OOF Balanced Accuracy |
|-----|-----------------------|
| `xgb_baseline_v001` | 0.9518 |
| `lgbm_baseline_v001` | 0.9484 |
| `cat_baseline_v001` | **0.9566** |

Each run stores: params (seed, n_splits, hyperparams, data sizes), `fold_ba` metric at steps 0/1/2, `oof_ba` summary metric, submission CSV and feature importance plot as artifacts.

### FastAPI prediction server

```bash
# Start the server (requires saved_models/ and data/processed/ to be populated)
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# Interactive API docs
# http://localhost:8000/docs

# Health check — shows how many fold models are loaded
curl http://localhost:8000/health

# Example prediction (partial input — missing fields are imputed)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Soil_Moisture": 25,
    "Temperature_C": 35,
    "Rainfall_mm": 5,
    "Humidity": 40,
    "Soil_Type": "Loamy",
    "Crop_Type": "Rice",
    "Season": "Kharif",
    "Crop_Growth_Stage": "Flowering",
    "Previous_Irrigation_mm": 10,
    "Field_Area_hectare": 2.5
  }'

# Response:
# {"predicted_class": "Medium", "probabilities": {"Low": 0.018, "Medium": 0.636, "High": 0.346}, "models_used": 9}
```

The server ensembles all available fold models (9 = 3 models × 3 folds).
All 19 input fields are optional — missing numerics are imputed with training medians,
missing categoricals are treated as "Missing" (a seen category during training).

### Docker

```bash
# Build the image
docker build -t ml-pipeline:latest .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/data/processed:/app/data/processed \
  -v $(pwd)/saved_models:/app/saved_models \
  ml-pipeline:latest

# Or with docker-compose
docker-compose up
```

### Tests and linting

```bash
# Run test suite
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Lint + format check
ruff check src/ tests/

# Auto-fix lint issues
ruff check src/ tests/ --fix
```

### CI (GitHub Actions)

Triggered on push to `main` or `dev`, and on PRs to `main`:
```
lint (ruff) → test (pytest + coverage) → docker build (smoke test)
```

See `.github/workflows/ci.yml`.

---

## Model details

### Training (`src/models/baseline_runner.py`)

- **CV**: `StratifiedKFold` — preserves class balance across folds
- **Metric**: Balanced Accuracy — handles class imbalance (High is rare)
- **Early stopping**: 50 rounds on validation mlogloss for all tree models
- **Per-fold outputs**: fold `.pkl` model, OOF proba slice, per-fold BA logged to MLflow
- **Caching**: if `saved_oof/{name}_proba.npy` exists, the model is not re-trained (use `rm saved_oof/*.npy` to force re-train)

| Model | Encoding | Class weighting |
|-------|----------|----------------|
| XGBoost | Label-encoded ints | `compute_sample_weight("balanced")` |
| LightGBM | pandas `category` dtype | `class_weight="balanced"` |
| CatBoost | pandas `category` dtype | `auto_class_weights="Balanced"` |
| LogisticRegression | Label-encoded + StandardScaler | `class_weight="balanced"` |

### Serving (`src/serving/app.py`)

On startup the app loads:
1. `data/processed/feature_artifacts.pkl` — groupby lookup tables + TE maps + training medians
2. `saved_models/label_encoders.pkl` — fitted LabelEncoders for XGB/LogReg
3. All `saved_models/{prefix}_fold*.pkl` files for each model type

At predict time:
1. Raw input dict → `build_single_inference_features()` → 795-column feature row
2. Feature row encoded 3 ways: label-encoded (XGB/LogReg), category dtype (LGBM), string (CatBoost)
3. All 9 fold models predict probabilities independently
4. Probabilities are averaged → argmax → `Low / Medium / High`

---

## Known issues / next steps

- `src/models/train.py` — MLflow-wired training entrypoint is a stub (Week 2 item)
- `src/models/export.py` — ONNX export is a stub (Week 3 item)
- `tests/test_api.py` — API tests are a placeholder pending serving implementation
- DVC remote not configured — `dvc push/pull` requires setting up S3/GCS/Azure in `.dvc/config`
- MLflow filesystem backend deprecated Feb 2026; migrate to SQLite with:
  ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  ```
  and update `train_config.yaml` → `tracking_uri: sqlite:///mlflow.db`
