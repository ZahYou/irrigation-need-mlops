# ml-tabular-classification-pipeline

> Production-grade ML pipeline from Kaggle to deployment — DVC · MLflow · FastAPI · Docker

*README filled in during Week 4. Placeholder sections below.*

## Quick start

```bash
git clone <repo-url>
pip install -e ".[dev]"
dvc pull            # fetch data (requires DVC remote access)
python -m src.models.train
uvicorn src.serving.app:app --reload
```

## Tech stack

| Tool | Why |
|------|-----|
| DVC | Reproducible data versioning without storing CSVs in git |
| MLflow | Experiment tracking + model registry, free local setup |
| LightGBM / XGBoost / CatBoost | GBDT ensemble — best out-of-box performance on tabular data |
| FastAPI | Async, typed, auto-docs — production-ready serving in <100 lines |
| Docker | Reproducible runtime; eliminates "works on my machine" |
| GitHub Actions | CI on every push — lint → test → docker build |
| ruff | 10-100× faster than flake8, single tool for lint + format |

## Architecture

*(Mermaid diagram added in Week 4)*

## Results

*(Metrics table added after Week 2 training run)*
