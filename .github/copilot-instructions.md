# Sentiment Analysis CI/CD Codebase Guide

## Architecture Overview

This is a **sentiment analysis ML pipeline** with FastAPI serving and MLflow experiment tracking. Core workflow:

1. **Data Ingestion** (`src/ingest.py`) → Raw CSV in `data/raw/`
2. **Preprocessing** (`src/preprocess.py`) → Text cleaning, TF-IDF vectorization → Pickle files in `data/processed/`
3. **Training** (`src/train.py`) → LogisticRegression with MLflow tracking → `models/model.pkl`
4. **Inference** (`src/inference.py`) → Loads model + vectorizer, returns sentiment label + confidence
5. **API** (`api/app.py`) → FastAPI endpoint `/predict` that calls `src/inference.py`

**Critical**: The vectorizer (`models/vectorizer.pkl`) must be fit during preprocessing and reused identically in inference (see lines 29-31 in `src/inference.py`). Text cleaning logic must match exactly between preprocessing and inference.

## Key Design Decisions

- **Configuration-driven**: All paths, hyperparameters in `config/config.yaml` (loaded via YAML)
- **MLflow integration**: Logs params/metrics to `mlflow/` directory; `docker-compose.yaml` runs MLflow UI on port 5000
- **Volume mounting**: Models directory mounted in Docker (`./models:/app/models`) to persist across container restarts without retraining
- **Sentiment labels**: Binary classification (0=Negative, 1=Positive) mapped to readable labels in `src/inference.py` line 23

## Development Workflows

### Training Pipeline
```bash
# Full pipeline: ingest → preprocess → train
python src/ingest.py
python src/preprocess.py
python src/train.py
```

### API Development
```bash
# Run locally (requires trained model in models/)
uvicorn api.app:app --reload

# Run via Docker
docker-compose up --build
```

### Testing
```bash
pytest tests/
# Tests use TestClient (FastAPI built-in) to mock API calls without server
```

## Project-Specific Patterns

### Config Management
All dynamic values (paths, hyperparams) in `config/config.yaml`. Access via:
```python
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
# Use config['data']['raw_path'], config['train']['C'], etc.
```

### Data Serialization
- Raw: CSV (`data/raw/sentiment_data.csv`)
- Processed: Pickle (`data/processed/*.pkl`) - vectorized features only
- Models: Pickle (`models/model.pkl`, `models/vectorizer.pkl`)

### Artifact Preservation
- **Vectorizer**: Saved in preprocessing (line 39 in `src/preprocess.py`), loaded globally in `src/inference.py` line 5-7 to ensure identical transforms
- **Model**: Logged to MLflow (line 47 in `src/train.py`) AND saved locally (line 49) for API use

## Cross-Component Boundaries

| Component | Input | Output | Dependencies |
|-----------|-------|--------|--------------|
| `src/preprocess.py` | Raw CSV | `X_train.pkl`, `y_train.pkl`, `vectorizer.pkl` | `config.yaml` |
| `src/train.py` | Vectorized pickles | `model.pkl`, MLflow artifacts | `src/preprocess.py` output |
| `src/inference.py` | Text string | JSON (label, confidence) | `model.pkl`, `vectorizer.pkl` |
| `api/app.py` | HTTP POST body | JSON response | `src/inference.py` |

## Common Pitfalls to Avoid

1. **Vectorizer mismatch**: Never refit vectorizer in inference. Load the same pickled instance from preprocessing.
2. **Text cleaning divergence**: Keep cleaning logic (`src/preprocess.py` line 13-16) synchronized with `src/inference.py` line 14-17.
3. **Config inconsistency**: Don't hardcode paths; always read from `config.yaml`.
4. **Model availability**: API expects `models/model.pkl` and `models/vectorizer.pkl` to exist. Docker volumes ensure they persist.

## External Dependencies

- **MLflow 2.7.1**: For experiment tracking, logging params/metrics, model registry
- **FastAPI 0.103.1**: Web framework, uses Pydantic for request validation
- **scikit-learn 1.3.0**: TfidfVectorizer, LogisticRegression
- **pandas 2.0.3**: Data manipulation (CSV read, pickle operations)

## Files to Know

- [config/config.yaml](config/config.yaml) - Single source of truth for paths and hyperparameters
- [src/train.py](src/train.py) - Training loop with MLflow integration
- [src/inference.py](src/inference.py) - Global model/vectorizer loading; called by API
- [api/app.py](api/app.py) - API entry point; imports `predict_sentiment` from `src.inference`
- [tests/test_api.py](tests/test_api.py) - TestClient usage pattern for API testing
