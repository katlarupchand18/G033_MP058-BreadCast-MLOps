# SCSE3040 MLOps Mini Project — Group G033, Problem MP058

**Project:** Bakery Bread Sales (Indore)  
**Faculty:** Amit Soni  
**Batch:** 71

## Group Members

| Enrollment Number | Name |
|-------------------|------|
| S24CSEU2683 | NishantMishraKumar |
| S24CSEU2695 | TanishqSingh |
| S24CSEU2612 | SHUBHAMSINGH |
| S24CSEU2700 | KATLARUPCHAND |

## Project Description

This project builds a complete MLOps pipeline for predicting daily bread sales (`loaves_sold`) for a chain of bakery outlets in Indore. The client, Indore Bakers' Guild, wants to reduce unsold inventory waste and avoid stockouts by accurately forecasting demand.

**Problem Type:** Supervised Regression  
**Target Variable:** `loaves_sold` (loaves)  
**Primary Metric:** Mean Absolute Error (MAE)

## Datasets

| File | Description | Rows |
|------|-------------|------|
| `data/raw/train.csv` | Training data (Jan–Jun 2026), messy | ~1,049 |
| `data/raw/later.csv` | Monitoring/retraining data (Jul–Aug 2026), clean | 300 |
| `data/raw/holdout_features.csv` | Final prediction data (Sep 2026), no target | 250 |

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── params.yaml
├── .gitignore
├── .dockerignore
├── Dockerfile
├── .dvc/
│   └── config
├── data/
│   └── raw/
│       ├── train.csv.dvc
│       ├── later.csv.dvc
│       └── holdout_features.csv
├── src/
│   └── bakery_mlops/
│       ├── __init__.py
│       ├── data_quality.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── predict.py
│       ├── monitoring.py
│       └── retrain.py
├── tests/
│   ├── test_data_quality.py
│   ├── test_preprocessing.py
│   └── test_api.py
├── app/
│   └── main.py
├── models/
│   └── model.joblib
├── reports/
│   └── findings.yaml
├── predictions/
│   └── holdout_predictions.csv
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── mlflow.db
└── .github/
    └── workflows/
        └── ci.yml
```

## Setup Instructions

### Prerequisites
- Python 3.12+
- Git
- DVC
- Docker (for containerization)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd G033_MP058
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Pull DVC-tracked data (if DVC remote configured):
   ```bash
   dvc pull
   ```
   Note: Raw CSV files are tracked by DVC, not Git. They are not included in the repository.

## Project Workflow

### Stage 1: GitHub, DVC, and Data Quality (COMPLETED)
- Initialized Git repository and configured DVC
- Tracked `data/raw/train.csv` and `data/raw/later.csv` with DVC
- Detected and documented all seven data quality issues in `reports/findings.yaml`
- Created reproducible detection code in `src/bakery_mlops/data_quality.py`

**Key Findings:**
- **Unit Mixup**: `price_rs` for `College Gate` outlet uses different unit (×100 factor)
- **Duplicate Rows**: 49 extra exact duplicate copies
- **Target Typos**: 15 rows with `loaves_sold` 10× too large (extra zero)
- No leaking columns, sentinel values, stuck sensors, or spelling variants detected

### Stage 2: Cleaning, MLflow, and Testing (IN PROGRESS)
- Build reproducible cleaning pipeline in `src/bakery_mlops/preprocessing.py`
- Train baseline models with MLflow tracking (`sqlite:///mlflow.db`)
- Register best model with `champion` alias
- Write automated tests (pytest + FastAPI TestClient)

### Stage 3: FastAPI, Docker, CI, Kubernetes (PENDING)
- FastAPI service with `/health` and `/predict` endpoints
- Pydantic validation for client rules (`shelf_hours ≤ 30`, `orders_previous_day ≥ 5`)
- Docker image `mp058-api:1.0` with `python:3.12-slim`
- GitHub Actions CI running pytest on every push
- Kubernetes manifests: Deployment `mp058-api` (4 replicas), Service `mp058-svc`

### Stage 4: Drift, Retraining, Holdout Predictions (PENDING)
- Population Stability Index (PSI) for all numeric features
- Group-wise MAE on `later.csv`
- Retrain v2 on train + first 70% of later.csv
- Promotion rule: v2 MAE ≤ 0.85 × v1 MAE on last 30%
- Final holdout predictions to `predictions/holdout_predictions.csv`

## Reproduction Instructions

### Run Data Quality Detection
```bash
source venv/bin/activate
python -m src.bakery_mlops.data_quality
```

### Run Tests (after Stage 2)
```bash
source venv/bin/activate
pytest tests/ -v
```

### Train Model (after Stage 2)
```bash
source venv/bin/activate
python -m src.bakery_mlops.train
```

### Start API Server (after Stage 3)
```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Build Docker Image (after Stage 3)
```bash
docker build -t mp058-api:1.0 .
docker run -p 8000:8000 mp058-api:1.0
```

## Data Quality Issues Summary

| Issue | Present | Details |
|-------|---------|---------|
| Leaking Column | ❌ No | All features available at prediction time |
| Unit Mixup | ✅ Yes | `price_rs` for `College Gate` ×100 (paise vs rupees) |
| Sentinel Value | ❌ No | No -1, -999, 999, 9999 found |
| Duplicate Rows | ✅ Yes | 49 extra copies |
| Stuck Sensor | ❌ No | No runs > 7 days |
| Target Typos | ✅ Yes | 15 rows with 10× values |
| Spelling Variants | ❌ No | Categories consistent across files |

## Original Data Checksums (SHA-256)

| File | SHA-256 |
|------|---------|
| `data/raw/train.csv` | `d02dae0740aa624fe95ee4e4b751b19734d8869559a956d0b22069ebab304977` |
| `data/raw/later.csv` | `3a7b5f2c0b90664aa4f573c74fba1b63a7881018a21ff8f53669b8c32894ef06` |
| `data/raw/holdout_features.csv` | `a03447913e82f234180ea61181a1c76aa6b6f80df011f5efa46ba2eb78c8e105` |

These checksums verify the original data files have not been modified.

## Submission

- **Repository:** Public GitHub repository (one per group)
- **Submission Form:** https://forms.cloud.microsoft/r/9ZPB1ZzcAK
- **Deadline:** 29 November 2026, 11:59 PM
- **Grace Period:** Until 4 December 2026, 11:59 PM

Each member must submit the same repository URL through the university form.

---

*This README is maintained as part of the SCSE3040 MLOps Mini Project requirements.*