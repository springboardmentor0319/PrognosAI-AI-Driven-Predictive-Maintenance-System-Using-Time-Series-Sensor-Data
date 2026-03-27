# PrognosAI: AI-Driven Predictive Maintenance (RUL)

PrognosAI predicts Remaining Useful Life (RUL) of turbofan engines using machine learning models trained on NASA C-MAPSS data. It provides:

- FastAPI prediction endpoints for single and batch inference
- PostgreSQL storage for predictions and evaluation history
- Grafana dashboards for fleet health, engine-level status, and model performance
- Dockerized deployment for one-command local setup

Stack: Python, XGBoost, FastAPI, PostgreSQL, Grafana, Docker

## Index

1. [Project Description](#project-description)
2. [Architecture and Components](#architecture-and-components)
3. [Installation](#installation)
4. [Run with Docker (Recommended)](#run-with-docker-recommended)
5. [Run Without Docker (Manual Setup)](#run-without-docker-manual-setup)
6. [API Usage](#api-usage)
7. [Populate Dashboard Data](#populate-dashboard-data)
8. [Project Structure](#project-structure)
9. [Dataset](#dataset)
10. [Training and Retraining Models](#training-and-retraining-models)
11. [Troubleshooting](#troubleshooting)
12. [License and Credits](#license-and-credits)

## Project Description

The goal of this project is to estimate how many operating cycles are left before an engine fails. This helps maintenance teams move from reactive maintenance to predictive maintenance.

What the system does:

- Accepts latest-cycle sensor data and predicts RUL
- Classifies health status as HEALTHY, WARNING, or CRITICAL
- Stores predictions in PostgreSQL for monitoring and historical analysis
- Visualizes fleet trends and model quality in Grafana

Primary data source: NASA C-MAPSS subsets FD001-FD004.

## Architecture and Components

- API service: `scripts/app.py` (FastAPI)
- Model training: `scripts/train.py`
- Batch prediction utility: `scripts/bulk_predict.py`
- Database schema: `scripts/schema.sql`
- Dashboards: `grafana_dashboard/*.json`
- Docker orchestration: `docker-compose.yml`

High-level flow:

1. Model files are loaded at API startup.
2. Client sends sensor payload to prediction endpoints.
3. API computes RUL and health status.
4. Prediction is persisted to PostgreSQL.
5. Grafana queries PostgreSQL and renders dashboards.

## Installation

Choose one of the following installation methods:

- Docker-based setup (recommended for easiest start)
- Manual local setup (Python + PostgreSQL + optional Grafana)

## Run with Docker (Recommended)

### Prerequisites

- Docker Desktop installed and running

### Steps

1. Clone the repository and enter project folder:

```bash
git clone https://github.com/springboardmentor0319/PrognosAI-AI-Driven-Predictive-Maintenance-System-Using-Time-Series-Sensor-Data.git
cd PrognosAI-AI-Driven-Predictive-Maintenance-System-Using-Time-Series-Sensor-Data
```

2. Create environment file:

```bash
cp .env.example .env
```

Windows PowerShell alternative:

```powershell
Copy-Item .env.example .env
```

3. Start all services:

```bash
docker compose up --build
```

4. Open services:

- API docs: http://localhost:8100/docs
- API health: http://localhost:8100/health
- UI: http://localhost:8100/
- Grafana: http://localhost:3100

Default Grafana login:

- Username: `admin`
- Password: `admin`

5. Verify containers:

```bash
docker compose ps
```

You should see these services:

- `postgres`
- `rul-api`
- `grafana`
- `grafana-init` (one-time init; exits after setup)

### Stop or reset

Stop services (keep DB volume):

```bash
docker compose down
```

Stop and delete volumes (full reset):

```bash
docker compose down -v
```

## Run Without Docker (Manual Setup)

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- (Optional) Grafana for dashboards

### Steps

1. Create and activate virtual environment:

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure PostgreSQL and schema:

```sql
CREATE DATABASE prognosai;
```

```bash
psql -U postgres -d prognosai -f scripts/schema.sql
```

4. Create `.env` and set DB values:

```bash
cp .env.example .env
```

For local non-docker run, make sure:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=prognosai
DB_USER=postgres
DB_PASSWORD=your_password
```

5. Start API from `scripts` directory:

```bash
cd scripts
uvicorn app:app --reload --port 8000
```

Open:

- API docs: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/health

## API Usage

Common endpoints:

- `POST /predict` single prediction
- `POST /predict/batch` batch prediction
- `POST /predict/sequence` sequence-based prediction
- `POST /ground-truth` submit true RUL for evaluation
- `GET /health` service health and loaded models

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subset": 1,
    "engine_id": 1,
    "cycle": 150,
    "setting_1": -0.0007,
    "setting_2": -0.0004,
    "setting_3": 100.0,
    "s2": 641.82,
    "s3": 1589.7,
    "s4": 1400.6,
    "s7": 554.36,
    "s8": 2388.1,
    "s9": 9046.2,
    "s11": 47.47,
    "s12": 521.66,
    "s13": 2388.1,
    "s14": 8138.6,
    "s17": 393,
    "s20": 39.06,
    "s21": 23.419
  }'
```

Docker users should use port `8100` instead of `8000` in API calls.

## Populate Dashboard Data

To push test predictions and model accuracy data into PostgreSQL:

```bash
cd scripts
python bulk_predict.py --subset all --with-ground-truth
```

Important:

- Local API default in script is `http://127.0.0.1:8000`
- If API is running in Docker, use:

```bash
python bulk_predict.py --subset all --with-ground-truth --base-url http://127.0.0.1:8100
```

Then refresh Grafana dashboards at http://localhost:3100.

## Project Structure

```text
.
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.grafana-init
├── .env.example
├── requirements.txt
├── scripts/
│   ├── app.py
│   ├── config.py
│   ├── train.py
│   ├── predict.py
│   ├── bulk_predict.py
│   ├── schema.sql
│   └── models/
├── docker/
│   ├── entrypoint.sh
│   ├── grafana_init.py
│   └── test_data/
├── grafana_dashboard/
└── ui/
```

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation):

| Subset | Train Engines | Test Engines | Operating Conditions | Fault Modes |
|--------|---------------|--------------|----------------------|-------------|
| FD001  | 100           | 100          | 1                    | 1           |
| FD002  | 260           | 259          | 6                    | 1           |
| FD003  | 100           | 100          | 1                    | 2           |
| FD004  | 248           | 249          | 6                    | 2           |

Evaluation metrics used:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 score
- NASA score (asymmetric, lower is better)

## Training and Retraining Models

Pre-trained models are already available in `scripts/models/`.

Retrain all subsets:

```bash
cd scripts
python train.py
```

Train one subset:

```bash
cd scripts
python train.py --subset 1
```

After retraining, restart the API so new model artifacts are loaded.

## Troubleshooting

1. API health shows DB unavailable

- Check `.env` DB values
- Confirm PostgreSQL is running
- Verify DB password and schema setup

2. Bulk prediction fails in Docker mode

- Use `--base-url http://127.0.0.1:8100`
- Confirm API health endpoint responds

3. Grafana dashboards are empty

- Run bulk prediction with `--with-ground-truth`
- Confirm data exists in PostgreSQL
- Wait for `grafana-init` to finish once after startup

4. Model not loaded for a subset

- Train the missing subset with `python train.py --subset <1|2|3|4>`
- Restart API

## License and Credits

- Dataset: NASA C-MAPSS turbofan engine degradation simulation data
- Frameworks and libraries: FastAPI, XGBoost, scikit-learn, PostgreSQL, Grafana, Docker
