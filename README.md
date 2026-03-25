# PrognosAI — Predictive Maintenance System

AI-driven Remaining Useful Life (RUL) prediction for turbofan engines, built on the NASA C-MAPSS dataset. Predicts how many cycles an engine has left before failure.

**Stack:** XGBoost · FastAPI · PostgreSQL · Grafana · Docker

---

## What You Get

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | `http://localhost:8100/docs` | Interactive API (make predictions) |
| Grafana | `http://localhost:3100` | Live dashboards (fleet health, accuracy) |

3 pre-built Grafana dashboards:
- **Fleet Overview** — health status of all engines
- **Engine Deep Dive** — per-engine RUL trend + sensor data
- **Model Performance** — RMSE, MAE, NASA Score accuracy metrics

---

## Requirements

Just one thing:

- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** — install and make sure it's running

That's it. No Python, no PostgreSQL, no Grafana installation needed.

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/springboardmentor0319/PrognosAI-AI-Driven-Predictive-Maintenance-System-Using-Time-Series-Sensor-Data.git
cd PrognosAI-AI-Driven-Predictive-Maintenance-System-Using-Time-Series-Sensor-Data
```

### 2. Start everything
```bash
docker compose up
```

Wait ~30 seconds for all services to start. You'll see:
```
rul_api       | ==> Starting uvicorn...
rul_grafana   | HTTP Server Listen on :3000
rul_grafana_init | Grafana provisioning complete.
```

### 3. Open in your browser
- API → **http://localhost:8100/docs**
- Grafana → **http://localhost:3100** (login: `admin` / `admin`)

---

## Populate the Dashboards

The dashboards need prediction data to display charts. Run this once after startup:

> **Requires Python** with dependencies installed (only needed for this step)

```bash
cd scripts

# Install dependencies
pip install -r ../requirements.txt

# Send test predictions for all 4 engine datasets
python bulk_predict.py --subset 1 --with-ground-truth
python bulk_predict.py --subset 2 --with-ground-truth
python bulk_predict.py --subset 3 --with-ground-truth
python bulk_predict.py --subset 4 --with-ground-truth
```

Refresh Grafana — all 3 dashboards will now have live data.

---

## Make a Prediction (API)

Open **http://localhost:8100/docs** and try the `/predict` endpoint, or use curl:

```bash
curl -X POST http://localhost:8100/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subset": 1, "engine_id": 1, "cycle": 150,
    "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
    "s2": 641.82, "s3": 1589.7, "s4": 1400.6, "s7": 554.36,
    "s8": 2388.1, "s9": 9046.2, "s11": 47.47, "s12": 521.66,
    "s13": 2388.1, "s14": 8138.6, "s17": 393, "s20": 39.06, "s21": 23.419
  }'
```

Response:
```json
{
  "engine_id": 1,
  "subset": "FD001",
  "predicted_rul": 42.5,
  "health_status": "WARNING",
  "unit": "cycles"
}
```

---

## Stopping & Restarting

```bash
# Stop (keeps all data)
docker compose down

# Start again (data is preserved)
docker compose up

# Full reset (wipes database)
docker compose down -v
docker compose up
```

---

## Project Structure

```
├── docker/                  # Container startup scripts
├── grafana_dashboard/       # Pre-built Grafana dashboards
├── scripts/
│   ├── app.py               # FastAPI service
│   ├── train.py             # Model training (XGBoost)
│   ├── predict.py           # Batch/single prediction CLI
│   ├── bulk_predict.py      # Load test data into dashboards
│   ├── config.py            # Shared config & feature engineering
│   ├── schema.sql           # PostgreSQL schema
│   ├── models/              # Trained model artifacts (4 subsets)
│   └── data/                # NASA C-MAPSS dataset
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.grafana-init
└── requirements.txt
```

---

## Retrain the Models

```bash
cd scripts
python train.py              # trains all 4 subsets
python train.py --subset 1   # trains FD001 only
```

After retraining, rebuild and push the Docker image:
```bash
docker build -t amalsalilan/rul-api:latest -f Dockerfile .
docker push amalsalilan/rul-api:latest
```

---

## Troubleshooting

**Grafana dashboards are empty**
→ Run the `bulk_predict.py` commands from the [Populate the Dashboards](#populate-the-dashboards) section.

**Port already in use**
→ Change the ports in `docker-compose.yml` (left side of `host:container`).

**Models not loading (`No module named ...`)**
→ Your local model files were saved with a different Python/numpy version. Retrain: `python train.py`

**Database not connecting**
→ Run `docker compose down -v` then `docker compose up` to reset.

**`docker compose` command not found**
→ Make sure Docker Desktop is running and updated to a recent version.
