# PrognosAI — AI-Driven Predictive Maintenance System

Predicts the **Remaining Useful Life (RUL)** of turbofan engines using machine learning on the NASA C-MAPSS dataset. The system tells you how many operational cycles an engine has left before failure.

**Stack:** Python · XGBoost · FastAPI · PostgreSQL · Grafana

---

## Prerequisites

Make sure the following are installed on your machine:

- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **PostgreSQL 14+** — [postgresql.org](https://www.postgresql.org/download/)
- **Grafana** — [grafana.com/get](https://grafana.com/grafana/download)

---

## 1. Clone the Repository

```bash
git clone https://github.com/springboardmentor0319/PrognosAI-AI-Driven-Predictive-Maintenance-System-Using-Time-Series-Sensor-Data.git
cd PrognosAI-AI-Driven-Predictive-Maintenance-System-Using-Time-Series-Sensor-Data
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Set Up the Database

**Create the database in PostgreSQL:**

```sql
CREATE DATABASE prognosai;
```

**Apply the schema:**

```bash
psql -U postgres -d prognosai -f scripts/schema.sql
```

---

## 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Open `.env` and set your values:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=prognosai
DB_USER=postgres
DB_PASSWORD=your_postgres_password
```

---

## 6. Train the Models

The pre-trained models are included in `scripts/models/`. Skip this step unless you want to retrain.

```bash
cd scripts

# Train all 4 subsets
python train.py

# Or train a specific subset
python train.py --subset 1
```

Training output shows RMSE, MAE, R², and NASA Score for each subset.

---

## 7. Start the API

```bash
cd scripts
uvicorn app:app --reload --port 8000
```

API is now running at **http://localhost:8000**

- Interactive docs → **http://localhost:8000/docs**
- Health check → **http://localhost:8000/health**

---

## 8. Make a Prediction

Use the interactive docs at `/docs` or call the API directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subset": 1, "engine_id": 1, "cycle": 150,
    "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
    "s2": 641.82, "s3": 1589.7, "s4": 1400.6, "s7": 554.36,
    "s8": 2388.1, "s9": 9046.2, "s11": 47.47, "s12": 521.66,
    "s13": 2388.1, "s14": 8138.6, "s17": 393, "s20": 39.06, "s21": 23.419
  }'
```

**Response:**
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

## 9. Set Up Grafana

1. Open Grafana at **http://localhost:3000** (default login: `admin` / `admin`)
2. Add a PostgreSQL datasource:
   - Go to **Connections → Data Sources → Add new**
   - Select **PostgreSQL**
   - Fill in your DB credentials (host, database, user, password)
   - Click **Save & Test**
3. Import the dashboards:
   - Go to **Dashboards → Import**
   - Upload each file from `grafana_dashboard/`:
     - `dashboard_1_fleet_overview.json`
     - `dashboard_2_engine_detail.json`
     - `dashboard_3_model_performance.json`
   - Select the PostgreSQL datasource when prompted

---

## 10. Populate the Dashboards

Send test predictions to fill the Grafana charts:

```bash
cd scripts
python bulk_predict.py --subset 1 --with-ground-truth
python bulk_predict.py --subset 2 --with-ground-truth
python bulk_predict.py --subset 3 --with-ground-truth
python bulk_predict.py --subset 4 --with-ground-truth
```

Refresh Grafana — all 3 dashboards will now have live data.

---

## Project Structure

```
├── grafana_dashboard/       # Grafana dashboard JSON files
├── scripts/
│   ├── app.py               # FastAPI service
│   ├── train.py             # Model training
│   ├── predict.py           # CLI prediction tool
│   ├── bulk_predict.py      # Batch prediction for dashboard data
│   ├── config.py            # Shared config and feature engineering
│   ├── schema.sql           # PostgreSQL schema
│   ├── models/              # Trained model artifacts
│   └── data/                # NASA C-MAPSS dataset
├── requirements.txt
└── .env.example
```

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation):

| Subset | Train Engines | Test Engines | Operating Conditions | Fault Modes |
|--------|--------------|--------------|----------------------|-------------|
| FD001  | 100          | 100          | 1                    | 1           |
| FD002  | 260          | 259          | 6                    | 1           |
| FD003  | 100          | 100          | 1                    | 2           |
| FD004  | 248          | 249          | 6                    | 2           |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error (cycles) |
| **MAE** | Mean Absolute Error (cycles) |
| **R²** | Proportion of variance explained |
| **NASA Score** | Asymmetric penalty — late predictions penalised more than early (lower = better) |
