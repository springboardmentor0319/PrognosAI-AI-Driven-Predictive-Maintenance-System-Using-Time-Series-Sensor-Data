"""
app.py — FastAPI RUL Prediction Service with PostgreSQL storage
===============================================================
Every prediction is automatically persisted to PostgreSQL so that
Grafana can query the data directly for monitoring dashboards.

Environment variables (set in .env or shell before running):
    DB_HOST      default: localhost
    DB_PORT      default: 5432
    DB_NAME      default: prognosai
    DB_USER      default: postgres
    DB_PASSWORD  default: postgres  (override this!)

Run:
    uvicorn app:app --reload --port 8000

Install deps:
    pip install fastapi uvicorn pydantic psycopg2-binary
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from uuid import uuid4
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import psycopg2
import psycopg2.pool

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from config import (
    COLUMN_NAMES, RUL_CAP, SENSORS,
    add_rolling_features, model_path, load_model_artifact,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rul_api")


# ---------------------------------------------------------------------------
# Database configuration — override via environment variables
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "prognosai"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),   # must be set via DB_PASSWORD env var
}

# Connection pool — shared across all requests (created at startup)
_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def get_conn():
    """Borrow a connection from the pool."""
    return _pool.getconn()


def put_conn(conn):
    """Return a connection to the pool."""
    _pool.putconn(conn)


# ---------------------------------------------------------------------------
# Model store (populated at startup)
# ---------------------------------------------------------------------------
MODELS: dict[int, dict] = {}


def load_all_models():
    for subset in [1, 2, 3, 4]:
        path = model_path(subset)
        if not path.exists():
            log.warning(f"No model for FD00{subset} — run: python train.py --subset {subset}")
            continue
        try:
            MODELS[subset] = load_model_artifact(subset)
            log.info(f"Loaded model  FD00{subset}  ← {path}")
        except Exception as exc:
            log.warning(f"Skipping FD00{subset}: {exc}")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    if not DB_CONFIG["password"]:
        log.error("DB_PASSWORD env var is not set — database writes are DISABLED.")
        log.error("Set DB_PASSWORD in your environment or .env file and restart.")
        _pool = None
    else:
        log.info("Connecting to PostgreSQL  %(host)s:%(port)s/%(dbname)s …" % DB_CONFIG)
        try:
            _pool = psycopg2.pool.ThreadedConnectionPool(minconn=2, maxconn=10, **DB_CONFIG)
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                ver = cur.fetchone()[0]
            put_conn(conn)
            log.info(f"PostgreSQL OK — {ver[:60]}")
        except Exception as exc:
            log.error(f"PostgreSQL connection failed: {exc}")
            log.error("API will run but DB writes are DISABLED. Fix DB_* env vars and restart.")
            _pool = None

    # ── ML models ───────────────────────────────────────────────────────────
    log.info("Loading ML model artifacts …")
    load_all_models()
    log.info(f"Startup complete — {len(MODELS)} model(s) loaded.")

    yield

    if _pool:
        _pool.closeall()
        log.info("DB pool closed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Engine RUL Prediction API",
    description=(
        "Predicts Remaining Useful Life (RUL) for turbofan engines using "
        "XGBoost models trained on the NASA C-MAPSS dataset. "
        "All predictions are stored in PostgreSQL for Grafana monitoring."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Test data directory (Docker: /app/test_data, local: scripts/data)
_here = Path(__file__).parent
TEST_DATA_DIR = _here / "test_data" if (_here / "test_data").exists() else _here / "data"
# Demo data directory — 5 engines per subset from training data (run-to-failure).
# In Docker the demo files land in test_data/ (same COPY layer as test files).
DEMO_DATA_DIR = TEST_DATA_DIR

# Serve the web UI as static files
# Docker: app.py → /app/app.py, ui → /app/ui  (parent / "ui")
# Local:  app.py → scripts/app.py, ui → project root / ui  (parent.parent / "ui")
UI_DIR = _here / "ui" if (_here / "ui").exists() else _here.parent / "ui"

if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(UI_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SensorSnapshot(BaseModel):
    """Latest-cycle sensor reading for one engine."""
    subset:    int   = Field(..., ge=1, le=4)
    engine_id: int   = Field(..., ge=1)
    cycle:     int   = Field(..., ge=1)
    setting_1: float
    setting_2: float
    setting_3: float
    s2:  float; s3:  float; s4:  float; s7:  float;  s8:  float
    s9:  float; s11: float; s12: float; s13: float;  s14: float
    s15: Optional[float] = None   # optional in schema; enforced per-subset in snapshot_to_df()
    s17: float; s20: float; s21: float

    @field_validator(
        "setting_1","setting_2","setting_3",
        "s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s17","s20","s21",
        mode="before",
    )
    @classmethod
    def must_be_finite(cls, v):
        import math
        if v is not None and (math.isnan(v) or math.isinf(v)):
            raise ValueError("sensor values must be finite (no NaN or Inf)")
        return v


class BatchRequest(BaseModel):
    instances: list[SensorSnapshot]


class SequenceRow(BaseModel):
    """One row in a multi-cycle engine history."""
    cycle:     int = Field(..., ge=1)
    setting_1: float; setting_2: float; setting_3: float
    s2:  float; s3:  float; s4:  float; s7:  float;  s8:  float
    s9:  float; s11: float; s12: float; s13: float;  s14: float
    s15: Optional[float] = None
    s17: float; s20: float; s21: float

    @field_validator(
        "setting_1","setting_2","setting_3",
        "s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s17","s20","s21",
        mode="before",
    )
    @classmethod
    def must_be_finite(cls, v):
        import math
        if v is not None and (math.isnan(v) or math.isinf(v)):
            raise ValueError("sensor values must be finite (no NaN or Inf)")
        return v


class SequenceRequest(BaseModel):
    subset:    int               = Field(..., ge=1, le=4)
    engine_id: int               = Field(..., ge=1)
    history:   list[SequenceRow] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    engine_id:     int
    subset:        str
    predicted_rul: float
    health_status: str         # HEALTHY / WARNING / CRITICAL
    unit:          str = "cycles"
    prediction_id: Optional[int] = None
    note:          str = ""


class GroundTruthInput(BaseModel):
    """Submit the true RUL for a past prediction to populate prediction_errors."""
    prediction_id: int   = Field(..., ge=1)
    true_rul:      float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _health_status(rul: float) -> str:
    if rul <= 10:  return "CRITICAL"
    if rul <= 30:  return "WARNING"
    return "HEALTHY"


def get_artifact(subset: int) -> dict:
    if subset not in MODELS:
        raise HTTPException(
            status_code=503,
            detail=f"Model FD00{subset} not loaded. Run: python train.py --subset {subset}",
        )
    return MODELS[subset]


def snapshot_to_df(snap: SensorSnapshot) -> pd.DataFrame:
    sensors = SENSORS[snap.subset]
    row = {
        "unit": snap.engine_id, "cycle": snap.cycle,
        "setting_1": snap.setting_1, "setting_2": snap.setting_2,
        "setting_3": snap.setting_3,
    }
    for s in sensors:
        val = getattr(snap, s, None)
        if val is None:
            raise HTTPException(
                status_code=422,
                detail=f"Sensor '{s}' is required for subset {snap.subset} but was not provided.",
            )
        row[s] = val
    return pd.DataFrame([row])


def run_inference(df: pd.DataFrame, artifact: dict) -> float:
    df  = add_rolling_features(df, artifact["sensors"], window=artifact["window"])
    row = df.sort_values("cycle").tail(1)
    X   = artifact["scaler"].transform(row[artifact["features"]].values)
    return round(float(np.clip(artifact["model"].predict(X)[0], 0, artifact["rul_cap"])), 1)


# ---------------------------------------------------------------------------
# DB write helpers
# ---------------------------------------------------------------------------

def _get_active_model_id(subset: int, conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT model_id FROM model_registry "
            "WHERE subset = %s::smallint AND is_active = TRUE "
            "ORDER BY trained_at DESC LIMIT 1",
            (subset,)
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(
            status_code=503,
            detail=f"No active model in model_registry for subset {subset}. "
                   "Apply schema.sql and ensure the INSERT rows ran successfully.",
        )
    return row[0]


def _upsert_engine(unit_number: int, subset: int, conn) -> int:
    """Return engine_id, creating the engine row if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT get_or_create_engine(%s::integer, %s::smallint)",
            (unit_number, subset)
        )
        return cur.fetchone()[0]


def _upsert_sensor_reading(engine_id: int, snap: SensorSnapshot, conn):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sensor_readings
                (engine_id, cycle,
                 setting_1, setting_2, setting_3,
                 s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21)
            VALUES
                (%s, %s,  %s, %s, %s,
                 %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (engine_id, cycle) DO UPDATE SET
                setting_1 = EXCLUDED.setting_1,
                setting_2 = EXCLUDED.setting_2,
                setting_3 = EXCLUDED.setting_3,
                s2  = EXCLUDED.s2,  s3  = EXCLUDED.s3,  s4  = EXCLUDED.s4,
                s7  = EXCLUDED.s7,  s8  = EXCLUDED.s8,  s9  = EXCLUDED.s9,
                s11 = EXCLUDED.s11, s12 = EXCLUDED.s12, s13 = EXCLUDED.s13,
                s14 = EXCLUDED.s14, s15 = EXCLUDED.s15, s17 = EXCLUDED.s17,
                s20 = EXCLUDED.s20, s21 = EXCLUDED.s21,
                recorded_at = NOW()
        """, (
            engine_id, snap.cycle,
            snap.setting_1, snap.setting_2, snap.setting_3,
            snap.s2, snap.s3, snap.s4, snap.s7, snap.s8, snap.s9,
            snap.s11, snap.s12, snap.s13, snap.s14, snap.s15,
            snap.s17, snap.s20, snap.s21,
        ))


def _insert_prediction(
    engine_id: int, model_id: int, cycle: int, n_history: int,
    predicted_rul: float, endpoint: str,
    request_id: str, latency_ms: float, conn,
    predicted_at: Optional[datetime] = None,
) -> int:
    with conn.cursor() as cur:
        if predicted_at is not None:
            cur.execute("""
                INSERT INTO predictions
                    (request_id, engine_id, model_id, cycle, n_history_cycles,
                     predicted_rul, rul_cap, endpoint, latency_ms, predicted_at)
                VALUES (%s, %s, %s, %s, %s,  %s, %s, %s, %s, %s)
                RETURNING prediction_id
            """, (
                request_id, engine_id, model_id, cycle, n_history,
                predicted_rul, RUL_CAP, endpoint, latency_ms, predicted_at,
            ))
        else:
            cur.execute("""
                INSERT INTO predictions
                    (request_id, engine_id, model_id, cycle, n_history_cycles,
                     predicted_rul, rul_cap, endpoint, latency_ms)
                VALUES (%s, %s, %s, %s, %s,  %s, %s, %s, %s)
                RETURNING prediction_id
            """, (
                request_id, engine_id, model_id, cycle, n_history,
                predicted_rul, RUL_CAP, endpoint, latency_ms,
            ))
        return cur.fetchone()[0]


def db_save(
    snap: SensorSnapshot,
    predicted_rul: float,
    n_history: int,
    endpoint: str,
    request_id: str,
    latency_ms: float,
) -> Optional[int]:
    """
    Full DB write for one prediction:
      engine upsert → sensor reading upsert → prediction insert.
    Returns prediction_id, or None if the DB is unavailable.
    """
    if _pool is None:
        log.warning("DB unavailable — prediction not persisted.")
        return None

    conn = get_conn()
    try:
        engine_id = _upsert_engine(snap.engine_id, snap.subset, conn)
        model_id  = _get_active_model_id(snap.subset, conn)
        _upsert_sensor_reading(engine_id, snap, conn)
        pred_id   = _insert_prediction(
            engine_id, model_id, snap.cycle, n_history,
            predicted_rul, endpoint, request_id, latency_ms, conn,
        )
        conn.commit()
        log.info(
            f"DB  prediction_id={pred_id}  engine={snap.engine_id}  "
            f"FD00{snap.subset}  cycle={snap.cycle}  RUL={predicted_rul}  "
            f"status={_health_status(predicted_rul)}"
        )
        return pred_id
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        log.error(f"DB write failed (engine={snap.engine_id}, FD00{snap.subset}): {exc}")
        return None
    finally:
        put_conn(conn)


# ---------------------------------------------------------------------------
# Routes — Prediction
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_single(snap: SensorSnapshot):
    """
    Predict RUL from the latest sensor snapshot of one engine.
    Saves sensor_readings + predictions rows to PostgreSQL.
    """
    t0       = time.perf_counter()
    artifact = get_artifact(snap.subset)
    df       = snapshot_to_df(snap)
    rul      = run_inference(df, artifact)
    latency  = round((time.perf_counter() - t0) * 1000, 2)

    pred_id = db_save(
        snap=snap, predicted_rul=rul, n_history=1,
        endpoint="/predict", request_id=str(uuid4()), latency_ms=latency,
    )

    note = (
        "Single snapshot — rolling features approximate. "
        "Use /predict/sequence for full-history accuracy."
    ) if snap.cycle < artifact["window"] else ""

    return PredictResponse(
        engine_id=snap.engine_id, subset=f"FD00{snap.subset}",
        predicted_rul=rul, health_status=_health_status(rul),
        prediction_id=pred_id, note=note,
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(req: BatchRequest):
    """
    Predict RUL for multiple engines in one request.
    All rows share the same request_id in the predictions table.
    """
    request_id = str(uuid4())
    results    = []

    for snap in req.instances:
        t0       = time.perf_counter()
        artifact = get_artifact(snap.subset)
        df       = snapshot_to_df(snap)
        rul      = run_inference(df, artifact)
        latency  = round((time.perf_counter() - t0) * 1000, 2)

        pred_id = db_save(
            snap=snap, predicted_rul=rul, n_history=1,
            endpoint="/predict/batch", request_id=request_id, latency_ms=latency,
        )
        results.append({
            "engine_id":     snap.engine_id,
            "subset":        f"FD00{snap.subset}",
            "predicted_rul": rul,
            "health_status": _health_status(rul),
            "prediction_id": pred_id,
            "unit":          "cycles",
        })

    return {"predictions": results, "count": len(results), "request_id": request_id}


# ---------------------------------------------------------------------------
# Test-data prediction helpers
# ---------------------------------------------------------------------------

def _load_test_last(dataset: str) -> pd.DataFrame:
    """Load test_FD00x.txt and return the last cycle per engine."""
    path = TEST_DATA_DIR / f"test_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Test data not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return (
        df.sort_values(["unit", "cycle"])
          .groupby("unit").last()
          .reset_index()
    )


class TestBatchRequest(BaseModel):
    dataset:     str
    engine_ids:  Optional[list[int]] = None  # None → use max_engines from the top
    max_engines: int = Field(default=10, ge=1, le=100)


@app.get("/test-data/{dataset}/engines", tags=["Test Data"])
def test_data_engines(dataset: str):
    """Return the list of engine IDs available in the test file for a dataset."""
    if dataset not in ("FD001", "FD002", "FD003", "FD004"):
        raise HTTPException(status_code=400, detail="dataset must be FD001–FD004")
    try:
        df = _load_test_last(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    engines = [
        {"unit_number": int(r["unit"]), "last_cycle": int(r["cycle"])}
        for _, r in df.iterrows()
    ]
    return {"dataset": dataset, "subset": int(dataset[-1]), "engines": engines}


@app.post("/predict/from-test", tags=["Prediction"])
def predict_from_test(req: TestBatchRequest):
    """
    Run batch predictions using the bundled test dataset — no manual sensor entry.
    Specify a dataset and optionally a list of engine IDs; defaults to the first
    max_engines engines.
    """
    if req.dataset not in ("FD001", "FD002", "FD003", "FD004"):
        raise HTTPException(status_code=400, detail="dataset must be FD001–FD004")
    try:
        df = _load_test_last(req.dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if req.engine_ids:
        df = df[df["unit"].isin(req.engine_ids)]
    else:
        df = df.head(req.max_engines)

    if df.empty:
        raise HTTPException(status_code=404, detail="No matching engines found.")

    subset   = int(req.dataset[-1])
    sensors  = SENSORS[subset]
    instances = []
    for _, row in df.iterrows():
        payload: dict = {
            "subset":    subset,
            "engine_id": int(row["unit"]),
            "cycle":     int(row["cycle"]),
            "setting_1": float(row["setting_1"]),
            "setting_2": float(row["setting_2"]),
            "setting_3": float(row["setting_3"]),
        }
        for s in sensors:
            payload[s] = float(row[s])
        instances.append(SensorSnapshot(**payload))

    return predict_batch(BatchRequest(instances=instances))


# ---------------------------------------------------------------------------
# Demo history mode — run predictions for ALL cycles of selected engines
# ---------------------------------------------------------------------------

def _load_demo_all(dataset: str) -> pd.DataFrame:
    """Load demo_FD00x.txt with all cycles. Falls back to full training file locally."""
    path = DEMO_DATA_DIR / f"demo_{dataset}.txt"
    if not path.exists():
        path = DEMO_DATA_DIR / f"train_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Demo data not found for {dataset} in {DEMO_DATA_DIR}")
    from config import COLUMN_NAMES as COL_NAMES
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


class HistoryBatchRequest(BaseModel):
    dataset:    str
    engine_ids: Optional[list[int]] = None  # None → all available engines in demo file


@app.get("/demo-data/{dataset}/engines", tags=["Demo History"])
def demo_data_engines(dataset: str):
    """Return engines available in the demo file (5 per subset, run-to-failure)."""
    if dataset not in ("FD001", "FD002", "FD003", "FD004"):
        raise HTTPException(status_code=400, detail="dataset must be FD001–FD004")
    try:
        df = _load_demo_all(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    summary = (
        df.groupby("unit")["cycle"]
          .agg(max_cycle="max", total_cycles="count")
          .reset_index()
    )
    engines = [
        {"unit_number": int(r["unit"]), "max_cycle": int(r["max_cycle"]), "total_cycles": int(r["total_cycles"])}
        for _, r in summary.iterrows()
    ]
    return {"dataset": dataset, "subset": int(dataset[-1]), "engines": engines}


@app.post("/predict/engine-history", tags=["Demo History"])
def predict_engine_history(req: HistoryBatchRequest):
    """
    Run predictions for EVERY cycle of selected demo engines.

    Uses run-to-failure training data (5 engines per subset).
    Timestamps are synthesised so cycle 1 is oldest and the final cycle is NOW,
    giving Grafana a real degradation time-series (HEALTHY → WARNING → CRITICAL).

    Returns a summary; all predictions are saved to the predictions table.
    """
    if req.dataset not in ("FD001", "FD002", "FD003", "FD004"):
        raise HTTPException(status_code=400, detail="dataset must be FD001–FD004")
    try:
        df = _load_demo_all(req.dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if req.engine_ids:
        df = df[df["unit"].isin(req.engine_ids)]
    if df.empty:
        raise HTTPException(status_code=404, detail="No matching engines found.")

    subset   = int(req.dataset[-1])
    artifact = get_artifact(subset)
    sensors  = artifact["sensors"]
    features = artifact["features"]

    # Compute rolling features for all selected engines in one pass
    df_feat = add_rolling_features(df, sensors, window=artifact["window"])

    # Anchor: the latest cycle of ANY engine in the selection maps to NOW
    global_max_cycle = int(df_feat["cycle"].max())
    now = datetime.now(timezone.utc)

    request_id = str(uuid4())
    total_saved = 0
    engine_summaries = []

    if _pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    conn = get_conn()
    try:
        model_id = _get_active_model_id(subset, conn)

        for unit, group in df_feat.groupby("unit"):
            unit = int(unit)
            engine_id = _upsert_engine(unit, subset, conn)

            preds_for_engine = []
            for _, row in group.iterrows():
                cycle = int(row["cycle"])
                # Synthetic timestamp: each cycle step = 1 hour back from now
                hours_ago     = global_max_cycle - cycle
                predicted_at  = now - timedelta(hours=hours_ago)

                # Direct inference on pre-computed feature row
                X   = artifact["scaler"].transform(row[features].values.reshape(1, -1))
                rul = round(float(np.clip(artifact["model"].predict(X)[0], 0, artifact["rul_cap"])), 1)

                # Upsert sensor reading (use available sensors for this subset)
                s_vals = {s: float(row[s]) for s in sensors}
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO sensor_readings
                            (engine_id, cycle,
                             setting_1, setting_2, setting_3,
                             s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21)
                        VALUES
                            (%s, %s,  %s, %s, %s,
                             %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (engine_id, cycle) DO UPDATE SET
                            setting_1 = EXCLUDED.setting_1,
                            setting_2 = EXCLUDED.setting_2,
                            setting_3 = EXCLUDED.setting_3,
                            s2=EXCLUDED.s2, s3=EXCLUDED.s3, s4=EXCLUDED.s4,
                            s7=EXCLUDED.s7, s8=EXCLUDED.s8, s9=EXCLUDED.s9,
                            s11=EXCLUDED.s11, s12=EXCLUDED.s12, s13=EXCLUDED.s13,
                            s14=EXCLUDED.s14, s15=EXCLUDED.s15, s17=EXCLUDED.s17,
                            s20=EXCLUDED.s20, s21=EXCLUDED.s21,
                            recorded_at = %s
                    """, (
                        engine_id, cycle,
                        float(row["setting_1"]), float(row["setting_2"]), float(row["setting_3"]),
                        s_vals.get("s2"), s_vals.get("s3"), s_vals.get("s4"),
                        s_vals.get("s7"), s_vals.get("s8"), s_vals.get("s9"),
                        s_vals.get("s11"), s_vals.get("s12"), s_vals.get("s13"),
                        s_vals.get("s14"), s_vals.get("s15"), s_vals.get("s17"),
                        s_vals.get("s20"), s_vals.get("s21"),
                        predicted_at,
                    ))

                _insert_prediction(
                    engine_id=engine_id, model_id=model_id,
                    cycle=cycle, n_history=cycle,
                    predicted_rul=rul, endpoint="/predict/engine-history",
                    request_id=request_id, latency_ms=0.0,
                    conn=conn, predicted_at=predicted_at,
                )
                preds_for_engine.append(rul)
                total_saved += 1

            conn.commit()
            final_rul  = preds_for_engine[-1] if preds_for_engine else 0
            engine_summaries.append({
                "unit_number":  unit,
                "total_cycles": len(preds_for_engine),
                "final_rul":    final_rul,
                "health_status": _health_status(final_rul),
            })
            log.info(
                f"History  unit={unit}  FD00{subset}  "
                f"cycles={len(preds_for_engine)}  final_RUL={final_rul}"
            )

    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        log.error(f"History prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        put_conn(conn)

    span_hours = global_max_cycle  # oldest cycle is ~global_max_cycle hours ago
    return {
        "dataset":          req.dataset,
        "total_predictions": total_saved,
        "engines_processed": len(engine_summaries),
        "time_span_hours":  span_hours,
        "request_id":       request_id,
        "engines":          engine_summaries,
    }


@app.post("/predict/fleet-snapshot", tags=["Demo History"])
def predict_fleet_snapshot(req: HistoryBatchRequest):
    """
    Simulate a realistic fleet by snapshotting each demo engine at a different
    life stage so the fleet shows a mix of HEALTHY / WARNING / CRITICAL engines.

    Life-stage percentages applied to each engine in order:
        [25%, 45%, 65%, 80%, 100%]
    — or evenly distributed when a different number of engines is selected.

    All predictions are saved with timestamp = NOW() so the Fleet Overview
    health panels update immediately.
    """
    if req.dataset not in ("FD001", "FD002", "FD003", "FD004"):
        raise HTTPException(status_code=400, detail="dataset must be FD001–FD004")
    try:
        df_all = _load_demo_all(req.dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Determine which engines to use and their snapshot percentages
    units = sorted(df_all["unit"].unique().tolist())
    if req.engine_ids:
        units = [u for u in units if u in req.engine_ids]
    if not units:
        raise HTTPException(status_code=404, detail="No matching engines found.")

    # Percentages tuned to produce HEALTHY / WARNING / CRITICAL spread
    # based on the model's actual prediction curve (calibrated to 10/30 thresholds)
    n = len(units)
    _base_pcts = [0.20, 0.30, 0.42, 0.52, 0.80]
    if n >= 5:
        pcts = _base_pcts[:n]
    elif n == 1:
        pcts = [0.40]
    else:
        step = 0.60 / max(n - 1, 1)
        pcts = [0.20 + step * i for i in range(n)]

    subset   = int(req.dataset[-1])
    artifact = get_artifact(subset)
    sensors  = artifact["sensors"]
    features = artifact["features"]

    if _pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    request_id = str(uuid4())
    results    = []

    conn = get_conn()
    try:
        model_id = _get_active_model_id(subset, conn)

        for unit, pct in zip(units, pcts):
            unit_df = df_all[df_all["unit"] == unit].copy()
            max_cycle = int(unit_df["cycle"].max())
            snap_cycle = max(1, round(pct * max_cycle))

            # Slice up to snapshot cycle (keeps rolling features accurate)
            sliced = unit_df[unit_df["cycle"] <= snap_cycle].copy()
            sliced = add_rolling_features(sliced, sensors, window=artifact["window"])

            last_row = sliced.sort_values("cycle").iloc[-1]
            X   = artifact["scaler"].transform(last_row[features].values.reshape(1, -1))
            rul = round(float(np.clip(artifact["model"].predict(X)[0], 0, artifact["rul_cap"])), 1)

            t0        = time.perf_counter()
            engine_id = _upsert_engine(int(unit), subset, conn)
            latency   = round((time.perf_counter() - t0) * 1000, 2)

            # Upsert sensor reading at snapshot cycle
            s_vals = {s: float(last_row.get(s, 0) or 0) for s in sensors}
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sensor_readings
                        (engine_id, cycle,
                         setting_1, setting_2, setting_3,
                         s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21)
                    VALUES (%s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (engine_id, cycle) DO UPDATE SET
                        setting_1=EXCLUDED.setting_1, setting_2=EXCLUDED.setting_2,
                        setting_3=EXCLUDED.setting_3,
                        s2=EXCLUDED.s2, s3=EXCLUDED.s3, s4=EXCLUDED.s4,
                        s7=EXCLUDED.s7, s8=EXCLUDED.s8, s9=EXCLUDED.s9,
                        s11=EXCLUDED.s11, s12=EXCLUDED.s12, s13=EXCLUDED.s13,
                        s14=EXCLUDED.s14, s15=EXCLUDED.s15, s17=EXCLUDED.s17,
                        s20=EXCLUDED.s20, s21=EXCLUDED.s21, recorded_at=NOW()
                """, (
                    engine_id, int(last_row["cycle"]),
                    float(last_row["setting_1"]), float(last_row["setting_2"]),
                    float(last_row["setting_3"]),
                    s_vals.get("s2"), s_vals.get("s3"), s_vals.get("s4"),
                    s_vals.get("s7"), s_vals.get("s8"), s_vals.get("s9"),
                    s_vals.get("s11"), s_vals.get("s12"), s_vals.get("s13"),
                    s_vals.get("s14"), s_vals.get("s15"), s_vals.get("s17"),
                    s_vals.get("s20"), s_vals.get("s21"),
                ))

            pred_id = _insert_prediction(
                engine_id=engine_id, model_id=model_id,
                cycle=int(last_row["cycle"]), n_history=snap_cycle,
                predicted_rul=rul, endpoint="/predict/fleet-snapshot",
                request_id=request_id, latency_ms=latency, conn=conn,
            )
            conn.commit()

            results.append({
                "unit_number":   int(unit),
                "snapshot_pct":  round(pct * 100),
                "snapshot_cycle": snap_cycle,
                "max_cycle":     max_cycle,
                "predicted_rul": rul,
                "health_status": _health_status(rul),
                "prediction_id": pred_id,
            })
            log.info(
                f"Fleet-snap  unit={unit}  FD00{subset}  "
                f"pct={round(pct*100)}%  cycle={snap_cycle}/{max_cycle}  RUL={rul}"
            )

    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        log.error(f"Fleet snapshot failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        put_conn(conn)

    return {
        "dataset":  req.dataset,
        "engines":  results,
        "count":    len(results),
        "request_id": request_id,
    }


@app.post("/predict/sequence", response_model=PredictResponse, tags=["Prediction"])
def predict_sequence(req: SequenceRequest):
    """
    Predict RUL from the full engine sensor history.
    Rolling features use the real historical window — most accurate endpoint.
    """
    artifact = get_artifact(req.subset)
    sensors  = artifact["sensors"]

    # Validate cycle ordering
    cycles = [r.cycle for r in req.history]
    if len(set(cycles)) != len(cycles):
        raise HTTPException(status_code=422, detail="history contains duplicate cycle numbers.")
    if cycles != sorted(cycles):
        raise HTTPException(status_code=422, detail="history cycles must be in ascending order.")
    if any(cycles[i+1] - cycles[i] != 1 for i in range(len(cycles) - 1)):
        raise HTTPException(
            status_code=422,
            detail="history cycles must be consecutive with no gaps (e.g. 1,2,3,…).",
        )

    rows = []
    for r in req.history:
        row = {
            "unit": req.engine_id, "cycle": r.cycle,
            "setting_1": r.setting_1, "setting_2": r.setting_2, "setting_3": r.setting_3,
        }
        for s in sensors:
            val = getattr(r, s, None)
            if val is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Sensor '{s}' required for FD00{req.subset} but missing in history.",
                )
            row[s] = val
        rows.append(row)

    t0      = time.perf_counter()
    df      = pd.DataFrame(rows)
    rul     = run_inference(df, artifact)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    # Build a SensorSnapshot from the last history row to reuse db_save
    last = rows[-1]
    snap = SensorSnapshot(
        subset=req.subset, engine_id=req.engine_id, cycle=last["cycle"],
        setting_1=last["setting_1"], setting_2=last["setting_2"], setting_3=last["setting_3"],
        s2=last["s2"], s3=last["s3"], s4=last["s4"],
        s7=last["s7"], s8=last["s8"], s9=last["s9"],
        s11=last["s11"], s12=last["s12"], s13=last["s13"],
        s14=last["s14"], s15=last.get("s15"), s17=last["s17"],
        s20=last["s20"], s21=last["s21"],
    )

    pred_id = db_save(
        snap=snap, predicted_rul=rul, n_history=len(req.history),
        endpoint="/predict/sequence", request_id=str(uuid4()), latency_ms=latency,
    )

    return PredictResponse(
        engine_id=req.engine_id, subset=f"FD00{req.subset}",
        predicted_rul=rul, health_status=_health_status(rul),
        prediction_id=pred_id,
        note=f"Computed from {len(req.history)} historical cycles.",
    )


# ---------------------------------------------------------------------------
# Routes — Evaluation
# ---------------------------------------------------------------------------

@app.post("/ground-truth", tags=["Evaluation"])
def submit_ground_truth(body: GroundTruthInput):
    """
    Submit the true RUL for a past prediction to populate prediction_errors.
    This drives the RMSE / MAE / NASA Score accuracy panels in Grafana.

    When to call this:
      - During test-set replay (you know the ground truth from RUL_FD00x.txt)
      - After an engine is retired (you now know when it actually failed)
    """
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT predicted_rul FROM predictions WHERE prediction_id = %s",
                (body.prediction_id,)
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"prediction_id={body.prediction_id} not found."
            )

        predicted_rul = float(row[0])
        error         = predicted_rul - body.true_rul
        abs_error     = abs(error)
        nasa_penalty  = (
            float(np.exp(-error / 13) - 1) if error < 0
            else float(np.exp(error / 10) - 1)
        )

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO prediction_errors
                    (prediction_id, true_rul, error, abs_error, nasa_penalty)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (body.prediction_id, body.true_rul, error, abs_error, nasa_penalty))

        conn.commit()

        return {
            "prediction_id": body.prediction_id,
            "predicted_rul": predicted_rul,
            "true_rul":      body.true_rul,
            "error":         round(error, 2),
            "abs_error":     round(abs_error, 2),
            "nasa_penalty":  round(nasa_penalty, 4),
            "status":        "LATE ⚠" if error > 10 else ("EARLY" if error < -10 else "OK ✓"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        log.error(f"ground-truth write failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        put_conn(conn)


# ---------------------------------------------------------------------------
# Routes — Admin
# ---------------------------------------------------------------------------

@app.post("/admin/seed", tags=["Admin"])
def admin_seed(force: bool = False):
    """
    Seed the database with demo data so every Grafana panel has something to show.
    Runs automatically on startup when the DB is empty.

    Set force=true to re-seed even if predictions already exist.
    """
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    # Check current state
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM predictions")
            total = cur.fetchone()[0]
    finally:
        put_conn(conn)

    if total > 0 and not force:
        return {"status": "skipped", "reason": f"DB already has {total} predictions. Use force=true to reseed."}

    datasets = ["FD001", "FD002", "FD003", "FD004"]
    summary  = {"fleet_snapshots": [], "histories": []}

    # Step 1 — fleet snapshot for all datasets (activates fleet panels + recent activity)
    for ds in datasets:
        try:
            result = predict_fleet_snapshot(HistoryBatchRequest(dataset=ds))
            summary["fleet_snapshots"].append({
                "dataset": ds,
                "engines": result["count"],
                "statuses": list({e["health_status"] for e in result["engines"]}),
            })
        except Exception as exc:
            log.error(f"Seed fleet-snapshot {ds}: {exc}")

    # Step 2 — history for engine 1 of each dataset (activates degradation curves)
    for ds in datasets:
        try:
            result = predict_engine_history(HistoryBatchRequest(dataset=ds, engine_ids=[1]))
            summary["histories"].append({
                "dataset":           ds,
                "total_predictions": result["total_predictions"],
                "time_span_hours":   result["time_span_hours"],
            })
        except Exception as exc:
            log.error(f"Seed engine-history {ds}: {exc}")

    log.info(f"Seed complete — forced={force}")
    return {"status": "seeded", "force": force, "summary": summary}


@app.post("/admin/flush", tags=["Admin"])
def admin_flush():
    """Wipe all prediction data and reset sequences for a clean slate."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE prediction_errors, predictions, sensor_readings, engines "
                "RESTART IDENTITY CASCADE"
            )
        conn.commit()
        log.info("DB flushed — all prediction data wiped.")
        return {"status": "flushed"}
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        put_conn(conn)


# ---------------------------------------------------------------------------
# Routes — Monitoring / Meta
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    """Service health check — confirms DB connectivity and loaded models."""
    db_ok = False
    if _pool:
        try:
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            put_conn(conn)
            db_ok = True
        except Exception:
            db_ok = False

    return {
        "status":        "ok",
        "database":      "connected" if db_ok else "unavailable",
        "db_host":       DB_CONFIG["host"],
        "db_name":       DB_CONFIG["dbname"],
        "models_loaded": [f"FD00{s}" for s in sorted(MODELS.keys())],
    }


@app.get("/model/info/{subset}", tags=["Meta"])
def model_info(subset: int):
    """Return metadata about the loaded model for a given subset."""
    artifact = get_artifact(subset)
    return {
        "subset":     f"FD00{subset}",
        "sensors":    artifact["sensors"],
        "rul_cap":    artifact["rul_cap"],
        "window":     artifact["window"],
        "n_features": len(artifact["features"]),
        "metrics":    artifact.get("metrics", {}),
    }


@app.get("/stats/fleet", tags=["Monitoring"])
def fleet_stats():
    """
    Live fleet health summary — useful for a quick sanity check.
    The same data is available in the v_fleet_health_summary Grafana view.
    """
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM v_fleet_health_summary")
            cols    = [d[0] for d in cur.description]
            summary = [dict(zip(cols, r)) for r in cur.fetchall()]

            cur.execute("SELECT COUNT(*) FROM predictions")
            total = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM predictions "
                "WHERE predicted_at > NOW() - INTERVAL '1 hour'"
            )
            last_hour = cur.fetchone()[0]

        return {
            "total_predictions":     total,
            "predictions_last_hour": last_hour,
            "health_breakdown":      summary,
        }
    finally:
        put_conn(conn)