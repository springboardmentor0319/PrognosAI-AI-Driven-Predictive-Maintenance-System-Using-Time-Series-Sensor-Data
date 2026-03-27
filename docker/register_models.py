"""
register_models.py — Upsert model_registry with real metrics from .pkl artifacts.

Run at container startup (before uvicorn) via entrypoint.sh.
Reads each rul_model_FD00{1-4}.pkl, extracts the metrics dict,
and UPDATEs model_registry so Dashboard 3 Panel 8 has real training data.
"""

import os
import sys
import pickle
import logging
import time
from pathlib import Path

import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("register_models")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "prognosai"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

MODEL_DIR = Path("/app/models")

# Add /app to sys.path so we can import config (load_model_artifact)
sys.path.insert(0, "/app")
from config import load_model_artifact, model_path, MODEL_DIR as CFG_MODEL_DIR  # noqa: E402


def wait_for_db(max_wait: int = 60) -> psycopg2.extensions.connection:
    """Poll until DB is ready AND model_registry table exists."""
    deadline = time.time() + max_wait
    while True:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM model_registry LIMIT 1")
            log.info("DB ready and schema applied.")
            return conn
        except psycopg2.OperationalError:
            pass
        except psycopg2.errors.UndefinedTable:
            log.warning("model_registry not yet created — waiting for schema init...")
        except Exception as exc:
            log.warning("DB not ready: %s", exc)

        if time.time() >= deadline:
            log.error("DB did not become ready within %ds. Aborting.", max_wait)
            raise TimeoutError(f"DB not ready after {max_wait}s")
        time.sleep(2)


def load_artifact(subset: int) -> dict | None:
    path = model_path(subset)
    if not path.exists():
        log.warning("Model not found: %s — skipping FD00%d", path, subset)
        return None
    try:
        # load_model_artifact handles both v1 and v2 formats
        artifact = load_model_artifact(subset)
        return artifact
    except Exception as exc:
        log.warning("Failed to load FD00%d: %s — skipping.", subset, exc)
        return None


def upsert_model_registry(conn, subset: int, artifact: dict) -> None:
    metrics = artifact.get("metrics", {})
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE model_registry
            SET
                model_version    = %s,
                algorithm        = %s,
                train_rmse       = %s,
                train_mae        = %s,
                train_r2         = %s,
                train_nasa_score = %s
            WHERE subset = %s::smallint
              AND is_active = TRUE
        """, (
            f"v1-FD00{subset}",
            artifact.get("algorithm", "XGBoost"),
            metrics.get("RMSE"),
            metrics.get("MAE"),
            metrics.get("R2"),
            metrics.get("NASA_Score"),
            subset,
        ))
        if cur.rowcount == 0:
            log.warning(
                "No active model_registry row for FD00%d — "
                "schema seed INSERT may not have run.", subset,
            )
        else:
            log.info(
                "Updated model_registry FD00%d: RMSE=%.2f MAE=%.2f R2=%.4f NASA=%.1f",
                subset,
                metrics.get("RMSE") or 0,
                metrics.get("MAE") or 0,
                metrics.get("R2") or 0,
                metrics.get("NASA_Score") or 0,
            )


def main():
    if not DB_CONFIG["password"]:
        log.warning("DB_PASSWORD not set — skipping model_registry update.")
        return

    conn = wait_for_db(max_wait=60)
    try:
        for subset in [1, 2, 3, 4]:
            artifact = load_artifact(subset)
            if artifact is None:
                continue
            upsert_model_registry(conn, subset, artifact)
        conn.commit()
        log.info("model_registry update complete.")
    except Exception:
        conn.rollback()
        log.exception("Failed to update model_registry.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
