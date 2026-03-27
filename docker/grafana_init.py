"""
grafana_init.py — One-shot Grafana API provisioner.

1. Poll GET /api/health until Grafana is ready (max 60s)
2. POST /api/datasources — create the PostgreSQL datasource
3. POST /api/dashboards/import — import each of 3 dashboard JSON files,
   substituting ${DS_POSTGRESQL} with the real datasource UID via the
   Grafana import API inputs mechanism (no string replacement needed).

Environment variables:
    GRAFANA_HOST        e.g. http://grafana:3000
    GF_ADMIN_USER       Grafana admin username
    GF_ADMIN_PASSWORD   Grafana admin password
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import json
import os
import sys
import time
import logging
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("grafana_init")

GRAFANA_HOST      = os.getenv("GRAFANA_HOST",        "http://grafana:3000")
GF_ADMIN_USER     = os.getenv("GF_ADMIN_USER",       "admin")
GF_ADMIN_PASSWORD = os.getenv("GF_ADMIN_PASSWORD",   "admin")

DB_HOST     = os.getenv("DB_HOST",     "postgres")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME",     "prognosai")
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DASHBOARD_DIR = Path("/app/grafana_dashboard")
AUTH          = (GF_ADMIN_USER, GF_ADMIN_PASSWORD)


def wait_for_grafana(max_wait: int = 60) -> None:
    """Poll GET /api/health until {"database": "ok"} is returned."""
    deadline = time.time() + max_wait
    while True:
        try:
            resp = requests.get(f"{GRAFANA_HOST}/api/health", timeout=5)
            if resp.json().get("database") == "ok":
                log.info("Grafana is ready.")
                return
        except Exception as exc:
            log.debug("Grafana not ready yet: %s", exc)

        if time.time() >= deadline:
            log.error("Grafana did not become ready within %ds.", max_wait)
            sys.exit(1)
        time.sleep(3)


def create_datasource() -> str:
    """
    Create the PostgreSQL datasource via Grafana API.
    Returns the UID assigned by Grafana.
    On HTTP 409 (already exists), fetches and returns the existing UID.
    """
    payload = {
        "name":     "PostgreSQL",
        "type":     "postgres",
        "access":   "proxy",
        "url":      f"{DB_HOST}:{DB_PORT}",
        "database": DB_NAME,
        "user":     DB_USER,
        "secureJsonData": {
            "password": DB_PASSWORD,
        },
        "jsonData": {
            "sslmode":         "disable",
            "postgresVersion": 1600,
            "timescaledb":     False,
        },
        "isDefault": True,
    }

    resp = requests.post(
        f"{GRAFANA_HOST}/api/datasources",
        json=payload, auth=AUTH, timeout=10,
    )

    if resp.status_code == 200:
        uid = resp.json()["datasource"]["uid"]
        log.info("Datasource created — UID: %s", uid)
        return uid

    if resp.status_code == 409:
        log.info("Datasource already exists — fetching UID.")
        get_resp = requests.get(
            f"{GRAFANA_HOST}/api/datasources/name/PostgreSQL",
            auth=AUTH, timeout=10,
        )
        get_resp.raise_for_status()
        uid = get_resp.json()["uid"]
        log.info("Existing datasource UID: %s", uid)
        return uid

    resp.raise_for_status()


def import_dashboard(path: Path, ds_uid: str) -> None:
    """
    Import a dashboard JSON via POST /api/dashboards/import.

    The dashboards have __inputs declaring DS_POSTGRESQL as a required
    datasource input. The import API substitutes ${DS_POSTGRESQL} with
    the real UID server-side when the inputs array is provided.
    """
    with open(path) as f:
        dashboard_json = json.load(f)

    title = dashboard_json.get("title", path.name)

    # Build the inputs array from the __inputs block in the dashboard JSON
    inputs = [
        {
            "name":     inp["name"],      # "DS_POSTGRESQL"
            "type":     inp["type"],      # "datasource"
            "pluginId": inp["pluginId"],  # "postgres"
            "value":    ds_uid,
        }
        for inp in dashboard_json.get("__inputs", [])
    ]

    resp = requests.post(
        f"{GRAFANA_HOST}/api/dashboards/import",
        json={
            "dashboard": dashboard_json,
            "overwrite":  True,
            "inputs":     inputs,
            "folderId":   0,
        },
        auth=AUTH, timeout=15,
    )

    if resp.status_code == 200:
        log.info("Imported: %s  (uid=%s)", title, resp.json().get("uid"))
    else:
        log.error("Failed to import %s: HTTP %d — %s", title, resp.status_code, resp.text[:200])
        resp.raise_for_status()


def main():
    wait_for_grafana(max_wait=60)

    ds_uid = create_datasource()

    dashboard_files = sorted(DASHBOARD_DIR.glob("*.json"))
    if not dashboard_files:
        log.error("No dashboard JSON files found in %s", DASHBOARD_DIR)
        sys.exit(1)

    for path in dashboard_files:
        log.info("Importing dashboard: %s", path.name)
        import_dashboard(path, ds_uid)

    log.info("Grafana provisioning complete.")


if __name__ == "__main__":
    main()
