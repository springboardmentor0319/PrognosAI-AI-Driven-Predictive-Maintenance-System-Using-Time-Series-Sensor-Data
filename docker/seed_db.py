"""
seed_db.py — Auto-seed the database with demo data on first startup.

Runs after uvicorn is up. Checks if the predictions table is empty;
if so, seeds fleet snapshot (all 4 datasets) + one engine history per
dataset so every Grafana panel has data to show.

Skips silently if the DB already has predictions.
"""

import sys
import time
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8000"


def wait_for_api(max_wait: int = 60):
    print("[seed] Waiting for API to be ready...", flush=True)
    for i in range(max_wait):
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2)
            print(f"[seed] API ready after {i+1}s", flush=True)
            return True
        except Exception:
            time.sleep(1)
    print("[seed] API did not become ready in time — skipping seed.", flush=True)
    return False


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())


def get(path: str) -> dict:
    resp = urllib.request.urlopen(f"{BASE}{path}", timeout=10)
    return json.loads(resp.read())


def is_empty() -> bool:
    try:
        data = get("/stats/fleet")
        return data.get("total_predictions", 0) == 0
    except Exception:
        return True


def seed():
    if not wait_for_api():
        sys.exit(0)

    if not is_empty():
        print("[seed] DB already has predictions — skipping seed.", flush=True)
        sys.exit(0)

    print("[seed] DB is empty — seeding demo data...", flush=True)

    datasets = ["FD001", "FD002", "FD003", "FD004"]

    # ── Step 1: Fleet snapshot for all 4 datasets ────────────────────────────
    # Gives: 5 engines × 4 datasets = 20 predictions at NOW()
    # Activates: all Fleet Overview stat panels, pie chart, at-risk table,
    #            Prediction Activity (3h) and Fleet Avg RUL (24h) trend panels.
    print("[seed] Step 1/2 — Fleet snapshots...", flush=True)
    for ds in datasets:
        try:
            result = post("/predict/fleet-snapshot", {"dataset": ds})
            counts = {e["health_status"] for e in result["engines"]}
            print(f"[seed]   {ds}: {result['count']} engines seeded  statuses={counts}", flush=True)
        except Exception as exc:
            print(f"[seed]   {ds} fleet-snapshot failed: {exc}", flush=True)

    # ── Step 2: History for engine 1 of each dataset ─────────────────────────
    # Gives: ~200 cycles × 4 datasets = ~800 predictions with synthetic timestamps
    # Activates: Engine Deep Dive degradation curve, sensor readings panels.
    print("[seed] Step 2/2 — Engine histories (degradation curves)...", flush=True)
    for ds in datasets:
        try:
            result = post("/predict/engine-history", {"dataset": ds, "engine_ids": [1]})
            print(
                f"[seed]   {ds} engine 1: {result['total_predictions']} predictions "
                f"spanning {result['time_span_hours']}h",
                flush=True,
            )
        except Exception as exc:
            print(f"[seed]   {ds} engine-history failed: {exc}", flush=True)

    print("[seed] Seed complete. All Grafana panels should now have data.", flush=True)


if __name__ == "__main__":
    seed()
