#!/bin/bash
set -e

echo "==> [entrypoint] Registering model metrics in model_registry..."
python /app/register_models.py

echo "==> [entrypoint] Starting uvicorn..."
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 &
UVICORN_PID=$!

echo "==> [entrypoint] Running DB seed (skipped if data already exists)..."
python /app/seed_db.py

echo "==> [entrypoint] Seed done. Waiting for uvicorn (pid=$UVICORN_PID)..."
wait $UVICORN_PID
