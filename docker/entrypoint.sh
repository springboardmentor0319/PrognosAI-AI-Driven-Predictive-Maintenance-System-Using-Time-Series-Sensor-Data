#!/bin/bash
set -e

echo "==> [entrypoint] Registering model metrics in model_registry..."
python /app/register_models.py

echo "==> [entrypoint] Starting uvicorn..."
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
