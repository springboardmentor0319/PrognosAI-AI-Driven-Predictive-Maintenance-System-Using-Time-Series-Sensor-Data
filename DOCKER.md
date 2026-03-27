# Docker Setup Guide (Local Only)

> This file is local only — not pushed to GitHub.

## Requirements
- Docker Desktop (running)

## Start Everything
```bash
docker compose up
```

## Access
| Service | URL | Login |
|---------|-----|-------|
| FastAPI | http://localhost:8100/docs | — |
| Grafana | http://localhost:3100 | admin / admin |

## Populate Dashboards
```bash
cd scripts
python bulk_predict.py --subset 1 --with-ground-truth
python bulk_predict.py --subset 2 --with-ground-truth
python bulk_predict.py --subset 3 --with-ground-truth
python bulk_predict.py --subset 4 --with-ground-truth
```

## Stop / Reset
```bash
docker compose down          # stop, keep data
docker compose down -v       # stop, wipe database
```

## Push New Images to Docker Hub
```bash
docker build -t amalsalilan/rul-api:latest -f Dockerfile .
docker push amalsalilan/rul-api:latest

docker build -t amalsalilan/grafana-init:latest -f Dockerfile.grafana-init .
docker push amalsalilan/grafana-init:latest
```
