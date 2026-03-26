FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/app.py           ./app.py
COPY scripts/config.py        ./config.py
COPY docker/register_models.py ./register_models.py
COPY docker/seed_db.py        ./seed_db.py
COPY docker/entrypoint.sh     ./entrypoint.sh

# Copy trained model artifacts (~7 MB for all 4 subsets)
COPY scripts/models/          ./models/
COPY ui/                      ./ui/

# Copy only the inference-time data files (no training data)
COPY scripts/data/test_FD*.txt  ./test_data/
COPY scripts/data/RUL_FD*.txt   ./test_data/
COPY scripts/data/demo_FD*.txt  ./test_data/

RUN chmod +x ./entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
