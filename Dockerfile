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

# Copy only the inference-time data files (explicit — no globs for buildx compatibility)
COPY scripts/data/test_FD001.txt  ./test_data/test_FD001.txt
COPY scripts/data/test_FD002.txt  ./test_data/test_FD002.txt
COPY scripts/data/test_FD003.txt  ./test_data/test_FD003.txt
COPY scripts/data/test_FD004.txt  ./test_data/test_FD004.txt
COPY scripts/data/RUL_FD001.txt   ./test_data/RUL_FD001.txt
COPY scripts/data/RUL_FD002.txt   ./test_data/RUL_FD002.txt
COPY scripts/data/RUL_FD003.txt   ./test_data/RUL_FD003.txt
COPY scripts/data/RUL_FD004.txt   ./test_data/RUL_FD004.txt
COPY scripts/data/demo_FD001.txt  ./test_data/demo_FD001.txt
COPY scripts/data/demo_FD002.txt  ./test_data/demo_FD002.txt
COPY scripts/data/demo_FD003.txt  ./test_data/demo_FD003.txt
COPY scripts/data/demo_FD004.txt  ./test_data/demo_FD004.txt

RUN chmod +x ./entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
