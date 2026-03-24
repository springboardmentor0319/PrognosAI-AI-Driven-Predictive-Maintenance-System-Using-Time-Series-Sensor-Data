from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
import time

from model.load_model import load_all_models
from model.predict import predict_rul
from utils.schema import SensorInput, BatchInput

from database.db import engine, Base, get_db
from database.crud import save_prediction

app = FastAPI()

@app.on_event("startup")
def startup():
    print("Connecting to DB...")
    Base.metadata.create_all(bind=engine)
    print("DB connected and tables ready")


models = load_all_models()


@app.get("/")
def home():
    return {"message": "PrognosAI API Running"}


# ------------------ SINGLE PREDICTION ------------------
@app.post("/predict_single")
def predict_single(input_data: SensorInput, db: Session = Depends(get_db)):

    dataset = input_data.dataset.lower()
    features = input_data.features

    result = predict_rul(models, dataset, features)

    if "error" in result:
        return result

    save_prediction(
        db=db,
        engine_id=input_data.engine_id,
        predicted_rul=result["predicted_rul"],
        status=result["status"]
    )

    return result


# ------------------ BATCH PREDICTION ------------------
@app.post("/predict_batch")
def predict_batch(input_data: BatchInput, db: Session = Depends(get_db)):

    dataset = input_data.dataset.lower()
    batch = input_data.batch

    results = []

    for item in batch:
        try:
            result = predict_rul(models, dataset, item.features)

            if "error" not in result:
                save_prediction(
                    db=db,
                    engine_id=item.engine_id,
                    predicted_rul=result["predicted_rul"],
                    status=result["status"]
                )

        except Exception as e:
            result = {"error": str(e)}

        results.append(result)

    return {
        "dataset": dataset,
        "results": results
    }


# ------------------ LATENCY ------------------
@app.get("/latency")
def check_latency():
    start = time.time()

    _ = sum([i for i in range(1000)])

    end = time.time()

    return {
        "latency_ms": round((end - start) * 1000, 2)
    }


# ------------------ MODEL INFO ------------------
@app.get("/model_info")
def get_model_info():
    info = {}

    for key, value in models.items():
        info[key] = value.__class__.__name__

    return info



