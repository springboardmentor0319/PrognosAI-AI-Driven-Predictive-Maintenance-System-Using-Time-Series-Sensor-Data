from fastapi import FastAPI
from model.load_model import load_all_models
from model.predict import predict_rul
from utils.schema import SensorInput, BatchInput

app = FastAPI()

# Load all models once
models = load_all_models()

@app.get("/")
def home():
    return {"message": "PrognosAI API Running"}

# Single prediction
@app.post("/predict_single")
def predict_single(input_data: SensorInput):

    dataset = input_data.dataset.lower()
    features = input_data.features

    result = predict_rul(models, dataset, features)

    return result

@app.post("/predict_batch")
def predict_batch(input_data: BatchInput):
    dataset = input_data.dataset.lower()
    batch = input_data.batch

    results = []


    for features in batch:
        try:
            result = predict_rul(models, dataset, features)
        except Exception as e:
            result = {"error": str(e)}

    results.append(result)

    return {
        "dataset": dataset,
        "results": results
    }