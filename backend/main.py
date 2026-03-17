from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
import psycopg2
import pandas as pd
import joblib

app = FastAPI()

# Load trained model
def load_model():
    return joblib.load("rul_model.pkl")
# PostgreSQL connection
conn = psycopg2.connect(
    host="localhost",
    database="prognosai",
    user="postgres",
    password="user1234"
)

cursor = conn.cursor()

def classify_rul(rul):
    if rul > 80:
        return "Healthy"
    elif rul > 30:
        return "At Risk"
    else:
        return "Critical"


@app.get("/predict")
def predict_and_store():

    model = load_model()
    # Load prediction CSV
    df = pd.read_csv("../notebooks/predictions_fd001.csv")

    # Create status column
    df["status"] = df["predicted_rul"].apply(classify_rul)

    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO engine_predictions (engine_id, predicted_rul, status)
            VALUES (%s,%s,%s)
            """,
            (int(row.engine_id), float(row.predicted_rul), row.status)
        )

    conn.commit()

    return {"message": "Predictions stored in PostgreSQL"}

def scheduled_prediction():
    print("Running scheduled prediction...")
    predict_and_store()

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_prediction, 'interval', minutes=5)

@app.on_event("startup")
def start_scheduler():
    scheduler.start()