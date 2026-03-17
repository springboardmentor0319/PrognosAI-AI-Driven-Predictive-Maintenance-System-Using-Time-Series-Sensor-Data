import pandas as pd
import psycopg2

# Load prediction file
df = pd.read_csv("notebooks/predictions_fd001.csv")

# Convert RUL to health status
def get_status(rul):
    if rul > 80:
        return "Healthy"
    elif rul > 30:
        return "At Risk"
    else:
        return "Critical"

df["status"] = df["predicted_rul"].apply(get_status)

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="prognosai",
    user="postgres",
    password="user1234",
    port="5432"
)

cursor = conn.cursor()

# Insert rows into database
for _, row in df.iterrows():
    cursor.execute(
        """
        INSERT INTO engine_predictions (engine_id, predicted_rul, status)
        VALUES (%s, %s, %s)
        """,
        (int(row.engine_id), float(row.predicted_rul), row.status)
    )

conn.commit()

cursor.close()
conn.close()

print("Predictions inserted successfully!")