from sqlalchemy.orm import Session
from database.models import EnginePrediction

def save_prediction(
    db: Session,
    engine_id: int,
    predicted_rul: float,
    status: str
):
    record = EnginePrediction(
        engine_id=engine_id,
        predicted_rul=predicted_rul,
        status=status
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    print("Saved ID:", record.id)

    return record