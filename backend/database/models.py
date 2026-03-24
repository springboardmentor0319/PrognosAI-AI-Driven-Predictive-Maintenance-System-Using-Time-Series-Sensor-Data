from sqlalchemy import Column, DateTime, Integer, Float, String, TIMESTAMP
from sqlalchemy.sql import func
from database.db import Base

class EnginePrediction(Base):
    __tablename__ = "engine_predictions"

    id = Column(Integer, primary_key=True, index=True)
    engine_id = Column(Integer, nullable=False)
   
    predicted_rul = Column(Float, nullable=False)
    status = Column(String(20), nullable=False)
    from sqlalchemy.sql import func
    created_at = Column(DateTime, server_default=func.now())