from pydantic import BaseModel
from typing import List

class SensorInput(BaseModel):
    dataset: str
    features: List[float]

class BatchInput(BaseModel):
    dataset: str
    batch: List[List[float]]