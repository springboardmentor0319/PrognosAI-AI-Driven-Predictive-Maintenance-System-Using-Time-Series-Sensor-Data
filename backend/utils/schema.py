from pydantic import BaseModel
from typing import List


class SensorInput(BaseModel):
    dataset: str
    engine_id: int
    cycle: int
    features: List[float]

class BatchItem(BaseModel):
    engine_id: int
    cycle: int
    features: List[float]

class BatchInput(BaseModel):
    dataset: str
    batch: List[BatchItem]