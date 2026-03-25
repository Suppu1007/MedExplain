from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class LabResultCreate(BaseModel):
    user_email: str
    test_name: str
    category: str
    value: float
    unit: str
    ref_min: Optional[float] = None
    ref_max: Optional[float] = None
    status: str  # normal | abnormal | critical

class LabResultDB(LabResultCreate):
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
