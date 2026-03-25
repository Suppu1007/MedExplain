from pydantic import BaseModel

class SynthesisRequest(BaseModel):
    lab_id: str
    scan_id: str
