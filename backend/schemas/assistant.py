from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    emergency: bool
    disclaimer: str
    sources: List[str]
