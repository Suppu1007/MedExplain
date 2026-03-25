
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
from backend.modules.cdss.service import cdss_service

api_router = APIRouter()

@api_router.post("/predict")
async def predict_condition(data: Dict[str, Any] = Body(...)):
    """
    Get clinical decision support based on patient data.
    """
    try:
        prediction = cdss_service.predict_disease(data)
        return {
            "status": "success",
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
