from fastapi import APIRouter, UploadFile, File
from app.services.image_analysis_service import analyze_xray

router = APIRouter()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    return analyze_xray(file)
