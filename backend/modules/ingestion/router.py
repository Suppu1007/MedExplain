
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.modules.ingestion.service import process_document

api_router = APIRouter()

@api_router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a medical report (PDF or Image) and extract text.
    """
    try:
        result = await process_document(file)
        return {
            "status": "success",
            "data": result
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
