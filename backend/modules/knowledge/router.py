from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from core.dependencies import get_current_user, is_admin, admin_required
from core.templates import templates

ui_router = APIRouter(tags=["Knowledge"])

@ui_router.get("/knowledge", response_class=HTMLResponse)
async def knowledge_page(
    request: Request,
    user_email: str = Depends(admin_required),
):
    """
    Knowledge Textbook Page.
    Internal training resources for Narrative AI.
    """
    is_admin_user = await is_admin(user_email)
    
    return templates.TemplateResponse(
        "knowledge.html",
        {
            "request": request,
            "active_page": "knowledge",
            "is_admin": is_admin_user,
            "user": user_email,
        }
    )

# =====================================================
# API ROUTES
# =====================================================
from fastapi import UploadFile, File, HTTPException
from services.rag_service import rag_service
from core.config import knowledge_collection
from datetime import datetime

@ui_router.post("/api/knowledge/upload")
async def upload_textbook(
    file: UploadFile = File(...),
    user_email: str = Depends(admin_required)
):
    """
    Ingests a new medical textbook/guideline PDF.
    """
    try:
        if not file.filename.lower().endswith(('.pdf', '.txt', '.md')):
            raise HTTPException(status_code=400, detail="Only PDF or Text files are supported.")

        content = await file.read()
        
        # Call RAG Service
        result = rag_service.ingest_document(content, file.filename)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=422, detail=result.get("message"))
            
        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed.")

@ui_router.get("/api/knowledge/library")
async def get_library(user_email: str = Depends(admin_required)):
    """
    Returns unique sources currently in the Knowledge Base.
    """
    try:
        # Aggregation to get unique filenames and their chunk counts/dates
        pipeline = [
            {
                "$group": {
                    "_id": "$filename",
                    "chunks": {"$sum": 1},
                    "ingested_at": {"$first": "$ingested_at"},
                    "source_id": {"$first": "$source_id"}
                }
            },
            {"$sort": {"ingested_at": -1}}
        ]
        
        library_data = list(knowledge_collection.aggregate(pipeline))
        
        # Format for UI
        results = []
        for item in library_data:
            results.append({
                "filename": item["_id"],
                "chunks": item["chunks"],
                "date": item["ingested_at"].strftime("%Y-%m-%d %H:%M") if item.get("ingested_at") else "N/A"
            })
            
        return results
    except Exception as e:
        print(f"Library Fetch Error: {e}")
        return []
