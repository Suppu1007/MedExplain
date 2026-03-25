from fastapi import APIRouter, Body, Depends, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from bson import ObjectId
from datetime import datetime
import traceback

# Local imports
from db.mongo import lab_results_collection
from core.dependencies import get_current_user, is_admin
from services.pdf_service import PDFService
from modules.lab_analysis.analyzer import LabAnalyzer
from modules.lab_analysis.ai_interpretation import generate_interpretation

router = APIRouter(tags=["Lab Analysis"])
from core.templates import templates

# =====================================================
# UI ROUTE: DASHBOARD
# =====================================================

@router.get("/labs", response_class=HTMLResponse)
async def get_labs_dashboard(request: Request, user_email: str = Depends(get_current_user)):
    """
    Renders the clinical analysis dashboard.
    URL: /labs
    """
    if not user_email:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "active_page": "labs",
        "user_email": user_email,
        "is_admin": await is_admin(user_email)
    })

# =====================================================
# API ROUTE: HYBRID ANALYSIS (JSON + TEXT)
# =====================================================

@router.post("/api/labs/full-analysis")
async def perform_full_analysis(
    file: UploadFile = File(...),
    user_email: str = Depends(get_current_user)
):
    """
    Real-time Streaming Endpoint:
    Yields JSON chunks for markers and narrative as they generate.
    """
    from fastapi.responses import StreamingResponse
    import json

    async def stream_analysis():
        try:
            # 1. Validation & Extraction
            content = await file.read()
            raw_text = PDFService.extract_text(content, file.filename)
            
            if not raw_text or len(raw_text.strip()) < 10:
                yield json.dumps({"status": "error", "message": "Document unreadable."}) + "\n"
                return

            full_structured = None
            full_narrative = ""

            # 2. Call the generator
            async for chunk in LabAnalyzer.process_report(raw_text):
                if chunk["type"] == "error":
                    yield json.dumps({"status": "error", "message": chunk["message"]}) + "\n"
                    return
                
                if chunk["type"] == "structured":
                    full_structured = chunk["data"]
                    yield json.dumps({"status": "structured", "data": full_structured}) + "\n"
                
                if chunk["type"] == "narrative_chunk":
                    full_narrative += chunk["data"]
                    yield json.dumps({"status": "narrative_chunk", "data": chunk["data"]}) + "\n"
                
                if chunk["type"] == "done":
                    # 3. Persistent Storage at the end
                    db_record = {
                        "user_email": user_email,
                        "filename": file.filename,
                        "recorded_at": datetime.utcnow(),
                        "structured_metrics": full_structured,
                        "human_narrative": full_narrative,
                        "global_confidence": full_structured.get("global_confidence", "95.0%")
                    }
                    lab_results_collection.insert_one(db_record)
                    yield json.dumps({"status": "done"}) + "\n"

        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream_analysis(), media_type="application/x-ndjson")


# =====================================================
# API ROUTE: FETCH RESULTS
# =====================================================

@router.get("/api/labs/results")
def get_user_results(user_email: str = Depends(get_current_user)):
    """
    Fetches past analyses for the specific user.
    """
    try:
        cursor = lab_results_collection.find({"user_email": user_email}).sort("recorded_at", -1)
        
        results = []
        for doc in cursor:
            results.append({
                "id": str(doc["_id"]),
                "date": doc["recorded_at"].strftime("%Y-%m-%d %H:%M"),
                "filename": doc.get("filename", "Unknown"),
                # We return the top-level metrics for the table
                "structured": doc.get("structured_metrics"),
                "narrative": doc.get("human_narrative")
            })
        return results
    except Exception as e:
        print(f"Results Fetch Error: {e}")
        return []

# =====================================================
# API ROUTE: STANDALONE INTERPRETATION
# =====================================================

@router.get("/api/labs/interpret/{lab_id}")
def get_standalone_interpretation(lab_id: str, user_email: str = Depends(get_current_user)):
    """
    Generates an AI explanation for an existing result in the DB.
    """
    try:
        if not ObjectId.is_valid(lab_id):
            raise HTTPException(status_code=400, detail="Invalid ID format.")

        record = lab_results_collection.find_one({
            "_id": ObjectId(lab_id),
            "user_email": user_email
        })
        
        if not record:
            raise HTTPException(status_code=404, detail="Record not found.")
        
        # This fallback is useful if you want to re-generate text for an old result
        explanation = generate_interpretation(record.get("structured_metrics", {}))
        
        return {"interpretation": explanation}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail="Interpretation failed.")



from fastapi import Response
from services.export_service import ExportService

@router.post("/api/labs/export")
async def export_lab_report(
    payload: dict = Body(...),  # Receives the current analysis object
    user_email: str = Depends(get_current_user)
):
    """
    Receives current analysis data and returns a downloadable PDF.
    """
    pdf_bytes = ExportService.generate_medical_pdf(payload, user_email)
    
    if not pdf_bytes:
        raise HTTPException(status_code=500, detail="PDF generation failed.")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=MediExplain_Report.pdf"}
    )