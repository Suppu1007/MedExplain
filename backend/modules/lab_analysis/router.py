from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from bson import ObjectId
from datetime import datetime
import traceback

# Local imports
from app.db.mongo import lab_results_collection
from app.core.dependencies import get_current_user
from app.services.pdf_service import PDFService
from app.modules.lab_analysis.analyzer import LabAnalyzer
from app.modules.lab_analysis.ai_interpretation import generate_interpretation

router = APIRouter(tags=["Lab Analysis"])
templates = Jinja2Templates(directory="app/frontend/templates")

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
        "user_email": user_email
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
    End-to-End Processing:
    1. OCR (PDF/Image to Text)
    2. Model Synthesis (JSON for visual map + Narrative Text)
    3. Storage (MongoDB)
    """
    try:
        # 1. Validation
        if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Invalid file format. Use PDF or Image.")

        # 2. Extract raw text via PDFService (Handles Images & PDFs)
        content = await file.read()
        raw_text = PDFService.extract_text(content, file.filename)
        
        if not raw_text or len(raw_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Document content is unreadable.")

        # 3. Model Synthesis via LabAnalyzer
        # Returns: {"structured": {...}, "narrative": "..."}
        result = LabAnalyzer.process_report(raw_text)
        
        if not result["structured"]:
            raise HTTPException(status_code=422, detail="AI failed to extract structured clinical markers.")

        # 4. Persistent Storage (The Record)
        db_record = {
            "user_email": user_email,
            "filename": file.filename,
            "recorded_at": datetime.utcnow(),
            "structured_metrics": result["structured"], # For the Body Map & Table
            "human_narrative": result["narrative"]      # For the Textual Explanation
        }
        
        lab_results_collection.insert_one(db_record)

        # 5. Return JSON to Frontend
        # We convert datetime for the response
        return {
            "status": "success",
            "data": {
                "structured_metrics": result["structured"],
                "human_narrative": result["narrative"],
                "filename": file.filename
            }
        }

    except Exception as e:
        print(f"❌ Analysis Endpoint Error: {e}")
        traceback.print_exc()
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail="Internal processing error.")

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
        print(f"❌ Results Fetch Error: {e}")
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
from app.services.export_service import ExportService

@router.post("/api/labs/export")
async def export_lab_report(
    payload: dict, # Receives the current analysis object
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