from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Optional, Dict, Any
import shutil
import os
from pathlib import Path

# Import services
from backend.services.vision_service import resnet_engine
from backend.services.rag_service import rag_service
from backend.modules.lab_analysis.analyzer import LabAnalyzer
from backend.services.llm_router import stream_llm_with_fallback
from backend.main import templates
from backend.core.dependencies import get_current_user

router = APIRouter(
    tags=["multimodal"]
)

@router.get("/multimodal", response_class=HTMLResponse)
async def multimodal_ui(request: Request, user_email: str = Depends(get_current_user)):
    return templates.TemplateResponse("multimodal.html", {
        "request": request,
        "active_page": "multimodal",
        "user_email": user_email
    })

@router.post("/api/multimodal/analyze")
async def analyze_multimodal(
    file_image: UploadFile = File(...),
    file_report: UploadFile = File(...)
):
    """
    Analyze both a medical image and a lab report simultaneously.
    Synthesizes findings from both sources.
    """
    
    # 1. Save files temporarily (needed for PDF text extraction)
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    image_path = temp_dir / f"multi_{file_image.filename}"
    report_path = temp_dir / f"multi_{file_report.filename}"
    
    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file_image.file, buffer)
            
        with open(report_path, "wb") as buffer:
            shutil.copyfileobj(file_report.file, buffer)
            
        # 2. Process Image (Vision Analysis)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        # Run inference (defaulting to 'chest' for now, or we could add a form field)
        vision_result = resnet_engine.run_inference(image_bytes, scan_type="chest")
        
        # Get RAG context for the image finding
        vision_context = rag_service.retrieve_context(vision_result.get('finding', 'unknown'))
        vision_result["rag_context"] = vision_context
        
        # 3. Process Lab Report (Text Analysis)
        import pdfplumber
        extracted_text = ""
        try:
            with pdfplumber.open(report_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
        except Exception as e:
            print(f"PDF Error: {e}")
            extracted_text = "Could not extract text from PDF."
                
        lab_result = LabAnalyzer.process_report(extracted_text)
        
        # 4. Synthesize Results (LLM)
        synthesis = await synthesize_findings(vision_result, lab_result)
        
        return JSONResponse(content={
            "vision_analysis": vision_result,
            "lab_analysis": lab_result,
            "unified_diagnosis": synthesis
        })

    except Exception as e:
        print(f"Multimodal Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Multimodal analysis failed: {str(e)}")
    finally:
        # Cleanup
        try:
            if image_path.exists(): os.remove(image_path)
            if report_path.exists(): os.remove(report_path)
        except:
            pass

async def synthesize_findings(vision_data: Dict, lab_data: Dict) -> Dict:
    """
    Uses LLM to find correlations between image and lab data.
    """
    
    # Extract key findings
    image_finding = vision_data.get("finding", "Unknown")
    image_conf = vision_data.get("confidence", "Unknown")
    
    lab_structured = lab_data.get("structured", {})
    abnormal_markers = []
    if lab_structured and isinstance(lab_structured, dict) and "markers" in lab_structured:
        for m in lab_structured["markers"]:
            # Check for various status strings
            status = str(m.get("status", "")).lower()
            if status in ["abnormal", "critical", "high", "low"]:
                abnormal_markers.append(f"{m.get('name')}: {m.get('value')} ({status})")
    
    prompt = f"""
    You are a Senior Chief Medical Officer at a top diagnostics center.
    Your task is to SYNTHESIZE findings from two different diagnostic modalities for the SAME PATIENT.
    
    === MODALITY 1: RADIOLOGY (Medical Imaging) ===
    - Primary Diagnosis: {image_finding}
    - AI Confidence: {image_conf}
    - Severity: {vision_data.get('priority', 'Unknown')}
    
    === MODALITY 2: PATHOLOGY (Lab Report) ===
    - Abnormal Biomarkers: {', '.join(abnormal_markers) if abnormal_markers else 'No critical abnormalities detected'}
    - Lab Summary: {lab_data.get("narrative", "")[:600]}...
    
    === GOAL ===
    Write a "Unified Clinical correlation" that answers:
    1. Do the blood markers support the radiology finding? (e.g., High WBCs supporting Pneumonia?)
    2. Are there any contradictions?
    3. What is the comprehensive assessment?
    
    Format nicely in Markdown.
    """
    
    full_response = ""
    for token in stream_llm_with_fallback(prompt):
        full_response += token
        
    return {"narrative": full_response}
