from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from pydantic import BaseModel
from bson import ObjectId
import json
import base64
import re

from db.mongo import lab_results_collection, db
from core.dependencies import get_current_user, is_admin
from core.templates import templates
from services.llm import stream_llm_response

# Service Imports
from services.llm import OLLAMA_URL, MODEL_NAME
from services.analysis_service import AnalysisService
from services.vision_service import resnet_engine
import httpx

router = APIRouter(tags=["Diagnostic Hub"])
imaging_collection = db["visual_analysis"]
synthesis_collection = db["clinical_synthesis"]

from schemas import SynthesisRequest

async def generate_synthesis(lab_context: str, scan_context: str):
    """
    Agentic Diagnostic Synthesis (Streaming).
    Correlates pathology (labs) and radiology (scans) for a unified clinical view.
    Yields: {"type": "token", "data": "..."} or {"type": "confidence", "data": "..."}
    """
    if not lab_context or not scan_context:
        yield {"type": "error", "message": "Incomplete clinical data streams."}
        return

    prompt = f"""
    ACT AS THE CHIEF MEDICAL OFFICER. Synthesize these heterogeneous data streams into a unified Diagnostic Synthesis.
    PERSONA: Senior Consultant, Clinical Pathologist & Radiologist.
    
    [LABORATORY STREAM]
    {lab_context}
    
    [VISUAL IMAGING STREAM]
    {scan_context}
    
    ELITE CLINICAL STEPS:
    1. **Symptomatic Fusion**: Correlate blood markers with imaging findings. 
    2. **Differential Diagnosis**: Provide 3 prioritized conditions with "Confidence %" for each.
    3. **Confirmatory Path**: Specific tests to resolve clinical ambiguity.
    4. **Patient Safety Guide**: Red Flags and Physician Dialogue questions.
    
    OUTPUT FORMAT:
    Use professional <b> headers. NO Markdown. Output strictly the clinical response in HTML (<p> and <b>).
    """

    full_content = ""
    try:
        from services.llm import stream_llm_with_fallback
        for token in stream_llm_with_fallback(prompt):
            full_content += token
            yield {"type": "token", "data": token}
        
        # After full content, estimate confidence
        confidence = "High"
        if "LOW" in full_content.upper(): confidence = "Low"
        elif "MEDIUM" in full_content.upper(): confidence = "Medium"
        
        yield {"type": "confidence", "data": confidence}
        yield {"type": "done"}
    except Exception as e:
        print(f"Synthesis Error: {e}")
        yield {"type": "error", "message": "AI Synthesis service unavailable."}

@router.get("/diagnosis", response_class=HTMLResponse)
async def diagnosis_page(request: Request, user_email: str = Depends(get_current_user)):
    if not user_email: return RedirectResponse("/login")
    
    # Fetch recent labs
    labs_cursor = lab_results_collection.find({"user_email": user_email}).sort("timestamp", -1).limit(5)
    labs = [{"id": str(d["_id"]), "filename": d.get("filename", "Lab Report"), "date": d.get("timestamp", datetime.utcnow()).strftime("%Y-%m-%d")} for d in labs_cursor]
    
    # Fetch recent scans
    scans_cursor = imaging_collection.find({"user_email": user_email}).sort("recorded_at", -1).limit(5)
    scans = [{"id": str(d["_id"]), "filename": d.get("filename", "Scan"), "date": d.get("recorded_at", datetime.utcnow()).strftime("%Y-%m-%d")} for d in scans_cursor]
    
    return templates.TemplateResponse("diagnosis.html", {
        "request": request, "active_page": "diagnosis", 
        "labs": labs, "scans": scans, "user_email": user_email,
        "is_admin": await is_admin(user_email)   # Fixed: was missing `await`, causing coroutine to be truthy for ALL users
    })

@router.post("/api/diagnosis/synthesize")
async def synthesize_reports(req: SynthesisRequest, user_email: str = Depends(get_current_user)):
    from fastapi.responses import StreamingResponse
    import json

    # 1. Fetch Records
    lab = lab_results_collection.find_one({"_id": ObjectId(req.lab_id), "user_email": user_email})
    scan = imaging_collection.find_one({"_id": ObjectId(req.scan_id), "user_email": user_email})
    
    if not lab or not scan:
        raise HTTPException(status_code=404, detail="Records not found.")
        
    # 2. Extract Context
    lab_summary = lab.get("human_narrative", "")
    lab_key_data = json.dumps(lab.get("structured_metrics", {}).get("triage", {}))
    full_lab = f"Narrative: {lab_summary}\nTriage Data: {lab_key_data}"

    scan_finding = scan.get("analysis", {}).get("finding", "Unknown")
    scan_narrative = scan.get("analysis", {}).get("narrative", "")
    full_scan = f"Primary Finding: {scan_finding}\nNarrative: {scan_narrative}"

    async def stream_synthesis():
        full_synth = ""
        confidence = "High"
        async for chunk in generate_synthesis(full_lab, full_scan):
            if chunk["type"] == "token":
                full_synth += chunk["data"]
                yield json.dumps({"status": "token", "data": chunk["data"]}) + "\n"
            elif chunk["type"] == "confidence":
                confidence = chunk["data"]
                yield json.dumps({"status": "confidence", "data": confidence}) + "\n"
            elif chunk["type"] == "done":
                # Persistence
                db_entry = {
                    "user_email": user_email,
                    "lab_id": req.lab_id,
                    "scan_id": req.scan_id,
                    "synthesis": full_synth,
                    "confidence": confidence,
                    "recorded_at": datetime.utcnow(),
                    "module": "diagnosis"
                }
                synthesis_collection.insert_one(db_entry)
                yield json.dumps({"status": "success", "lab_summary": f"ID {req.lab_id[-4:]}", "scan_summary": f"ID {req.scan_id[-4:]}"}) + "\n"
            elif chunk["type"] == "error":
                yield json.dumps({"status": "error", "message": chunk["message"]}) + "\n"

    return StreamingResponse(stream_synthesis(), media_type="application/x-ndjson")

@router.post("/api/diagnosis/upload_and_synthesize")
async def upload_and_synthesize(
    lab_file: UploadFile = File(...),
    scan_file: UploadFile = File(...),
    scan_type: str = Form(...),
    user_email: str = Depends(get_current_user)
):
    from fastapi.responses import StreamingResponse
    import json

    async def stream_upload_synthesis():
        try:
            # A. PROCESS LAB REPORT
            lab_bytes = await lab_file.read()
            # Optimization: Use the new streaming-capable analyzer if possible, 
            # but here we just need the summary for synthesis context.
            lab_record = await AnalysisService.process_and_save_report(user_email, lab_bytes)
            
            if not lab_record:
                yield json.dumps({"status": "error", "message": "Failed to analyze Lab Report."}) + "\n"
                return
                
            lab_id = lab_record["id"]
            lab_summary = lab_record.get("human_narrative", "")
            lab_context = f"Narrative: {lab_summary}\nMetrics: {json.dumps(lab_record['structured_metrics'].get('triage', {}))}"

            # B. PROCESS IMAGING SCAN
            scan_bytes = await scan_file.read()
            vision_result = resnet_engine.run_inference(scan_bytes, scan_type=scan_type)
            
            scan_record = {
                "user_email": user_email,
                "filename": scan_file.filename,
                "scan_type": scan_type,
                "recorded_at": datetime.utcnow(),
                "analysis": {
                    "reading": vision_result['finding'],
                    "confidence": vision_result['confidence'],
                    "heatmap": vision_result['heatmap'],
                    "narrative": vision_result.get('narrative', f"AI detected {vision_result['finding']}"),
                    "finding": vision_result['finding']
                }
            }
            res_scan = imaging_collection.insert_one(scan_record)
            scan_id = str(res_scan.inserted_id)
            scan_context = f"Finding: {vision_result['finding']}\nNarrative: {scan_record['analysis']['narrative']}"

            # C. SYNTHESIZE (STREAMING)
            full_synth = ""
            confidence = "High"
            async for chunk in generate_synthesis(lab_context, scan_context):
                if chunk["type"] == "token":
                    full_synth += chunk["data"]
                    yield json.dumps({"status": "token", "data": chunk["data"]}) + "\n"
                elif chunk["type"] == "confidence":
                    confidence = chunk["data"]
                    yield json.dumps({"status": "confidence", "data": confidence}) + "\n"
                elif chunk["type"] == "done":
                    # SAVE SYNTHESIS
                    synthesis_collection.insert_one({
                        "user_email": user_email,
                        "lab_id": lab_id,
                        "scan_id": scan_id,
                        "synthesis": full_synth,
                        "confidence": confidence,
                        "recorded_at": datetime.utcnow(),
                        "module": "diagnosis"
                    })
                    yield json.dumps({
                        "status": "success", 
                        "lab_summary": f"Processed {lab_file.filename}", 
                        "scan_summary": f"Processed {vision_result['finding']}"
                    }) + "\n"
                elif chunk["type"] == "error":
                    yield json.dumps({"status": "error", "message": chunk["message"]}) + "\n"

        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream_upload_synthesis(), media_type="application/x-ndjson")
