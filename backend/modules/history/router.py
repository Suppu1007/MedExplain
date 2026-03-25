from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from db.mongo import db
from core.dependencies import get_current_user, is_admin
from datetime import datetime

router = APIRouter(tags=["History"])
from core.templates import templates

@router.get("/history", response_class=HTMLResponse)
async def get_history_ui(request: Request, user_email: str = Depends(get_current_user)):
    if not user_email:
        return RedirectResponse("/login")

    try:
        # 1. Fetch Lab Reports
        labs = list(db["lab_results"].find({"user_email": user_email}))
        
        # 2. Fetch Visual Scans
        scans = list(db["visual_analysis"].find({"user_email": user_email}))

        # 3. Fetch Synthesized Diagnoses
        syntheses = list(db["clinical_synthesis"].find({"user_email": user_email}))

        timeline = []

        # --- PROCESS LABS ---
        for l in labs:
            # Use 'or {}' to handle cases where the field is null in DB
            structured = l.get('structured_metrics') or {}
            triage = structured.get('triage') or {}
            markers = structured.get('markers') or []
            
            # Ensure we have a date object for sorting
            date_obj = l.get("recorded_at") or l.get("timestamp") or datetime.now()

            timeline.append({
                "type": "Lab Report",
                "icon": "biotech",
                "raw_date": date_obj,
                "date": date_obj.strftime("%b %d, %Y"),
                "title": f"Markers: {len(markers)} Analyzed",
                "status": str(triage.get('urgency', 'NORMAL')).upper(),
                "action_url": "/labs"
            })
        
        # --- PROCESS SCANS (Fixed the Crash Here) ---
        for s in scans:
            # Defensive check: (s.get('analysis') or {}) prevents the NoneType error
            analysis = s.get('analysis') or {}
            
            # Handle legacy string comparisons (migration fallback)
            if isinstance(analysis, str):
                target = "Unknown"
                status = "NORMAL"
                title = f"Scan: {analysis}"
            else:
                # New Schema
                finding = analysis.get('finding', 'Unspecified')
                svg_id = analysis.get('svg_id', 'scan')
                triage = analysis.get('triage', 'NORMAL')
                
                target = svg_id
                status = str(triage).upper()
                title = f"Scan: {finding}"

            date_obj = s.get("timestamp") or s.get("recorded_at") or datetime.now()

            timeline.append({
                "type": "Imaging Scan",
                "icon": "wallpaper",
                "raw_date": date_obj,
                "date": date_obj.strftime("%b %d, %Y"),
                "title": title,
                "status": status,
                "thumbnail": analysis.get('heatmap'), # Pass heatmap as preview
                "action_url": "/visualanalysis"
            })

        # --- PROCESS SYNTHESES ---
        for sy in syntheses:
            date_obj = sy.get("recorded_at") or datetime.now()
            timeline.append({
                "type": "Integrated Diagnosis",
                "icon": "psychology",
                "raw_date": date_obj,
                "date": date_obj.strftime("%b %d, %Y"),
                "title": "Unified Clinical Assessment",
                "status": "COMPLETED",
                "action_url": "/diagnosis"
            })

        # 3. Robust Sorting (Newest First)
        timeline.sort(key=lambda x: x['raw_date'], reverse=True)

        # Check admin status
        is_admin_user = await is_admin(user_email)

        return templates.TemplateResponse("history.html", {
            "request": request, 
            "active_page": "history", 
            "is_admin": is_admin_user,
            "timeline": timeline
        })

    except Exception as e:
        print(f"History Router Error: {e}")
        # Return empty timeline instead of crashing the whole page
        return templates.TemplateResponse("history.html", {
            "request": request, 
            "active_page": "history", 
            "timeline": []
        })