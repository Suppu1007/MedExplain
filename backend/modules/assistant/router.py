from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
from datetime import datetime

from core.dependencies import get_current_user
from db.mongo import conversations_collection
from services.emergency import detect_emergency
from services.llm import generate_response, stream_generate_response

from rag.retriever import retrieve_context
from rag.prompt import build_rag_prompt

api_router = APIRouter(
    prefix="/api",
    tags=["AI Assistant"],
)

from core.templates import templates

from schemas import ChatRequest, ChatResponse


from db.mongo import db

async def get_clinical_context(user_email: str) -> str:
    """Fetch and summarize the last 5 clinical records for the user."""
    try:
        # Fetch 5 most recent labs
        labs = list(db["lab_results"].find({"user_email": user_email}).sort("recorded_at", -1).limit(5))
        # Fetch 5 most recent imaging scans
        scans = list(db["visual_analysis"].find({"user_email": user_email}).sort("recorded_at", -1).limit(5))
        
        summary = []
        for l in labs:
            date = l.get("recorded_at", datetime.now()).strftime("%Y-%m-%d")
            markers = l.get("structured_metrics", {}).get("markers", [])
            marker_str = ", ".join([f"{m['name']}:{m['value']}" for m in markers[:3]])
            summary.append(f"Lab ({date}): {marker_str}...")
            
        for s in scans:
            date = s.get("recorded_at", datetime.now()).strftime("%Y-%m-%d")
            finding = s.get("analysis", {}).get("finding", "Unknown")
            summary.append(f"Scan ({date}): {finding}")
            
        return "\n".join(summary) if summary else "No previous clinical records found."
    except Exception as e:
        print(f"Error fetching clinical context: {e}")
        return "Error retrieving history."

# =========================
# CHAT ENDPOINT (MEDQUAD CSV RAG)
# =========================
@api_router.post("/assistant/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user_email: str = Depends(get_current_user),
):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Emergency detection
    emergency = detect_emergency(message)

    # Retrieve MedQuAD context
    context, sources = retrieve_context(message)
    
    # Fetch clinical context (Memory)
    clinical_context = await get_clinical_context(user_email)

    # Build safe RAG prompt
    prompt = build_rag_prompt(message, context, clinical_context)

    # LLM inference
    reply = generate_response(prompt)

    # Mongo audit log
    conversations_collection.insert_one({
        "user_email": user_email,
        "question": message,
        "reply": reply,
        "sources": sources,
        "emergency": emergency,
        "created_at": datetime.utcnow(),
        "pipeline": "rag_medquad_csv",
    })

    return {
        "reply": reply,
        "emergency": emergency,
        "sources": sources,
        "disclaimer": (
            "This response is for personalized educational purposes based on your history "
            "and is not a medical diagnosis."
        ),
    }


# =========================
# STREAMING ENDPOINT (SSE)
# =========================
@api_router.post("/assistant/stream")
async def stream_chat(
    payload: ChatRequest,
    user_email: str = Depends(get_current_user),
):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Fetch clinical context (Memory)
    clinical_context = await get_clinical_context(user_email)

    def event_generator():
        # Emergency detection
        emergency = detect_emergency(message)

        # Retrieve RAG context
        context, sources = retrieve_context(message)

        # Build prompt
        prompt = build_rag_prompt(message, context, clinical_context)

        # Stream LLM response
        full_reply = ""
        try:
            for token in stream_generate_response(prompt):
                full_reply += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # End of stream
        yield f"data: {json.dumps({'done': True})}\n\n"

        # Save to DB
        conversations_collection.insert_one({
            "user_email": user_email,
            "question": message,
            "reply": full_reply,
            "sources": sources,
            "emergency": emergency,
            "created_at": datetime.utcnow(),
            "pipeline": "rag_medquad_csv_stream",
        })

    return StreamingResponse(event_generator(), media_type="text/event-stream")

