from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime

from app.core.dependencies import get_current_user
from app.db.mongo import conversations_collection
from app.services.emergency import detect_emergency
from app.services.llm import generate_response

from app.rag.retriever import retrieve_context
from app.rag.prompt import build_rag_prompt

api_router = APIRouter(
    prefix="/api/assistant",
    tags=["AI Assistant"],
)

# =========================
# SCHEMAS
# =========================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    emergency: bool
    disclaimer: str
    sources: list[str]


# =========================
# CHAT ENDPOINT (MEDQUAD CSV RAG)
# =========================
@api_router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user_email: str = Depends(get_current_user),
):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 🚨 Emergency detection
    emergency = detect_emergency(message)

    # 🔍 Retrieve MedQuAD context
    context, sources = retrieve_context(message)

    # 🧠 Build safe RAG prompt
    prompt = build_rag_prompt(message, context)

    # 🤖 LLM inference
    reply = generate_response(prompt)

    # 🗃️ Mongo audit log
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
            "This response is for educational purposes only "
            "and is not a medical diagnosis."
        ),
    }
