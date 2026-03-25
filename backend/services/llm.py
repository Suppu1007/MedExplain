import os
import requests
import httpx
import json
import base64
from typing import Generator
from groq import Groq

# =====================================================
# CONFIGURATION & INITIALIZATION
# =====================================================
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:1b") 
VISION_MODEL = os.getenv("VISION_MODEL", "moondream")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = (
    "You are MediExplain, a medical education AI. Provide thorough, "
    "step-by-step educational explanations using metaphors. "
    "Never truncate your answers. Provide full clinical context. "
    "Do not diagnose or prescribe.\n\n"
)

# =====================================================
# 1. GROQ ENGINE (Strictly for High-Speed Analysis)
# =====================================================
def stream_groq_response(prompt: str) -> Generator[str, None, None]:
    """
    Handles clinical data synthesis using Groq Llama 3.3 70B.
    """
    if not groq_client:
        yield from stream_llm_response(prompt) # Fallback to local text
        return

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=8192,
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"Groq API Error: {e}")
        yield from stream_llm_response(prompt)


# =====================================================
# 2. OLLAMA TEXT ENGINE (Strictly for Chat/Fallback)
# =====================================================
def stream_llm_response(prompt: str) -> Generator[str, None, None]:
    """
    Streams from local Ollama using /api/chat (Standard for Text).
    """
    url = f"{OLLAMA_URL}/api/chat"
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "options": {
            "num_ctx": 8192,
            "num_predict": 4096,
            "temperature": 0.2
        }
    }

    try:
        with httpx.stream("POST", url, json=payload, timeout=None) as response:
            if response.status_code != 200:
                yield f"Error: Ollama text service returned {response.status_code}"
                return

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
    except Exception as e:
        print(f"Ollama Connection Error: {e}")
        yield "Local AI service is offline."


# =====================================================
# 3. OLLAMA VISION ENGINE (Strictly for Imaging)
# =====================================================
def stream_vision_response(prompt: str, image_bytes: bytes) -> Generator[str, None, None]:
    """
    Uses Local Ollama (Moondream) to analyze images using /api/generate.
    """
    url = f"{OLLAMA_URL}/api/generate"
    
    # Encode image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [base64_image],
        "stream": True
    }

    try:
        with httpx.stream("POST", url, json=payload, timeout=None) as response:
            if response.status_code != 200:
                yield f"Vision Error: Ollama vision returned {response.status_code}"
                return

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Local Vision Error: {e}")
        yield "Visual analysis engine is offline."


# =====================================================
# 4. SYNC & UTILITY HELPERS
# =====================================================

def generate_response(prompt: str) -> str:
    """
    Synchronous version for public chat and non-streaming tasks.
    Uses Groq first for quality, then local Ollama /api/chat.
    """
    if groq_client:
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            return res.choices[0].message.content
        except: pass

    # Local Fallback
    url = f"{OLLAMA_URL}/api/chat"
    try:
        response = requests.post(url, json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_ctx": 4096}
        }, timeout=120)
        return response.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"Sync LLM Error: {e}")
        return "AI generation currently unavailable."

def stream_generate_response(prompt: str) -> Generator[str, None, None]:
    """Compatibility alias for the Assistant module."""
    yield from stream_llm_with_fallback(prompt)

def stream_llm_with_fallback(prompt: str, use_groq: bool = False) -> Generator[str, None, None]:
    """
    Main Router: Separates Cloud (Groq) from Local (Ollama) logic.
    """
    if use_groq and GROQ_API_KEY:
        yield from stream_groq_response(prompt)
    else:
        yield from stream_llm_response(prompt)