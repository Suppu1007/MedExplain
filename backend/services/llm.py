# app/services/llm.py
import os
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

SYSTEM_PROMPT = (
    "You are a medical education assistant. "
    "Provide educational information only. "
    "Do not diagnose or prescribe treatments."
)

def generate_response(prompt: str) -> str:
    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser question:\n{prompt}"

        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
            },
            timeout=120,
        )

        res.raise_for_status()
        data = res.json()

        return data.get("response", "").strip() or (
            "I could not generate a response at this time."
        )

    except Exception as e:
        print("LLM ERROR:", e)
        return (
            "The AI service is temporarily unavailable.\n\n"
            "This assistant provides educational information only "
            "and is not a medical diagnosis."
        )
