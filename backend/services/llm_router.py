from typing import Generator
from services.llm import stream_groq_response, stream_llm_response, GROQ_API_KEY

def stream_llm_with_fallback(prompt: str) -> Generator[str, None, None]:
    """
    The main entry point for all AI modules.
    Prioritizes Groq (8k context) -> Falls back to Ollama (8k context).
    """
    try:
        if GROQ_API_KEY:
            # High speed, 8192 token context
            yield from stream_groq_response(prompt)
        else:
            # Local, 8192 token context
            yield from stream_llm_response(prompt)
            
    except Exception as e:
        print(f"LLM Router Error: {e}")
        yield "MediExplain is temporarily unable to provide a full analysis. Please try a shorter query."