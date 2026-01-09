from app.modules.assistant.prompt import SYSTEM_PROMPT

def generate_ai_response(user_message: str, context: str = "") -> str:
    """
    Replace this with:
    - OpenAI
    - Azure OpenAI
    - Ollama (local)
    """

    # ---- TEMP MOCK (safe baseline) ----
    response = f"""
🩺 MediExplain AI Response

Your question:
"{user_message}"

Explanation:
Based on general medical knowledge, this topic relates to common clinical concepts.
This information is for educational purposes only and is NOT a diagnosis.

Recommendation:
Please consult a healthcare professional for medical advice.
"""

    return response.strip()
