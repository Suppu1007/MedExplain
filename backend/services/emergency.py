EMERGENCY_KEYWORDS = [
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "severe bleeding",
    "loss of consciousness",
    "stroke",
    "heart attack",
]

def detect_emergency(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in EMERGENCY_KEYWORDS)
