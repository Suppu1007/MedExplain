from app.services.llm_router import stream_llm_with_fallback

def classify_lab_value(value, ref_min, ref_max):
    """
    Categorizes a lab result value based on provided reference ranges.
    Calculates 'critical' status if the value is 30% beyond the range limits.
    """
    if ref_min is None or ref_max is None:
        return "normal"
    
    try:
        val = float(value)
        r_min = float(ref_min)
        r_max = float(ref_max)

        # Critical thresholds: 30% deviation from the normal limits
        if val < (r_min * 0.7) or val > (r_max * 1.3):
            return "critical"
        
        # Abnormal: Outside range but not yet critical
        if val < r_min or val > r_max:
            return "abnormal"
            
    except (ValueError, TypeError):
        # If conversion fails, default to normal to avoid false alarms
        return "normal"
        
    return "normal"

def build_lab_prompt(lab_data: dict) -> str:
    """
    Constructs a structured prompt for the LLM to provide 
    educational medical explanations.
    """
    return f"""
You are MediExplain, an expert educational medical AI assistant.

Your goal is to explain a specific laboratory test result in simple, plain language that a patient can understand. 

LABORATORY DATA:
- Test Name: {lab_data.get('test_name')}
- Category: {lab_data.get('category')}
- Result: {lab_data.get('value')} {lab_data.get('unit')}
- Reference Range: {lab_data.get('ref_min')} - {lab_data.get('ref_max')}
- Automatic Classification: {lab_data.get('status')}

GUIDELINES:
1. EDUCATIONAL ONLY: Provide information about what this marker generally indicates in the human body.
2. NO DIAGNOSIS: Do not tell the user they "have" a specific disease. Use phrases like "this is often associated with" or "doctors look at this to evaluate...".
3. CLARITY: Explain the medical term in 1 sentence.
4. ACTIONABLE ADVICE: Suggest specific questions they should ask their doctor regarding this result.
5. CONCISE: Keep the entire response under 150 words.

Explain what this result means and its general clinical significance.
"""

def generate_interpretation(lab_data: dict) -> str:
    """
    Calls the LLM service to generate a human-readable 
    explanation of a lab result.
    """
    # Ensure lab_data is handled as a dictionary (standard for MongoDB results)
    prompt = build_lab_prompt(lab_data)
    
    explanation_text = ""
    try:
        # Use the streaming service to gather the full response
        for token in stream_llm_with_fallback(prompt):
            explanation_text += token
            
    except Exception as e:
        print(f"❌ LLM Interpretation Error: {e}")
        return "MediExplain is currently unable to synthesize an explanation for this result. Please consult your healthcare provider."

    return explanation_text.strip()