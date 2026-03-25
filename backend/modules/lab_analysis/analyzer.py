import json
import re
from typing import Dict, Any, Optional
from app.services.llm_router import stream_llm_with_fallback
from app.modules.lab_analysis.ai_interpretation import classify_lab_value

class LabAnalyzer:
    @staticmethod
    def process_report(raw_text: str) -> Dict[str, Any]:
        """
        The primary entry point for report intelligence.
        Extracts structured clinical markers, maps them to anatomical regions, 
        identifies specialists, and generates a patient-friendly narrative.
        """
        if not raw_text or len(raw_text.strip()) < 10:
            return {"structured": None, "narrative": "Insufficient text found in report."}

        # 1. Construct the Intelligence Prompt
        prompt = f"""
        You are the MediExplain Clinical Intelligence Engine. 
        Analyze the provided medical lab report text and output TWO distinct sections.

        ### SECTION 1: STRUCTURED DATA
        Wrap this section in [STRUCTURED_START] and [STRUCTURED_END].
        Return a valid JSON object ONLY with this schema:
        {{
          "markers": [
            {{
              "name": "string", 
              "value": float, 
              "unit": "string", 
              "status": "normal|abnormal|critical",
              "category": "e.g., Metabolic, Renal, Hepatic, Cardiac, Respiratory, CBC",
              "ref_min": float|null,
              "ref_max": float|null
            }}
          ],
          "body_mapping": [
            {{
              "organ_id": "brain|heart|liver|lungs|kidneys", 
              "status": "success|warning|danger"
            }}
          ],
          "triage": {{
            "specialist": "string",
            "urgency": "Low|Medium|High"
          }},
          "precautions": ["string", "string", "string"]
        }}

        ### SECTION 2: EDUCATIONAL NARRATIVE
        Wrap this section in [NARRATIVE_START] and [NARRATIVE_END].
        - Use simple, empathetic, plain language.
        - Explain clinical terms using metaphors (e.g., "The liver is like a filter...").
        - DO NOT provide a diagnosis. Focus on explaining what the markers do.
        - Suggest specific questions for their doctor.

        TEXT TO ANALYZE:
        {raw_text[:4000]}
        """

        # 2. Execute LLM Stream
        full_output = ""
        try:
            for token in stream_llm_with_fallback(prompt):
                full_output += token
        except Exception as e:
            print(f"❌ LLM Router Error: {e}")
            return {"structured": None, "narrative": "AI Service currently unavailable."}

        # 3. Parse and Clean the Hybrid Response
        return LabAnalyzer._parse_and_validate(full_output)

    @staticmethod
    def _parse_and_validate(ai_text: str) -> Dict[str, Any]:
        """
        Uses regex to isolate JSON and Narrative blocks, then validates the data integrity.
        """
        try:
            # Extract JSON block
            json_match = re.search(r'\[STRUCTURED_START\](.*?)\[STRUCTURED_END\]', ai_text, re.DOTALL)
            # Extract Narrative block
            narrative_match = re.search(r'\[NARRATIVE_START\](.*?)\[NARRATIVE_END\]', ai_text, re.DOTALL)

            structured_data = None
            narrative_text = "Analysis could not be summarized in plain text."

            if json_match:
                # Remove possible markdown inside the markers
                clean_json = re.sub(r'```json|```', '', json_match.group(1)).strip()
                structured_data = json.loads(clean_json)
                
                # Double-check classification for safety
                for m in structured_data.get("markers", []):
                    # Ensure status is accurate based on Python logic
                    m["status"] = classify_lab_value(m.get("value"), m.get("ref_min"), m.get("ref_max"))

            if narrative_match:
                narrative_text = narrative_match.group(1).strip()
            elif not json_match:
                # Fallback if no tags were followed
                narrative_text = ai_text

            return {
                "structured": structured_data,
                "narrative": narrative_text
            }

        except Exception as e:
            print(f"❌ Parser Error: {e}")
            return {
                "structured": None, 
                "narrative": "The AI encountered an error while formatting your results. Please check the raw markers below."
            }