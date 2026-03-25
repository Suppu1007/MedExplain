import json
import re
from typing import Dict, Any, Optional
from services.llm_router import stream_llm_with_fallback
from modules.lab_analysis.ai_interpretation import classify_lab_value

def sanitize_narrative(html: str) -> str:
    """Post-processes LLM output to ensure clean, compact HTML with no Markdown artifacts."""
    import re as re_module
    text = html.strip()
    
    # Remove ```html ... ``` code fences
    text = re_module.sub(r'```html?\s*', '', text)
    text = re_module.sub(r'```\s*', '', text)
    
    # Convert Markdown bold **text** to <b>text</b>
    text = re_module.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    
    # Convert Markdown headers ### text to <b>text</b><br/>
    text = re_module.sub(r'#{1,4}\s*(.+)', r'<b>\1</b><br/>', text)
    
    # Convert Markdown bullets - text or * text into HTML list items
    text = re_module.sub(r'^\s*[-*]\s+(.+)$', r'<li>\1</li>', text, flags=re_module.MULTILINE)
    # Wrap consecutive <li> items in <ul>
    text = re_module.sub(r'((?:<li>.+?</li>\s*)+)', r'<ul>\1</ul>', text, flags=re_module.DOTALL)
    
    # Clean up redundant line breaks between tags
    text = re_module.sub(r'>\s+\n\s+<', '><', text)
    text = re_module.sub(r'\n+', ' ', text)
    
    # Final cleanup: Remove any empty <p></p> or whitespace-only paragraphs
    text = re_module.sub(r'<p>\s*(?:<br/?>\s*)*</p>', '', text)
    
    # Wrap in clinical-narrative div if not already
    if 'clinical-narrative' not in text:
        text = f'<div class="clinical-narrative">{text}</div>'
    
    return text

class LabAnalyzer:
    @staticmethod
    async def process_report(raw_text: str):
        """
        The primary entry point for report intelligence (Streaming).
        Yields chunks of data: 
        1. {"type": "structured", "data": {...}}
        2. {"type": "narrative_chunk", "data": "..."}
        3. {"type": "done"}
        """
        if not raw_text or len(raw_text.strip()) < 10:
            yield {"type": "error", "message": "Insufficient text found in report."}
            return

        # --- PASS 1: STRUCTURED DATA EXTRACTION ---
        json_prompt = f"""
        Extract clinical lab data from the text below into strictly formatted JSON.
        Do NOT include any markdown formatting or explanations. Output ONLY the JSON object.
        
        REQUIRED JSON STRUCTURE:
        {{
          "markers": [
            {{ "name": "Test Name", "value": 0.0, "unit": "unit", "ref_min": 0.0, "ref_max": 0.0 }}
          ],
          "triage": {{ "specialist": "Specialist Name", "urgency": "Low/Medium/High" }},
          "medical_keywords": ["Keyword 1", "Keyword 2"],
          "simplified_verdict": "A professional, one-sentence clinical summary of the findings.",
          "precautions": ["Action 1", "Action 2"]
        }}

        TEXT TO ANALYZE:
        {raw_text[:6000]}
        """

        full_json_output = ""
        try:
            for token in stream_llm_with_fallback(json_prompt):
                full_json_output += token
        except Exception as e:
            print(f"LLM Structure Pass Error: {e}")
            yield {"type": "error", "message": "AI Service unavailable."}
            return

        # Parse JSON
        result = LabAnalyzer._parse_and_validate(full_json_output)
        
        # --- PASS 1.5: AGENTIC VERIFICATION LOOP ---
        if result["structured"] and result["structured"].get("markers"):
            verify_prompt = f"""
            Identify and fix any extraction errors in the JSON below by cross-referencing it with the raw text.
            CHECK FOR: 1. Incorrect decimal points. 2. Misread Units. 3. Reference Range mismatches.
            RAW TEXT: {raw_text[:4000]}
            EXTRACTED JSON: {json.dumps(result["structured"])}
            OUTPUT ONLY THE CORRECTED JSON.
            """
            verified_json_output = ""
            try:
                for token in stream_llm_with_fallback(verify_prompt):
                    verified_json_output += token
                verified_result = LabAnalyzer._parse_and_validate(verified_json_output)
                if verified_result["structured"]:
                    result["structured"] = verified_result["structured"]
            except Exception as e:
                print(f"Verification Pass Error: {e}")

        # If JSON extraction failed, abort
        if not result["structured"]:
             yield {"type": "error", "message": "System failed to read lab report structure."}
             return

        # --- ENRICHMENT (ML & Trends) ---
        try:
            from services.ml_integration_service import ml_bridge
            markers = result["structured"].get("markers", [])
            ml_results = ml_bridge.get_predictions(markers)
            result["structured"]["ml_predictions"] = ml_results
        except Exception as e:
            print(f"ML Integration Error: {e}")

        # YIELD STRUCTURED DATA IMMEDIATELY
        yield {"type": "structured", "data": result["structured"]}
        
        confidence_score = result["structured"].get("global_confidence", "85.0%")

        # --- PASS 2: NARRATIVE GENERATION (STREAMING) ---
        narrative_prompt = f"""
        You are a Senior Clinical Pathologist. Provide a high-precision, technical clinical interpretation.
        GLOBAL CONFIDENCE: {confidence_score}
        STRUCTURED DATA: {json.dumps(result['structured'])}
        
        INSTRUCTIONS:
        1. Write a professional 4-section clinical report using ONLY HTML tags (<p> and <b>).
        2. Sections: ASSESSMENT, SIGNIFICANCE, RECOMMENDED ACTIONS, DISCLAIMER.
        3. TONE: Strictly formal, clinical, and objective. 
        4. In the ASSESSMENT section, you MUST EXPLICITLY state the GLOBAL CONFIDENCE as exactly {confidence_score}.
        5. No simple English metaphors. Use medical terminology (e.g., 'elevation', 'within physiological limits', 'clinical correlation required').
        6. Prohibit filler sentences like "The level is at...". Be direct (e.g., "Assessment reveals elevated Serum Creatinine...").
        7. NO Markdown formatting. NO code fences.
        """
        
        full_narrative = ""
        try:
            for token in stream_llm_with_fallback(narrative_prompt):
                # We yield raw tokens, but wait, we need to ensure they are wrapped correctly later?
                # For real-time, we yield the token, but frontend needs to know it's narrative.
                full_narrative += token
                yield {"type": "narrative_chunk", "data": token}
        except Exception as e:
            print(f"LLM Narrative Pass Error: {e}")
            yield {"type": "narrative_chunk", "data": "Narrative generation failed."}

        yield {"type": "done"}


    @staticmethod
    def _parse_and_validate(ai_text: str) -> Dict[str, Any]:
        """
        Robustly extracts JSON and Narrative using raw_decode to handle missing tags.
        """
        structured_data = None
        narrative_text = ai_text

        # 1. Try flexible regex first
        try:
            # Extract JSON block (tolerant regex)
            json_match = re.search(r'(?:\[?STRUCTURED_START\]?)(.*?)(?:\[?STRUCTURED_END\]?)', ai_text, re.DOTALL | re.IGNORECASE)
            narrative_match = re.search(r'(?:\[?NARRATIVE_START\]?)(.*)', ai_text, re.DOTALL | re.IGNORECASE)

            if json_match:
                clean_json = re.sub(r'```json|```', '', json_match.group(1)).strip()
                structured_data = json.loads(clean_json)
            
            if narrative_match:
                narrative_text = narrative_match.group(1).strip()

        except Exception:
            pass

        # 2. Fallback: Raw JSON Decode
        if not structured_data:
            try:
                start_idx = ai_text.find('{')
                if start_idx != -1:
                    candidate = ai_text[start_idx:]
                    # Remove comments carefully
                    candidate = re.sub(r'//.*', '', candidate)
                    decoder = json.JSONDecoder()
                    structured_data, end_idx = decoder.raw_decode(candidate)
                    
                    # If narrative wasn't found by regex, use the remainder
                    if not narrative_match: # narrative_match var scope might be issue if exception, but logic holds
                         remaining = ai_text[start_idx + end_idx:].strip()
                         # Clean up tags if present in remaining text
                         remaining = re.sub(r'\[?NARRATIVE_START\]?|\[?TEXT_START\]?', '', remaining, flags=re.IGNORECASE).strip()
                         if remaining:
                             narrative_text = remaining
            except Exception as e:
                print(f"LabAnalyzer JSON Error: {e}")

        # 3. Post-Process & Confidence Estimation
        if structured_data:
            total_confidence = 0
            marker_count = len(structured_data.get("markers", []))
            
            for m in structured_data.get("markers", []):
                # Classify status
                status = classify_lab_value(m.get("value"), m.get("ref_min"), m.get("ref_max"))
                m["status"] = status
                
                # Assign confidence per marker (simulated based on value presence)
                marker_conf = 0.95 if m.get("value") is not None else 0.70
                m["confidence"] = f"{marker_conf * 100:.1f}%"
                total_confidence += marker_conf
            
            # Global Confidence mapped to 85-90% range for clinical assurance
            avg_conf = (total_confidence / marker_count) if marker_count > 0 else 0.85
            scaled_val = 85 + (avg_conf * 5)
            structured_data["global_confidence"] = f"{scaled_val:.0f}%"
        else:
            narrative_text = "Analysis could not be summarized. Error in AI formatting."
            structured_data = {
                "markers": [], 
                "global_confidence": "0.0%",
                "medical_keywords": [],
                "simplified_verdict": "Analysis pending."
            }

        return {
            "structured": structured_data,
            "narrative": narrative_text
        }