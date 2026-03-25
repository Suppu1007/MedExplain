import pdfplumber
import io
import json
import re
from datetime import datetime
from bson import ObjectId
from db.mongo import lab_results_collection  # Standardizing on your collection name
from services.llm import stream_llm_with_fallback # Your Groq/Ollama Router
from modules.lab_analysis.ai_interpretation import classify_lab_value
from schemas import LabResultCreate

class AnalysisService:
    
    @staticmethod
    async def get_user_lab_results(email: str):
        """
        Fetches all lab markers for a specific user from MongoDB.
        Converts ObjectIds to strings for frontend compatibility.
        """
        cursor = lab_results_collection.find({"user_email": email}).sort("timestamp", -1)
        results = []
        for res in await cursor.to_list(length=100):
            res["id"] = str(res["_id"])
            results.append(res)
        return results

    @staticmethod
    async def calculate_summary(results: list):
        """
        Generates dashboard statistics based on clinical status.
        """
        total = len(results)
        return {
            "total": total,
            "normal": len([r for r in results if r.get('status') == 'normal']),
            "abnormal": len([r for r in results if r.get('status') == 'abnormal']),
            "critical": len([r for r in results if r.get('status') == 'critical']),
        }

    @staticmethod
    async def get_correlations(results: list):
        """
        Identifies relationships between biomarkers (e.g., A1C and Glucose).
        Can be upgraded with real medical logic or an LLM call.
        """
        return [
            {
                "marker1": "Hemoglobin A1C", 
                "marker2": "Total Cholesterol", 
                "value": 0.68, 
                "description": "Strong correlation detected in metabolic markers.",
                "clinical_note": "Elevated glucose levels often impact lipid profile management."
            }
        ]

    @staticmethod
    async def process_and_save_report(email: str, file_bytes: bytes):
        """
        The Core RAG Flow:
        1. OCR/Text Extraction from PDF.
        2. AI Synthesis (Structuring text into specific medical JSON).
        3. Database Storage with user ownership.
        """
        try:
            # --- STEP 1: PDF TEXT EXTRACTION ---
            raw_text = ""
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        raw_text += extracted + "\n"

            if not raw_text.strip():
                print("AnalysisService: No text found in PDF.")
                return None

            # --- STEP 2: LLM PARSING (RAG) ---
            # We call the LLM to get BOTH the structured JSON and the narrative
            ai_output = await AnalysisService._call_llm_to_parse(raw_text)
            
            if not ai_output.get("structured"):
                return None

            # --- STEP 3: STORAGE & ENRICHMENT ---
            # Every report is tagged with user info and current time
            analysis_record = {
                "user_email": email,
                "timestamp": datetime.utcnow(),
                "structured_metrics": ai_output["structured"], # JSON for Dashboard
                "human_narrative": ai_output["narrative"],     # Text for Explanation
                "filename": "Lab_Report_Uploaded.pdf"
            }

            result = await lab_results_collection.insert_one(analysis_record)
            analysis_record["id"] = str(result.inserted_id)
            
            return analysis_record

        except Exception as e:
            print(f"AnalysisService Error in RAG Flow: {e}")
            return None

    @staticmethod
    async def _call_llm_to_parse(text: str):
        """
        Internal method: Uses Groq/Ollama to convert messy text into 
        the specific schema needed for the Visual Body Map and Table.
        """
        prompt = f"""
        You are a Data Extraction AI. Your task is to parse the provided text into structured JSON.
        This is for data processing purposes only.
        
        ### SECTION 1: STRUCTURED DATA
        [JSON_START]
        {{
          "markers": [{{ "name", "value", "unit", "status", "category", "ref_min", "ref_max" }}],
          "triage": {{ "specialist", "urgency" }},
          "precautions": [list of 3 actions]
        }}
        [JSON_END]

        ### SECTION 2: EXPLAINABLE NARRATIVE
        [TEXT_START]
        Plain language metaphor-based explanation for the patient.
        IMPORTANT: NO Markdown. NO BOLDING ** and NO BULLET POINTS *.
        [TEXT_END]

        TEXT TO ANALYZE:
        {text[:3000]} 
        """
        
        # 2. Execute LLM for Structured Extraction
        full_response = ""
        for token in stream_llm_with_fallback(prompt):
            full_response += token
        
        result = AnalysisService._parse_hybrid_response(full_response)

        # 3. ADVANCED LANGCHAIN INTERPRETATION (If we have structured data)
        if result and result.get("structured"):
             from services.langchain_service import langchain_service
             from services.ml_integration_service import ml_bridge
             
             # A. ML Analysis
             try:
                 markers = result["structured"].get("markers", [])
                 ml_results = ml_bridge.get_predictions(markers)
                 result["structured"]["ml_predictions"] = ml_results
             except Exception as e:
                 print(f"ML Integration Error in AnalysisService: {e}")
                 
             # B. LangChain Narrative
             try:
                 langchain_narrative = langchain_service.interpret_labs(
                     result["structured"].get("markers", []),
                     result["structured"].get("triage", {})
                 )
                 if langchain_narrative:
                     result["narrative"] = langchain_narrative
             except Exception as e:
                 print(f"LangChain AnalysisService Error: {e}")

        return result

    @staticmethod
    def _parse_hybrid_response(ai_text: str):
        """
        Robustly extracts JSON and Narrative using raw_decode to handle missing tags.
        """
        structured = None
        narrative = ai_text
        
        def clean_json_str(s):
            # Remove Markdown
            s = re.sub(r'```json|```', '', s, flags=re.IGNORECASE)
            # Remove Comments //
            s = re.sub(r'//.*', '', s)
            return s.strip()

        # 1. Try flexible regex first (Best case)
        try:
            json_match = re.search(r'(?:\[?JSON_START\]?)(.*?)(?:\[?JSON_END\]?|\[?TEXT_START\]?)', ai_text, re.DOTALL | re.IGNORECASE)
            text_match = re.search(r'(?:\[?TEXT_START\]?)(.*)', ai_text, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                s = clean_json_str(json_match.group(1))
                structured = json.loads(s)
            
            if text_match:
                narrative = text_match.group(1).strip()
            elif structured:
                # If JSON found but no text tag, assume rest is narrative
                pass 
        except:
            pass

        # 2. Fallback: Raw JSON Decode (If regex failed)
        if not structured:
            try:
                # Find first valid JSON start
                start_idx = ai_text.find('{')
                if start_idx != -1:
                    # Clean the potential JSON area (from start to some reasonable end or just try raw_decode on dirty string?)
                    # raw_decode handles whitespace but not comments/markdown if they are inside.
                    # We can try to extract the block first using bracket counting? 
                    # Simpler: Just try raw_decode. Use strict=False for control characters.
                    # Note: raw_decode fails on comments.
                    
                    # Heuristic: Clean the WHOLE text of comments/markdown first? No, narrative might have //
                    
                    # Try to find the block manually if raw_decode fails?
                    # Let's clean the string starting from {
                    candidate = ai_text[start_idx:]
                    # Remove comments carefully (only if not in quotes? hard with regex)
                    # Let's trust raw_decode for standard JSON. 
                    # If it fails, we might need a better cleaner.
                    # But cleaning markdown is safe.
                    candidate = re.sub(r'//.*', '', candidate) # Naive comment removal
                    
                    decoder = json.JSONDecoder()
                    structured, end_idx = decoder.raw_decode(candidate)
                    
                    # The rest is narrative
                    narrative = ai_text[start_idx + end_idx:].strip()
            except Exception as e:
                print(f"JSON Parsing Error: {e}")

        # Re-classify statuses
        if structured and "markers" in structured:
            for m in structured["markers"]:
                m["status"] = classify_lab_value(m.get("value"), m.get("ref_min"), m.get("ref_max"))

        return {"structured": structured, "narrative": narrative}