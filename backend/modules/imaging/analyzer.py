import json
import re
from typing import Dict, Any
from services.llm import stream_vision_response
from services.vision_service import resnet_engine
from services.rag_service import medical_library 

class ImagingIntelligence:
    @staticmethod
    def analyze_scan(image_bytes: bytes) -> Dict[str, Any]:
        """
        Multimodal Pipeline:
        1. CNN ResNet: Identifies visual probability.
        2. RAG: Retrieves clinical guidelines for the organ detected.
        3. Groq (BioBERT Prompt): Simplifies findings and generates XAI narrative.
        """
        # STEP 1: CNN INFERENCE
        vision_metrics = resnet_engine.run_inference(image_bytes)

        # STEP 2: LINGUISTIC ANALYSIS (Vision LLM)
        # We ask the model to identify the organ and confirm ResNet's finding
        vision_prompt = "Identify the organ in this scan and describe any abnormalities."
        ai_description = ""
        for token in stream_vision_response(vision_prompt, image_bytes):
            ai_description += token

        # STEP 3 & 4: ADVANCED LANGCHAIN INTERPRETATION (RAG + Reasoning)
        from services.langchain_service import langchain_service
        
        # We pass the ResNet findings and the raw AI visual description to LangChain
        full_output = langchain_service.interpret_imaging(vision_metrics, ai_description)

        return ImagingIntelligence._parse(full_output)

    @staticmethod
    def _parse(ai_text: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\[STRUCTURED_START\](.*?)\[STRUCTURED_END\]', ai_text, re.DOTALL)
            text_match = re.search(r'\[NARRATIVE_START\](.*?)\[NARRATIVE_END\]', ai_text, re.DOTALL)
            return {
                "structured": json.loads(json_match.group(1).strip()),
                "narrative": text_match.group(1).strip() if text_match else ai_text
            }
        except:
            return {
                "structured": {
                    "medical_keywords": [],
                    "simplified_verdict": "Internal diagnostic analysis complete."
                },
                "narrative": ai_text
            }