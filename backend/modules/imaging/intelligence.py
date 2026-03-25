import json
import re
from typing import Dict, Any
from services.llm import stream_vision_response
from services.vision_service import resnet_engine
from services.rag_service import medical_library 
from services.llm import stream_llm_with_fallback

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

        # STEP 3: RAG GROUNDING
        # Search textbooks for guidelines based on what the AI 'saw'
        clinical_context = medical_library.query_knowledge(f"Clinical guidelines for {ai_description}")

        # STEP 4: FINAL XAI SYNTHESIS (High-Precision Clinical Persona)
        prompt = f"""
        ACT AS A SENIOR CLINICAL RADIOLOGIST. 
        Synthesize this CNN ResNet-50 detection: {vision_metrics}
        Using these Clinical Guidelines: {clinical_context}

        Provide a structured explainable output.
        [STRUCTURED_START]
        {{
          "anatomy_target": "brain|heart|lungs|liver|kidneys",
          "triage": {{ "specialist": "string", "urgency": "{vision_metrics['status'].upper()}" }},
          "resnet_metrics": {{ "confidence": {vision_metrics['confidence']}, "layer": "layer4" }},
          "precautions": ["Critical Safety Precaution 1", "Critical Safety Precaution 2"],
          "reasoning": "High-precision clinical rationale"
        }}
        [STRUCTURED_END]

        [NARRATIVE_START]
        Write a professional 5-section clinical report using ONLY HTML tags. 
        Output EXACTLY 5 <p> blocks with <b> headers.
        
        <p><b> Imaging Findings</b><br>Technical summary.</p>
        <p><b> Clinical Significance</b><br>Diagnostic implications.</p>
        <p><b> Specialist Recommendations</b><br>Next clinical steps.</p>
        <p><b> Precautions</b><br>Safety protocols.</p>
        <p><b> Clinical Disclaimer</b><br>AI screening disclaimer.</p>
        [NARRATIVE_END]
        """

        full_output = ""
        # Using Groq for final fast synthesis
        for token in stream_llm_with_fallback(prompt, use_groq=True):
            full_output += token

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
            return {"structured": None, "narrative": ai_text}
