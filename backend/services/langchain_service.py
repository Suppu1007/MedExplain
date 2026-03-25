import os
import json
from typing import Dict, Any, Generator
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from services.rag_service import medical_library

class LangChainService:
    def __init__(self):
        # Configuration matches services/llm.py
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

        if self.groq_api_key:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=self.groq_api_key,
                temperature=0.2
            )
        else:
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.ollama_url,
                temperature=0.2
            )

    def interpret_imaging(self, vision_metrics: Dict[str, Any], ai_description: str) -> Dict[str, Any]:
        """
        Uses LangChain to correlate vision findings with medical RAG knowledge.
        """
        # 1. RAG Retrieval
        context = medical_library.query_knowledge(f"Clinical guidelines for {ai_description}")

        # 2. Prompt Construction
        prompt = ChatPromptTemplate.from_template("""
        ACT AS A SENIOR CLINICAL DIAGNOSTICIAN.
        Your goal is to provide a patient-friendly but medically rigorous interpretation of an imaging scan.

        VISION ANALYSIS FINDINGS:
        - Organ: {anatomy}
        - AI Detection: {description}
        - Confidence: {confidence}
        - Status: {status}

        CLINICAL REFERENCE DATA:
        {context}

        INSTRUCTIONS:
        1. STRUCTURED DATA: Provide JSON only between [STRUCTURED_START] and [STRUCTURED_END].
        {{
          "anatomy_target": "{anatomy}",
          "triage": {{ "specialist": "string", "urgency": "{status}" }},
          "recommendations": ["list 2 actionable steps"],
          "reasoning": "1-sentence clinical rationale",
          "medical_keywords": ["#TAG1", "#TAG2"],
          "simplified_verdict": "Clear, non-technical English summary of findings"
        }}
        
        2. NARRATIVE: Provide a **Professional Clinical Narrative** between [NARRATIVE_START] and [NARRATIVE_END].
        - Structure:
          1. **Clinical Assessment**:
             - **Primary System Affected**: (e.g. Respiratory, Musculoskeletal)
             - **Severity Score**: Estimate a severity (1-10) for the patient.
          2. **Imaging Breakdown**: Professional yet accessible description of findings. Include laterality and size if applicable.
          3. **Lifestyle & Clinical Correlation**: 
             - **Risk Factors**: (e.g. "Long-term smoking may correlate with...")
             - **Environmental Link**: (e.g. "Occupational exposure...")
          4. **Action Plan**: Evidence-based next steps. Use bold labels (e.g. <b>Action</b>: ...).
          5. **Questions for Your Doctor**: 3 smart questions for their next visit. Use bold labels (e.g. <b>Question 1</b>: ...).
          
        - Tone: Clinical, authoritative, direct. NO metaphors.
        - Format: Clear paragraphs with bold <b> headers. Use <b> tags for all styling.
        - **STRICT FORMATTING**: DO NOT use asterisk bullets (*) or Markdown lists.
        - Educational only, no final diagnosis.
        """)

        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "anatomy": vision_metrics.get("organ", "unknown"),
            "description": ai_description,
            "confidence": vision_metrics.get("confidence", 0),
            "status": vision_metrics.get("status", "unknown"),
            "context": context
        })

        return response

    def interpret_labs(self, markers: list, triage: dict) -> str:
        """
        Uses LangChain for deep biomarker analysis and RAG correlation.
        """
        if not markers:
            return "<b>Clinical Assessment</b><br>The primary system affected cannot be determined due to the absence of laboratory results. Please upload a valid lab report to generate a detailed correlation.<br><br><b>Action Plan</b><br><b>Action</b>: Upload a digital or scanned lab report (PDF/JPG) for analysis."

        # 1. RAG Retrieval for the most abnormal markers
        abnormal_markers = [m['name'] for m in markers if m.get('status') != 'normal']
        query_text = "Clinical significance of " + ", ".join(abnormal_markers[:3])
        context = medical_library.query_knowledge(query_text)

        prompt = ChatPromptTemplate.from_template("""
        ACT AS A SENIOR CLINICAL PATHOLOGIST. 
        Your goal is to provide a medically rigorous, highly accurate interpretation of the following laboratory results.

        RESULTS TO ANALYZE:
        {markers}
        
        TRIAGE: {triage}

        CLINICAL REFERENCE DATA:
        {context}

        Write a professional 4-section clinical report using ONLY HTML tags.
        Output EXACTLY 4 <p> blocks with <b> headers. No other text before or after.

        <p><b> Clinical Assessment</b><br>
        Identify the primary physiological system affected (e.g., Renal, Hepatic, Hematological).
        Summarize the overall clinical impression of the biomarkers. 2-3 sentences.</p>

        <p><b> Clinical Significance</b><br>
        Interpret the abnormal findings in a data-driven clinical context. 
        Explain WHY these values are concerning by referencing clinical thresholds. 2-3 sentences.</p>

        <p><b> Recommended Actions</b><br>
        Provide 2-3 specific, evidence-based clinical next steps (e.g., specialist consultation, specific follow-up tests, or monitoring). Use bold labels for each action (e.g., <b>Action</b>: ...).</p>

        <p><b> Disclaimer</b><br>
        This evaluation was generated by an AI screening system. It is not a definitive medical diagnosis. 
        Clinical correlation by a board-certified physician is required to confirm all findings.</p>

        STRICT CLINICAL REPORT RULES:
        1. Return EXACTLY 4 <p> blocks with <b> headers.
        2. Tone: Senior Pathologist (Formal, precise, objective).
        3. No non-clinical descriptions or metaphors.
        4. NO Markdown syntax (no **, no ##, no ```).
        5. Professional yet compassionate tone.
        """)

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "markers": json.dumps(markers),
            "triage": json.dumps(triage),
            "context": context
        })

langchain_service = LangChainService()
