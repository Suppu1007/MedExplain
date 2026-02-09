# MediExplain System Arch
> Multimodal Explainable AI (XAI) Pipeline

This document outlines the technical architecture of MediExplain, categorized into six distinct layers that function as a cohesive intelligent system.

## 1. The Clinical NLP Layer: BioBERT (The "Brain")
*   **Technology**: BioBERT v1.1
*   **Role**:
    *   **Clinical NER**: Identifies biomarkers (e.g., Creatinine, LDL) and pathologies.
    *   **Medical Translation**: Decodes complex nomenclature into patient-friendly language.

## 2. The Computer Vision Layer: ResNet-50 (The "Eyes")
*   **Technology**: CNN ResNet-50
*   **Role**:
    *   **Pathology Classification**: Classifies X-ray and MRI regions as "Normal" or "Abnormal".
    *   **Feature Extraction**: Detects densities, fractures, and organ borders using deep residual connections.

## 3. The Explainability Layer: Grad-CAM & SHAP (The "Why")
*   **Grad-CAM (Visual XAI)**: Overlays heatmaps on medical scans to highlight decision regions.
*   **SHAP (Numerical XAI)**: Quantifies the influence of specific lab values on the overall health urgency score.

## 4. The Knowledge Grounding Layer: RAG (The "Source of Truth")
*   **Technology**: Qdrant Vector DB + BioASQ/MedQuad Datasets
*   **Role**:
    *   **Fact Checking**: Anchors outputs in verified medical guidelines.
    *   **Hallucination Prevention**: Constrains the LLM to retrieved context.

## 5. The Ingestion Engine: EasyOCR
*   **Technology**: EasyOCR (ResNet + LSTM)
*   **Role**: Preserves spatial alignment of tabular data in scanned reports, ensuring accurate key-value pair extraction.

## 6. The Orchestration Layer: FastAPI & Groq (The Infrastructure)
*   **Groq Llama 3.3 (70B)**: The "Linguistic Interface" synthesizing the final narrative.
*   **FastAPI**: The asynchronous nervous system coordinating all services.

---

## Architecture Diagram

```mermaid
graph TD
    %% Nodes
    User([User])
    UI[Frontend Interface<br>Dashboard / Upload]
    
    subgraph Orchestration [Orchestration Layer]
        API[FastAPI Backend]
        Groq[Groq Llama 3.3 Engine]
    end

    subgraph Ingestion [Ingestion Layer]
        OCR[EasyOCR Engine<br>ResNet + LSTM]
    end

    subgraph Processing [Processing Layers]
        BioBERT[Clinical NLP<br>BioBERT v1.1]
        ResNet[Computer Vision<br>ResNet-50]
    end

    subgraph Explainability [Explainability Layer XAI]
        SHAP[SHAP Analysis<br>Feature Importance]
        GradCAM[Grad-CAM<br>Visual Heatmap]
    end

    subgraph Knowledge [Knowledge Layer]
        Qdrant[(Qdrant Vector DB)]
        RAG[RAG Retriever<br>BioASQ / MedQuad]
    end

    %% Flow
    User -->|Upload Report/Scan| UI
    UI -->|HTTPS POST| API

    %% Data Flow
    API -->|Raw PDF/Image| OCR
    OCR -->|Extracted Text| BioBERT
    OCR -->|Image Tensor| ResNet

    %% Intelligence Flow
    BioBERT -->|Entities| SHAP
    ResNet -->|Feature Maps| GradCAM

    %% Context Flow
    BioBERT -->|Query| RAG
    ResNet -->|Findings| RAG
    RAG <-->|Semantic Search| Qdrant
    
    %% Synthesis
    RAG -->|Context + Facts| Groq
    SHAP -->|Numerical Impact| Groq
    GradCAM -->|Visual Evidence| Groq
    
    Groq -->|Final Narrative| API
    API -->|JSON Response| UI

    %% Styling
    style User fill:#333,stroke:#fff,color:#fff
    style UI fill:#eee,stroke:#333
    style API fill:#0d6efd,stroke:#fff,color:#fff
    style Groq fill:#6610f2,stroke:#fff,color:#fff
    style Qdrant fill:#ffc107,stroke:#333
    style BioBERT fill:#198754,stroke:#fff,color:#fff
    style ResNet fill:#198754,stroke:#fff,color:#fff
    style SHAP fill:#fd7e14,stroke:#fff,color:#fff
    style GradCAM fill:#fd7e14,stroke:#fff,color:#fff
```

## Data Flow Summary
1.  **Ingestion**: User uploads a report. EasyOCR digitizes the content, preserving structure.
2.  **Analysis**:
    *   **Text**: BioBERT extracts clinical entities. SHAP calculates their risk contribution.
    *   **Vision**: ResNet-50 detects pathologies. Grad-CAM generates a heatmap of the focus area.
3.  **Grounding**: The system queries Qdrant to retrieve relevant medical literature (RAG).
4.  **Synthesis**: Groq Llama 3 combines the clinical findings, XAI insights, and retrieved context into a patient-friendly narrative.
5.  **Delivery**: FastAPI delivers the structured JSON and narrative to the frontend.
