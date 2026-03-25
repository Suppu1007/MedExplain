import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from core.config import knowledge_collection

# Load the embedding model (This must match the model used for ingestion)
# We use a global instance to avoid re-loading on every request
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

import uuid
from datetime import datetime
from services.pdf_service import PDFService

class MongoDBRAGService:
    def ingest_document(self, file_bytes: bytes, filename: str) -> dict:
        """
        Processes a raw file (PDF/Image), extracts text, chunks it,
        embeds it, and stores it in MongoDB.
        """
        # 1. Extract Text
        print(f"RAG Ingestion: Extracting text from {filename}...")
        text = PDFService.extract_text(file_bytes, filename)
        if not text or len(text.strip()) < 50:
             # Fallback for very short or empty text
            return {"status": "error", "message": "Could not extract readable text from this document."}

        # 2. Chunking (Simple overlap strategy)
        chunk_size = 1000
        overlap = 200
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 100: # Ignore tiny chunks
                chunks.append(chunk)
        
        print(f"RAG Ingestion: Generated {len(chunks)} chunks. Embedding...")

        # 3. Embedding & Storage
        batch_entries = []
        source_id = str(uuid.uuid4()) # Unique ID for this document upload

        for idx, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()
            
            entry = {
                "source_id": source_id,
                "filename": filename,
                "chunk_index": idx,
                "content": chunk,
                "embedding": embedding,
                "ingested_at": datetime.utcnow()
            }
            batch_entries.append(entry)

        if batch_entries:
            knowledge_collection.insert_many(batch_entries)
        
        return {
            "status": "success", 
            "chunks": len(batch_entries), 
            "source_id": source_id
        }

    def retrieve_context(self, finding_text: str, symptoms: str = None, top_k=2):
        """
        Takes the disease name from ResNet, converts to vector, 
        and finds the match in MongoDB.
        """
        try:
            # 1. Convert the ResNet finding (string) into a query embedding (vector)
            # If symptoms are provided, we append them to refine the semantic search
            query_text = f"{finding_text} {symptoms}" if symptoms else finding_text
            query_embedding = embedder.encode(query_text)

            # 2. Fetch all docs from MongoDB
            # Optimized: Only fetch fields needed for cosine similarity first if possible, 
            # but for now we fetch all as per existing logic (can be optimized for scale later)
            docs = list(knowledge_collection.find({}, {"_id": 0, "embedding": 1, "content": 1, "filename": 1}))
            
            if not docs:
                return "Knowledge base empty. Please ingest medical textbooks into MongoDB."

            # 3. Calculate Similarity
            vectors = np.array([d["embedding"] for d in docs])
            
            # Ensure the query embedding is shaped correctly for sklearn
            scores = cosine_similarity([query_embedding], vectors)[0]

            # 4. Sort and return top matches
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = scores[idx]
                if score > 0.25: # relevancy threshold
                    doc = docs[idx]
                    results.append(f"[Source: {doc.get('filename', 'Textbook')}]\n{doc['content']}")
            
            if not results:
                return f"No specific high-confidence textbook match for {finding_text}."

            return "\n\n".join(results)
            
        except Exception as e:
            print(f"RAG Error: {e}")
            return f"Clinical correlation advised for finding: {finding_text}."

# IMPORTANT: This variable name MUST match the import in your router
rag_service = MongoDBRAGService()


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# Import from the config we just fixed
from core.config import knowledge_collection 

# Load embedding model once
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class MedicalRAG:
    def query_knowledge(self, disease_finding: str) -> str:
        """Searches MongoDB for clinical context."""
        try:
            # 1. Vectorize the finding
            query_vector = embedder.encode(disease_finding)

            # 2. Fetch all docs from MongoDB
            docs = list(knowledge_collection.find({}))
            if not docs:
                return "Clinical guidelines pending library ingestion."

            # 3. Calculate Similarity
            vectors = np.array([d["embedding"] for d in docs])
            scores = cosine_similarity([query_vector], vectors)[0]

            # 4. Get Top Match
            top_idx = int(np.argmax(scores)) 
            
            if scores[top_idx] < 0.25: # Lowered threshold slightly to be more permissive
                return f"No specific textbook match for {disease_finding}. Ensure clinical correlation."
            
            match = docs[top_idx]
            citation = match.get("filename", "Medical Guidelines")
            return f"[Source: {citation}]\n{match['content']}"
        except Exception as e:
            print(f"RAG Query Error: {e}")
            return "Knowledge base currently unavailable."

# IMPORTANT: Export as 'medical_library' to match your lab_analysis/analyzer.py
medical_library = MedicalRAG()