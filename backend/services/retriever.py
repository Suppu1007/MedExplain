import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from core.config import knowledge_collection

# Load a lightweight clinical embedding model
# This must be the SAME model used to create the embeddings in your DB
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class MongoDBRAGService:
    @staticmethod
    def retrieve_context(finding_text: str, top_k=2):
        """
        Takes the disease name from ResNet, converts to vector, 
        and finds the match in MongoDB.
        """
        # 1. Convert the ResNet finding (string) into a query embedding (vector)
        query_embedding = embedder.encode(finding_text)

        # 2. Fetch all docs from MongoDB (Your provided logic)
        docs = list(knowledge_collection.find({}))
        if not docs:
            return "Knowledge base empty. Please ingest medical textbooks."

        # 3. Calculate Similarity
        vectors = np.array([d["embedding"] for d in docs])
        # Ensure query_embedding is the right shape
        scores = cosine_similarity([query_embedding], vectors)[0]

        # 4. Sort and return top matches
        top_docs = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )[:top_k]

        # Join the content for the "Black Box"
        return "\n\n".join(d["content"] for _, d in top_docs)

# Instance for the router
rag_service = MongoDBRAGService()