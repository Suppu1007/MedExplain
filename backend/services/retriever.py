from app.core.config import knowledge_collection
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_context(query_embedding, top_k=3):
    docs = list(knowledge_collection.find({}))
    if not docs:
        return ""

    vectors = np.array([d["embedding"] for d in docs])
    scores = cosine_similarity([query_embedding], vectors)[0]

    top_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return "\n".join(d["content"] for _, d in top_docs)
