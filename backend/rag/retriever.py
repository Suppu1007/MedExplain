import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# CONFIG
# ===============================
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "/app/data/embeddings.json"

_model = None
_documents = None
_embeddings = None


def _load_store():
    global _model, _documents, _embeddings

    if _documents is not None:
        return

    if not os.path.exists(EMBEDDINGS_PATH):
        raise RuntimeError(
            "❌ embeddings.json not found. "
            "Run: python -m app.rag.ingest"
        )

    with open(EMBEDDINGS_PATH, encoding="utf-8") as f:
        store = json.load(f)

    _documents = store["documents"]
    _embeddings = np.array(store["embeddings"])
    _model = SentenceTransformer(EMBED_MODEL)

    print("✅ RAG store loaded")


# ===============================
# RETRIEVER
# ===============================
def retrieve_context(query: str, top_k: int = 3):
    _load_store()

    query_vec = _model.encode([query])
    scores = cosine_similarity(query_vec, _embeddings)[0]

    top_indices = scores.argsort()[-top_k:][::-1]

    contexts = []
    sources = set()

    for idx in top_indices:
        contexts.append(_documents[idx]["text"])
        sources.add(_documents[idx]["source"])

    return "\n\n".join(contexts), list(sources)
