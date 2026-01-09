import os
import json
import csv
from sentence_transformers import SentenceTransformer
from xml.etree import ElementTree as ET

# ===============================
# CONTAINER-SAFE PATHS
# ===============================
BASE_DATA_PATH = "/app/data"
BIOASQ_PATH = os.path.join(BASE_DATA_PATH, "bioasq")
MEDQUAD_PATH = os.path.join(BASE_DATA_PATH, "medquad", "medDataset_processed.csv")
OUTPUT_PATH = os.path.join(BASE_DATA_PATH, "embeddings.json")

EMBED_MODEL = "all-MiniLM-L6-v2"


# ===============================
# BioASQ XML Loader
# ===============================
def load_bioasq():
    docs = []
    idx = 0

    for root_dir, _, files in os.walk(BIOASQ_PATH):
        for file in files:
            if not file.lower().endswith(".xml"):
                continue

            try:
                tree = ET.parse(os.path.join(root_dir, file))
                root = tree.getroot()

                for qa in root.iter("QAPair"):
                    q = qa.findtext("Question")
                    a = qa.findtext("Answer")

                    if q and a:
                        docs.append({
                            "id": f"bioasq_{idx}",
                            "text": f"Q: {q}\nA: {a}",
                            "source": "BioASQ"
                        })
                        idx += 1
            except Exception:
                continue

    print(f"✅ BioASQ loaded: {len(docs)}")
    return docs


# ===============================
# MedQuAD CSV Loader
# ===============================
def load_medquad():
    docs = []
    idx = 0

    print("📄 MedQuAD path:", MEDQUAD_PATH)
    print("📄 Exists:", os.path.exists(MEDQUAD_PATH))

    with open(MEDQUAD_PATH, encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = list(reader)

    headers = [h.lower() for h in rows[0]]
    print("📄 Headers:", headers)

    q_idx = headers.index("question")
    a_idx = headers.index("answer")

    for row in rows[1:]:
        q = row[q_idx].strip()
        a = row[a_idx].strip()
        if q and a:
            docs.append({
                "id": f"medquad_{idx}",
                "text": f"Q: {q}\nA: {a}",
                "source": "MedQuAD"
            })
            idx += 1

    print(f"✅ MedQuAD loaded: {len(docs)}")
    return docs


# ===============================
# MAIN
# ===============================
def main():
    documents = []
    documents.extend(load_bioasq())
    documents.extend(load_medquad())

    print(f"📚 Total documents: {len(documents)}")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        [d["text"] for d in documents],
        show_progress_bar=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "documents": documents,
            "embeddings": embeddings.tolist()
        }, f)

    print(f"✅ embeddings.json created at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
