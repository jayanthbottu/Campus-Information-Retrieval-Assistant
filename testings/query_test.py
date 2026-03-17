import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

def extract_amounts(text):
    return re.findall(r"Rs\.\s?[\d,]+", text)

INDEX_PATH = "index"

print("Loading model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Loading index...")
index = faiss.read_index(f"{INDEX_PATH}/sr_index.faiss")

with open(f"{INDEX_PATH}/metadata.json", "r") as f:
    metadata = json.load(f)

def search(query, k=15):
    def grounded_search(query, k=15):
        query_embedding = model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding)

        scores, indices = index.search(query_embedding, k)

        candidates = [metadata[idx] for idx in indices[0]]

        # Grounding filter
        filtered = []
        for chunk in candidates:
            content = chunk["content"].lower()
            if "fellowship" in content or "stipend" in content:
                filtered.append(chunk)

        # Fallback if nothing found
        if not filtered:
            return candidates[:5]

        return filtered[:5]
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding)

    scores, indices = index.search(query_embedding, k)

    candidates = []
    for idx in indices[0]:
        candidates.append(metadata[idx])

    query_words = query.lower().split()

    def hybrid_score(chunk):
        content = chunk["content"].lower()
        keyword_hits = sum(1 for word in query_words if word in content)

        source_bonus = 2 if "Handbook" in chunk["source_file"] else 0
        fellowship_bonus = 3 if "fellowship" in content else 0

        return keyword_hits + source_bonus + fellowship_bonus

    candidates.sort(key=hybrid_score, reverse=True)

    return candidates[:5]

while True:
    query = input("\nAsk something: ")
    results = search(query)

    for i, r in enumerate(results):
        print(f"\nResult {i+1}")
        print("Source:", r["source_file"])
        print(r["content"][:500])