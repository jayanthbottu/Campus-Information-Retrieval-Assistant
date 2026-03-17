import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "clean_faculty.json"
INDEX_FILE = "faculty_faiss.index"
METADATA_FILE = "faculty_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"
HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 64

# ---------------------------------------------------------
# LOAD MODEL (ONLY ONCE)
# ---------------------------------------------------------

model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------
# BUILD INDEX IF NOT EXISTS
# ---------------------------------------------------------

if not os.path.exists(INDEX_FILE):

    print("Building new FAISS HNSW index...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        faculty_data = json.load(f)

    documents = []
    metadata_store = []

    for faculty in faculty_data:
        name = faculty["name"]
        designation = faculty.get("designation", "") or ""
        department = faculty.get("department", "") or ""
        school = faculty.get("school", "") or ""
        research_areas = faculty.get("research_areas", [])

        research_text = ", ".join(research_areas)

        semantic_text = (
            f"{name} is a {designation} in the Department of {department}. "
            f"School: {school}. "
            f"Research areas include {research_text}."
        )

        documents.append(semantic_text)

        metadata_store.append({
            "id": faculty["id"],
            "name": name,
            "designation": designation,
            "department": department,
            "school": school,
            "profile_link": faculty.get("profile_link"),
            "image_link": faculty.get("image_link")
        })

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dimension, HNSW_M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH

    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2, ensure_ascii=False)

    print("Index built and saved successfully.")

else:
    print("Existing index found. Loading...")

# ---------------------------------------------------------
# LOAD INDEX
# ---------------------------------------------------------

index = faiss.read_index(INDEX_FILE)

# Always reconfigure efSearch after loading
if hasattr(index, "hnsw"):
    index.hnsw.efSearch = EF_SEARCH

# ---------------------------------------------------------
# LOAD METADATA
# ---------------------------------------------------------

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Index loaded and configured.")

# ---------------------------------------------------------
# SEARCH FUNCTION
# ---------------------------------------------------------

def search_faculty(query, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        faculty = metadata[idx].copy()
        faculty["score"] = float(score)
        results.append(faculty)

    return results


# ---------------------------------------------------------
# EXAMPLE QUERY
# ---------------------------------------------------------

if __name__ == "__main__":
    results = search_faculty("faculty working on electric vehicles")

    for r in results:
        print(r)