import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import re

# ============================================================
# CONFIG
# ============================================================

INDEX_PATH = "index"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ============================================================
# DOMAIN TAXONOMY (same as builder)
# ============================================================

DOMAINS = {
    "overview": [
    "sr university",
    "sru",
    "overview",
    "about",
    "vision",
    "mission",
    "profile",
    "campus life"
],
    "sports": ["cricket","football","basketball","volleyball",
               "badminton","athletics","kabaddi","tennis","sports","games"],
    "faculty": ["professor","assistant professor","associate professor",
                "hod","dean","head of department"],
    "departments": ["department","school of","cse","ece",
                    "mechanical","civil","it"],
    "facilities": ["library","laboratory","lab","hostel",
                   "transport","wifi","auditorium","canteen"],
    "admissions": ["admission","eligibility","entrance",
                   "application","registration"],
    "fees": ["fee","tuition","payment","scholarship"],
    "policies": ["policy","rule","regulation","guideline",
                 "promotion","attendance","exam","invigilator"],
    "clubs": ["club","society","association","technical club"]
}

ENUMERATION_KEYWORDS = [
    "list","which","what are","available","games","sports","facilities"
]

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(EMBED_MODEL_NAME)

# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(query):
    query_lower = query.lower()
    for word in ENUMERATION_KEYWORDS:
        if word in query_lower:
            return "ENUMERATION"
    return "DESCRIPTIVE"

# ============================================================
# HARD DOMAIN ROUTING
# ============================================================

def detect_domain(query):
    query_lower = query.lower()
    scores = {}

    for domain, keywords in DOMAINS.items():
        score = sum(query_lower.count(k) for k in keywords)
        scores[domain] = score

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return None

    return best

# ============================================================
# ENUMERATION HANDLER
# ============================================================

def handle_enumeration(domain):
    meta_path = os.path.join(INDEX_PATH, f"{domain}_meta.json")

    if not os.path.exists(meta_path):
        return "No data available."

    with open(meta_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    all_items = []

    for entry in entries:
        if entry.get("contains_enumeration"):
            all_items.extend(entry.get("enumeration_items", []))

    unique_items = sorted(list(set(all_items)))

    if not unique_items:
        return "No structured list available."

    return "\n".join(unique_items)

# ============================================================
# DESCRIPTIVE HANDLER
# ============================================================

def handle_descriptive(query, domain, top_k=5):
    index_path = os.path.join(INDEX_PATH, f"{domain}.faiss")
    meta_path = os.path.join(INDEX_PATH, f"{domain}_meta.json")

    if not os.path.exists(index_path):
        return "No data available."

    index = faiss.read_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        if idx < len(entries):
            results.append(entries[idx]["content"])

    if not results:
        return "No relevant information found."

    return "\n\n".join(results)

# ============================================================
# MAIN QUERY FUNCTION
# ============================================================

def query_system(query):
    intent = detect_intent(query)
    domain = detect_domain(query)

    if not domain:
        return "Query outside institutional scope."

    if intent == "ENUMERATION":
        return handle_enumeration(domain)
    else:
        return handle_descriptive(query, domain)

# ============================================================
# INTERACTIVE LOOP
# ============================================================

if __name__ == "__main__":
    print("SRU Institutional Query Engine")
    print("Type 'exit' to quit\n")

    while True:
        user_query = input("Ask: ")

        if user_query.lower() == "exit":
            break

        response = query_system(user_query)
        print("\nResult:\n")
        print(response)
        print("\n" + "-"*50 + "\n")