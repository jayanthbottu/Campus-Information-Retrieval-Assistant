import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging

# ============================================================
# CONFIG
# ============================================================

KNOWLEDGE_BASE_PATH = "knowledge_base"
INDEX_PATH = "index"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

os.makedirs(INDEX_PATH, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

logging.info("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL_NAME)

# ============================================================
# BUILD DOMAIN INDEXES
# ============================================================

def build_indexes():
    domain_files = [f for f in os.listdir(KNOWLEDGE_BASE_PATH) if f.endswith(".json")]

    for file in domain_files:
        domain = file.replace(".json", "")
        logging.info(f"Building index for domain: {domain}")

        with open(os.path.join(KNOWLEDGE_BASE_PATH, file), "r", encoding="utf-8") as f:
            entries = json.load(f)

        if not entries:
            logging.info(f"Skipping {domain} (no entries)")
            continue

        contents = [entry["content"] for entry in entries]

        logging.info(f"Embedding {len(contents)} chunks for {domain}")
        embeddings = model.encode(
            contents,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # Save FAISS index
        faiss_path = os.path.join(INDEX_PATH, f"{domain}.faiss")
        faiss.write_index(index, faiss_path)

        # Save metadata mapping
        meta_path = os.path.join(INDEX_PATH, f"{domain}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

        logging.info(f"Saved {domain} index with {len(contents)} vectors")

    logging.info("All domain indexes built successfully.")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_indexes()