import os
import re
import json
import uuid
import logging
from collections import defaultdict

import fitz
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "dataset"
KNOWLEDGE_BASE_PATH = "knowledge_base"
SLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

USE_SLM_FALLBACK = True
MAX_CHUNK_WORDS = 450
ENUMERATION_CONF_THRESHOLD = 0.5

os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ============================================================
# DOMAIN TAXONOMY
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

TRIGGERS = ["include","includes","such as","provides","are","consists of"]

# ============================================================
# LOAD SLM (Fallback Only)
# ============================================================

tokenizer = None
model = None

if USE_SLM_FALLBACK:
    logging.info("Loading SLM for fallback...")
    tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        SLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pages(path):
    doc = fitz.open(path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text.strip())
    return pages

# ============================================================
# SEMANTIC CHUNKING
# ============================================================

def chunk_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    temp = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count += len(words)
        temp.append(sentence)

        if word_count >= MAX_CHUNK_WORDS:
            chunks.append(" ".join(temp))
            temp = []
            word_count = 0

    if temp:
        chunks.append(" ".join(temp))

    return chunks

# ============================================================
# DOMAIN CLASSIFICATION
# ============================================================

DOMAIN_WEIGHTS = {
    "sports": 3,
    "faculty": 2,
    "departments": 0.5,
    "facilities": 2,
    "admissions": 2,
    "fees": 2,
    "policies": 2,
    "clubs": 2
}

def classify_domain(content):
    content_lower = content.lower()
    scores = {}

    for domain, keywords in DOMAINS.items():
        score = sum(content_lower.count(k) for k in keywords)
        weighted = score * DOMAIN_WEIGHTS.get(domain, 1)
        scores[domain] = weighted

    best = max(scores, key=scores.get)

    if scores[best] < 2:   # threshold to avoid noise
        return "other", 0

    return best, scores[best]

# ============================================================
# RULE-BASED ENUMERATION EXTRACTION
# ============================================================

def extract_enumeration(content):
    content_lower = content.lower()
    items = []
    confidence = 0

    for trigger in TRIGGERS:
        if trigger in content_lower:
            pattern = rf"{trigger}([^\.]+)"
            match = re.search(pattern, content_lower)
            if match:
                segment = match.group(1)
                candidates = re.split(r",| and ", segment)

                for c in candidates:
                    c = re.sub(r"[^a-zA-Z\s]", "", c).strip()
                    if 2 < len(c) < 30:
                        items.append(c.title())

                confidence += 0.5

    items = list(set(items))
    return items, min(confidence, 1.0)

# ============================================================
# SLM FALLBACK (LIMITED)
# ============================================================

def slm_extract(domain, content):
    prompt = f"""
Extract explicit items belonging to domain: {domain}.
Return JSON only:
{{ "items": [] }}

Text:
{content}
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("items", [])
    except:
        pass

    return []

# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

def build_knowledge_base():
    knowledge_store = defaultdict(list)
    total_chunks = 0
    slm_calls = 0

    for file in os.listdir(DATASET_PATH):
        if not file.endswith(".pdf"):
            continue

        logging.info(f"Processing {file}")
        pages = extract_pages(os.path.join(DATASET_PATH, file))

        for page_number, page_text in enumerate(pages):
            chunks = chunk_text(page_text)

            for chunk in chunks:
                if len(chunk.split()) < 30:
                    continue

                domain, score = classify_domain(chunk)
                enum_items, enum_conf = extract_enumeration(chunk)

                if (
                    USE_SLM_FALLBACK and
                    domain != "other" and
                    enum_conf < ENUMERATION_CONF_THRESHOLD and
                    len(chunk.split()) < 300
                ):
                    enum_items = slm_extract(domain, chunk)
                    slm_calls += 1

                entry = {
                    "id": str(uuid.uuid4()),
                    "source_file": file,
                    "page_number": page_number + 1,
                    "entity_type": domain,
                    "contains_enumeration": len(enum_items) > 0,
                    "enumeration_items": enum_items,
                    "content": chunk
                }

                knowledge_store[domain].append(entry)
                total_chunks += 1

        logging.info(f"{file} processed.")

    logging.info(f"Total chunks: {total_chunks}")
    logging.info(f"Total SLM fallback calls: {slm_calls}")

    for domain, entries in knowledge_store.items():
        path = os.path.join(KNOWLEDGE_BASE_PATH, f"{domain}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

        logging.info(f"Saved {domain} → {len(entries)} entries")

    logging.info("Knowledge base built successfully.")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_knowledge_base()