# CIRA — Campus Information Retrieval Assistant

<p align="center">
  <img src="static/cira.png" alt="CIRA Logo" width="120"/>
</p>

<p align="center">
  <b>A locally-hosted, privacy-first campus AI for SR University</b><br/>
  Powered by a fine-tuned Small Language Model · Hybrid RAG · Zero Cloud Dependency
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square"/>
  <img src="https://img.shields.io/badge/FAISS-CPU-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/SLM-On--Premise-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Cloud-0%25-green?style=flat-square"/>
</p>

---

## What is CIRA?

CIRA (**C**ampus **I**nformation **R**etrieval **A**ssistant) is a fully on-premise conversational AI built for SR University. Students and staff can ask natural-language questions about admissions, fees, faculty, hostel, placements, campus locations, exam rules, clubs, and more — and get precise, grounded answers in real time.

Every answer is sourced directly from verified campus documents. No data ever leaves the server. No cloud API is called.

---

## Features

| Capability | How it works |
|---|---|
| **Instant greetings & small talk** | Canned rule-based replies, zero latency |
| **Name lookup** | Direct token match against faculty & leadership JSON — no ML needed |
| **Leadership profiles** | Role-keyword search in `sru_pillars.json` with image + profile link |
| **Faculty search** | Department alias resolution + FAISS semantic fallback |
| **Interactive campus map** | Pin + nearby-place dots, pan/zoom, fullscreen popup |
| **FAQ answering** | Exact → FAISS semantic → RapidFuzz fuzzy — four-tier pipeline |
| **RAG best chunk** | BM25 + FAISS Reciprocal Rank Fusion over chunked corpus |
| **SLM fallback** | Fine-tuned causal LM loaded warm at startup, fires only when all else fails |
| **Handcrafted KB** | `sru_kb.py` for clean, curated answers to high-frequency queries |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Intent Router                 │
│  Greeting → Name → Leadership →         │
│  Faculty → Map → KB → FAQ → RAG → SLM  │
└─────────────────────────────────────────┘
    │
    ├─► sru_pillars.json      (leadership)
    ├─► faculty_metadata.json (faculty)
    ├─► locations.json        (campus map)
    ├─► sru_kb.py             (curated KB)
    ├─► sru_faqs.json         (FAQ store)
    │       └─► FAISS + RapidFuzz
    ├─► chunk_metadata.json   (RAG corpus)
    │       └─► BM25 + FAISS RRF
    └─► ask.py Generator      (SLM fallback)
            └─► AutoModelForCausalLM (local)
```

The SLM (`ask.py`) is imported once at startup and loaded in a **background thread** — the server is live immediately while the model warms up. A `/slm-status` endpoint exposes readiness so the frontend badge updates in real time.

---

## Project Structure

```
CIRA/
├── cira_server.py          # FastAPI backend — intent router + all retrieval layers
├── ask.py                  # SLM inference pipeline (Retriever + Generator)
├── sru_kb.py               # Handcrafted knowledge base lookup
├── cira.html               # Frontend — neumorphic glassmorphism UI
│
├── data/
│   ├── sru_pillars.json        # Chancellor, VC, Registrar, etc.
│   ├── faculty_metadata.json   # All faculty with dept, designation, photo
│   ├── locations.json          # Campus locations with x/y coordinates
│   ├── sru_faqs.json           # Curated FAQ store (query + response + keywords)
│   ├── chunk_metadata.json     # RAG corpus chunks
│   ├── faculty_faiss.index     # FAISS index for faculty embeddings
│   └── sru_index.faiss         # FAISS index for RAG corpus
│
├── static/
│   ├── SRUniversity.png        # Campus map image
│   └── cira.png                # CIRA logo
│
└── requirements.txt
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/cira.git
cd cira
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure paths

Open `cira_server.py` and edit the two CONFIG lines at the top to match your machine:

```python
CORPUS_PATH = Path(r"path/to/your/clean_corpus.txt")
MODEL_DIR   = Path(r"path/to/your/club-slm")
```

If you don't have a trained SLM yet, the server still starts and runs all tiers above the SLM fallback. The `/slm-status` endpoint will report `available: false`.

### 5. Run the server

```bash
uvicorn cira_server:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Open the frontend

Open `cira.html` directly in your browser, or serve it from a local server:

```bash
python -m http.server 5500
# then visit http://localhost:5500/cira.html
```

---

## API Reference

### `POST /chat`

Send a message to CIRA.

**Request body:**
```json
{
  "message": "Where is the library?",
  "mode": "normal"
}
```

**Response — text:**
```json
{
  "type": "text",
  "answer": "The Central Library is located in Block A..."
}
```

**Response — campus map:**
```json
{
  "type": "campus_map",
  "name": "CENTRAL LIBRARY",
  "answer": "📍 CENTRAL LIBRARY is marked on the campus map below.",
  "map_image": "SRUniversity.png",
  "x": 412,
  "y": 275,
  "nearby": [
    { "name": "CANTEEN", "x": 380, "y": 310 },
    { "name": "2ND BLOCK", "x": 440, "y": 250 }
  ]
}
```

**Response — faculty / leadership:**
```json
{
  "type": "image_list",
  "answer": "Found 3 faculty member(s):",
  "sources": [
    {
      "name": "Dr. Example",
      "designation": "Associate Professor",
      "image_link": "https://...",
      "profile_link": "https://sru.edu.in/..."
    }
  ]
}
```

### `GET /slm-status`

Returns whether the SLM has finished loading.

```json
{ "ready": true, "available": true }
```

---

## Answer Priority

CIRA resolves every query through a strict waterfall — the first layer that returns a confident answer wins:

```
0  Greeting         →  instant canned reply
1  Name lookup      →  direct token match in faculty + pillars JSON
2  Leadership       →  role-keyword search in sru_pillars.json
3  Faculty          →  dept alias + FAISS semantic search
4  Campus Map       →  nav-phrase + location noun detection
5  KB Lookup        →  sru_kb.py curated answers
6  FAQ              →  exact match → FAISS semantic → RapidFuzz → difflib
7  RAG              →  BM25 + FAISS Reciprocal Rank Fusion
8  SLM Fallback     →  fine-tuned causal LM (only fires if all above fail)
```

---

## Training the SLM

The SLM is a fine-tuned causal language model trained on SR University's internal document corpus. Training scripts are in a separate pipeline (`train_slm.py`). Once trained, point `MODEL_DIR` to the output folder.

The model is loaded via HuggingFace `transformers` (`AutoModelForCausalLM` + `AutoTokenizer`) and runs entirely on local CPU/GPU — no internet connection required after download.

---

## Team

| Name | Role |
|---|---|
| **B. Jayanth** | Team Lead & Project Architecture |
| **K. Sindhu** | UI/UX Design & Frontend |
| **D. Maruthi** | Backend Integration |
| **CH. Jignesh Shourya** | Data Pipelines & Processing |
| **G. Mahendra** | SLM Architecture & Training |
| **K. Nikhil** | Testing & Quality Assurance |

**Project Guide:** Mr. Gotte Ranjit Kumar — SR University, Department of CSE

---

## License

This project was developed as a Major Project (2026) at SR University. All campus data used is proprietary to SR University and is not included in this repository.

---

<p align="center">CIRA · Campus Intelligence · SR University · 2026</p>
