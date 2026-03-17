# ============================================================
# CIRA SERVER  v1.0  —  COMBINED EDITION (server + SLM in one file)
#
# Answer priority:
#   0. Greeting         → instant canned reply
#   1. Name lookup      → direct check in faculty + pillars JSON (no ML)
#   2. Leadership       → sru_pillars.json (role keywords)
#   3. Faculty          → faculty_metadata.json (dept alias + name)
#   4. Map              → locations.json (nav words + strong location nouns)
#   5. KB lookup        → sru_kb.py (handcrafted clean answers)
#   6. FAQ exact/fuzzy  → sru_faqs.json exact → FAISS semantic → RapidFuzz
#   7. RAG chunk        → BM25 + FAISS RRF best chunk
#   8. SLM fallback     → Generator (RAG-grounded, loaded at startup)
#
# SLM is loaded ONCE at startup (warm, always-on).
# It fires ONLY when all structured sources fail.
# ============================================================

from __future__ import annotations

import warnings, os, logging, sys, threading, argparse, json, math, re, difflib
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── CONFIG ── edit these two lines to match your machine ─────────────────────
CORPUS_PATH = Path(r"data\clean_corpus.txt")
MODEL_DIR   = Path(r"models\club-slm")
# ─────────────────────────────────────────────────────────────────────────────

import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer

try:
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _SLM_DEPS = True
except ImportError as e:
    _SLM_DEPS = False
    print(f"[WARN] SLM dependencies missing: {e}\n"
          "Run: pip install torch transformers scikit-learn\n"
          "SLM fallback will be disabled.")

try:
    from rapidfuzz import fuzz, process as rfprocess
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("[WARN] rapidfuzz not installed — falling back to difflib for fuzzy FAQ matching. "
          "Install with: pip install rapidfuzz")

sys.path.insert(0, os.path.dirname(__file__))
from sru_kb import kb_lookup

# =============================================================================
# ── ASK.PY — SLM COMPONENTS (inlined) ────────────────────────────────────────
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: int
    text: str
    headings: list[str]


@dataclass
class Match:
    chunk_id: int
    score: float
    headings: list[str]
    text: str


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(value: str) -> str:
    value = value.replace("\r", "").replace("\t", " ")
    value = value.replace("\xc3\xa2\xc2\x80\xc2\x99", "'")
    value = value.replace("\xe2\x80\x99", "'")
    value = value.replace("\xc3\xa2\xc2\x80\xc2\x9c", '"')
    value = value.replace("\xe2\x80\x9c", '"')
    value = value.replace("\xc3\xa2\xc2\x80\xc2\x9d", '"')
    value = value.replace("\xe2\x80\x9d", '"')
    value = value.replace("\xc3\xa2\xc2\x80\xc2\x93", "-")
    value = value.replace("\xc3\xa2\xc2\x80\xc2\x94", "-")
    value = value.replace("\xe2\x80\x93", "-")
    value = value.replace("\xe2\x80\x94", "-")
    return value.strip()


def is_heading_line(line: str) -> bool:
    return len(line.split()) <= 18 and bool(re.fullmatch(r"[A-Z0-9 .:&'()/,-]+", line))


def split_into_chunks(text: str, max_words: int = 110) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_headings: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current_lines, current_words
        if not current_lines:
            return
        chunks.append(Chunk(
            chunk_id=len(chunks) + 1,
            text=" ".join(current_lines).strip(),
            headings=list(dict.fromkeys(current_headings)),
        ))
        current_lines = []
        current_words = 0

    for raw in normalize_text(text).split("\n"):
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"--- PAGE \d+ ---", line, flags=re.IGNORECASE):
            flush()
            current_headings = [h for h in current_headings if not h.startswith("--- PAGE")]
            current_headings.append(line)
            continue
        if line.startswith("DOCUMENT:") or re.match(r"=+\s*DOCUMENT:", line):
            flush()
            current_headings = [re.sub(r"=+", "", line).strip()]
            continue
        if is_heading_line(line):
            if current_words >= max_words // 2:
                flush()
            kept = [h for h in current_headings if h.startswith("DOCUMENT:") or h.startswith("--- PAGE")]
            current_headings = kept + [line]
            continue
        line_words = len(line.split())
        force_break = bool(re.match(r"^\d+[.)]\s+", line)) or bool(re.match(r"^\d+\s+[A-Z]", line))
        if current_lines and (current_words + line_words > max_words or force_break):
            flush()
        current_lines.append(line)
        current_words += line_words

    flush()
    return chunks


def compact_answer(text: str, max_sentences: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:max_sentences]) if parts else normalize_text(text)


# ─────────────────────────────────────────────────────────────────────────────
# Domain detection
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "academics":    ["academic", "curriculum", "syllabus", "semester", "credit", "course",
                     "attendance", "timetable", "schedule", "subject", "elective", "module",
                     "lecture", "lab", "practical", "registration", "calendar", "workload",
                     "cgpa", "gpa", "grade", "result", "transcript", "degree", "program"],
    "placements":   ["placement", "internship", "package", "company", "recruiter", "lpa",
                     "ctc", "job", "offer", "campus", "drive", "hiring", "eligible",
                     "placement cell", "career", "industry", "ppo", "pre-placement"],
    "scholarships": ["scholarship", "waiver", "concession", "merit", "financial aid",
                     "stipend", "grant", "fee waiver", "discount", "subsidy"],
    "admissions":   ["admission", "apply", "application", "eligibility", "document",
                     "seat", "intake", "cutoff", "entrance", "jee", "neet", "rank",
                     "deadline", "form", "counselling", "allotment", "lateral entry"],
    "hostel":       ["hostel", "mess", "accommodation", "room", "dormitory", "warden",
                     "resident", "boarding", "canteen", "dining", "curfew", "visitor",
                     "hostel fee", "single room", "double room"],
    "exams":        ["exam", "examination", "test", "quiz", "mid", "end sem", "backlog",
                     "arrear", "supplementary", "revaluation", "answer sheet", "hall ticket",
                     "admit card", "internal", "external", "assessment", "marks", "grade",
                     "passing", "fail", "promotion"],
    "contacts":     ["contact", "faculty", "professor", "hod", "head of department",
                     "dean", "registrar", "office", "email", "phone", "number", "committee",
                     "coordinator", "staff", "administration", "department"],
    "finance":      ["fee", "fees", "tuition", "payment", "fine", "penalty", "refund",
                     "challan", "invoice", "due", "installment", "late fee", "bank",
                     "demand draft", "online payment", "receipt"],
    "policies":     ["policy", "rule", "regulation", "discipline", "ragging", "anti-ragging",
                     "code of conduct", "leave", "absence", "appeal", "grievance",
                     "approval", "permission", "suspension", "expulsion", "misconduct"],
    "student_life": ["club", "event", "fest", "cultural", "sports", "library", "transport",
                     "bus", "shuttle", "facility", "gym", "playground", "auditorium",
                     "activity", "committee", "nss", "ncc", "society", "association"],
}


def detect_domain(question: str) -> str:
    q = question.lower()
    best, best_count = "general", 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in q)
        if count > best_count:
            best_count, best = count, domain
    return best if best_count > 0 else "general"


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────

class Retriever:
    def __init__(self, corpus_path: Path) -> None:
        print(f"[INFO] Loading corpus from {corpus_path} …", flush=True)
        text = corpus_path.read_text(encoding="utf-8", errors="ignore")
        self.chunks = split_into_chunks(text)
        print(f"[INFO] {len(self.chunks)} chunks indexed.", flush=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def _bonus(self, domain: str, q: str, headings: list[str], text: str, base: float) -> float:
        if base < 0.04:
            return 0.0
        joined = " ".join(headings).lower()
        tl = text.lower()
        b = 0.0
        domain_boosts = {
            "academics":    (["curriculum", "syllabus", "course", "academic"], 0.18),
            "placements":   (["placement", "internship", "career"], 0.20),
            "scholarships": (["scholarship", "waiver", "concession", "merit"], 0.22),
            "admissions":   (["admission", "eligib", "apply", "document"], 0.20),
            "hostel":       (["hostel", "accommodat", "mess", "room"], 0.22),
            "exams":        (["exam", "assessment", "grading", "result"], 0.20),
            "contacts":     (["contact", "faculty", "committee", "office"], 0.18),
            "finance":      (["fee", "finance", "payment", "tuition"], 0.20),
            "policies":     (["policy", "rule", "regulation", "conduct", "discipline"], 0.18),
            "student_life": (["club", "event", "activit", "facilit"], 0.16),
        }
        if domain in domain_boosts:
            kws, boost = domain_boosts[domain]
            if any(kw in joined for kw in kws):
                b += boost
        pairs = [
            ("attendance", "attendance", 0.14), ("credit", "credit", 0.12),
            ("backlog", "backlog", 0.16), ("revaluation", "revaluation", 0.14),
            ("refund", "refund", 0.18), ("ragging", "ragging", 0.20),
            ("leave", "leave", 0.14), ("library", "library", 0.14),
            ("mentor", "mentor", 0.12), ("chairperson", "chairperson", 0.12),
            ("executive committee", "executive committee", 0.16),
            ("fee", "fee", 0.18), ("responsibilit", "responsibilit", 0.16),
            ("objective", "objective", 0.16), ("recruitment", "recruitment", 0.16),
        ]
        for qkw, tkw, boost in pairs:
            if qkw in q and tkw in tl:
                b += boost
        return b

    def search(self, question: str, domain: str = "general", top_k: int = 6, min_score: float = 0.08) -> list[Match]:
        q = question.lower()
        qvec = self.vectorizer.transform([question])
        scores = cosine_similarity(qvec, self.matrix).ravel()
        ranked: list[Match] = []
        for i, base in enumerate(scores):
            if base <= 0:
                continue
            chunk = self.chunks[i]
            boosted = float(base) + self._bonus(domain, q, chunk.headings, chunk.text, float(base))
            tl = chunk.text.lower()
            if len(chunk.text.split()) < 18 and "fee structure" in tl:
                boosted -= 0.18
            if boosted < min_score:
                continue
            ranked.append(Match(chunk_id=chunk.chunk_id, score=boosted,
                                headings=chunk.headings, text=chunk.text))
        ranked.sort(key=lambda m: m.score, reverse=True)
        return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Base domain engine
# ─────────────────────────────────────────────────────────────────────────────

class BaseEngine:
    _ABBR = ("Rs.", "Dr.", "Mr.", "Mrs.", "Prof.", "St.", "No.", "Dept.")

    def split_units(self, text: str) -> list[str]:
        p = text
        for a in self._ABBR:
            p = p.replace(a, a.replace(".", "<dot>"))
        parts = [s.strip() for s in re.split(
            r"(?<=[.!?])\s+|\s(?=\d+[.)]\s)|\s(?=[A-Z][A-Za-z &()/.-]{1,30}:)", p
        ) if s.strip()]
        restored = []
        for part in parts:
            for a in self._ABBR:
                part = part.replace(a.replace(".", "<dot>"), a)
            restored.append(part)
        return restored

    def clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\bSource:\s*https?://\S+\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+[,.]", lambda m: m.group().strip(), text)
        text = re.sub(r"^\d+[.)]\s*", "", text)
        text = re.sub(r"\b\d+\.\d+\s+", "", text)
        text = text.strip(" ,")
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def rerank(self, question: str, matches: list[Match]) -> list[Match]:
        return matches

    def extract(self, question: str, matches: list[Match]) -> str | None:
        return None

    def threshold(self) -> float:
        return 0.10

    def _first(self, pattern: str, matches: list[Match], n: int = 3, flags: int = re.IGNORECASE) -> str | None:
        for m in matches[:n]:
            found = re.search(pattern, m.text, flags)
            if found:
                return self.clean(found.group(1).strip())
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Domain engines
# ─────────────────────────────────────────────────────────────────────────────

class AcademicsEngine(BaseEngine):
    def rerank(self, q: str, ms: list[Match]) -> list[Match]:
        ql = q.lower()
        kws = [("credit", 0.8), ("attendance", 0.8), ("syllabus", 0.8),
               ("elective", 0.6), ("registration", 0.6), ("timetable", 0.6)]
        return sorted(ms[:3], key=lambda m: sum(v for k, v in kws if k in ql and k in m.text.lower()), reverse=True)

    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        if "attendance" in ql:
            return self._first(r"((?:minimum|mandatory|required)\s+attendance[^.]{0,160}\.)", ms)
        if "credit" in ql:
            return self._first(r"(\d+\s+credits?[^.]{0,120}\.)", ms)
        if "semester" in ql or "calendar" in ql:
            return self._first(r"((?:semester|academic year)[^.]{0,200}\.)", ms)
        if "elective" in ql:
            return self._first(r"(elective[^.]{0,200}\.)", ms)
        return None

    def threshold(self) -> float:
        return 0.09


class PlacementsEngine(BaseEngine):
    def rerank(self, q: str, ms: list[Match]) -> list[Match]:
        ql = q.lower()
        def score(m: Match) -> float:
            t = m.text.lower()
            s = m.score
            if "package" in ql and ("lpa" in t or "ctc" in t):   s += 1.2
            if "eligib" in ql and "eligib" in t:                   s += 0.8
            if "internship" in ql and "internship" in t:           s += 1.0
            return s
        return sorted(ms[:3], key=score, reverse=True)

    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        if any(k in ql for k in ("package", "salary", "ctc", "lpa")):
            r = self._first(r"((?:average|highest|minimum|maximum|median)\s+(?:package|salary|ctc)[^.]{0,200}\.)", ms)
            return r or self._first(r"(\d+(?:\.\d+)?\s*LPA[^.]{0,100}\.)", ms)
        if "eligib" in ql:
            return self._first(r"((?:student|candidate)s?\s+(?:must|should|need to|are required to)[^.]{0,200}\.)", ms)
        if "internship" in ql:
            return self._first(r"(internship[^.]{0,200}\.)", ms)
        if "compan" in ql or "recruiter" in ql:
            for m in ms[:3]:
                items = re.findall(r"(\d+[.)]\s*[A-Z][A-Za-z &.,()-]{2,80})", m.text)
                if items:
                    return self.clean(", ".join(i.strip() for i in items[:6]))
        return None

    def threshold(self) -> float:
        return 0.09


class ScholarshipsEngine(BaseEngine):
    def rerank(self, q: str, ms: list[Match]) -> list[Match]:
        ql = q.lower()
        def score(m: Match) -> float:
            t = m.text.lower(); s = m.score
            if "merit" in ql and "merit" in t:   s += 1.0
            if "waiver" in ql and "waiver" in t: s += 1.0
            if "eligib" in ql and "eligib" in t: s += 0.8
            return s
        return sorted(ms[:3], key=score, reverse=True)

    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        if any(k in ql for k in ("amount", "value", "how much")):
            return self._first(r"((?:scholarship|waiver|concession)[^.]{0,200}(?:Rs\.|₹|\d+\s*%)[^.]{0,100}\.)", ms)
        if "eligib" in ql or "criteria" in ql or "who" in ql:
            return self._first(r"((?:student|candidate)s?\s+(?:with|scoring|having|who)[^.]{0,200}\.)", ms)
        return self._first(r"((?:scholarship|fee waiver|merit)[^.]{0,250}\.)", ms)

    def threshold(self) -> float:
        return 0.09


class AdmissionsEngine(BaseEngine):
    def rerank(self, q: str, ms: list[Match]) -> list[Match]:
        ql = q.lower()
        def score(m: Match) -> float:
            t = m.text.lower(); s = m.score
            if "document" in ql and "document" in t:             s += 1.0
            if "deadline" in ql and ("deadline" in t or "last date" in t): s += 1.0
            if "eligib" in ql and "eligib" in t:                  s += 0.8
            if "seat" in ql and "seat" in t:                      s += 0.8
            return s
        return sorted(ms[:3], key=score, reverse=True)

    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        if "document" in ql:
            for m in ms[:3]:
                items = re.findall(r"(\d+[.)]\s*[A-Za-z][^.\n]{5,100})", m.text)
                if items:
                    return " ".join(self.clean(i) for i in items[:6])
            return self._first(r"((?:following documents|documents required)[^.]{0,300}\.)", ms)
        if "eligib" in ql or "qualify" in ql or "minimum" in ql:
            return self._first(r"((?:minimum|required)\s+(?:marks?|percentage|score|cgpa|rank)[^.]{0,200}\.)", ms)
        if "seat" in ql or "intake" in ql:
            return self._first(r"(\d+\s+(?:seats?|students?|intake)[^.]{0,150}\.)", ms)
        if "deadline" in ql or "last date" in ql or "when" in ql:
            return self._first(r"((?:deadline|last date|closing date)[^.]{0,150}\.)", ms)
        return None

    def threshold(self) -> float:
        return 0.09


class HostelEngine(BaseEngine):
    def rerank(self, q: str, ms: list[Match]) -> list[Match]:
        ql = q.lower()
        def score(m: Match) -> float:
            t = m.text.lower(); s = m.score
            if "fee" in ql and "hostel fee" in t: s += 1.5
            if "mess" in ql and "mess" in t:      s += 1.0
            if "rule" in ql and "rule" in t:      s += 0.8
            return s
        return sorted(ms[:3], key=score, reverse=True)

    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        if any(k in ql for k in ("fee", "cost", "charge")):
            r = self._first(r"((?:hostel fee|mess fee|accommodation fee)[^.]{0,200}\.)", ms)
            return r or self._first(r"(Rs\.[^.]{0,150}(?:hostel|mess|room)[^.]{0,100}\.)", ms)
        if any(k in ql for k in ("rule", "curfew", "timing")):
            return self._first(r"((?:curfew|in-time|out-time|timing)[^.]{0,200}\.)", ms)
        if "warden" in ql or "contact" in ql:
            return self._first(r"((?:warden|chief warden)[^.]{0,150}\.)", ms)
        return None

    def threshold(self) -> float:
        return 0.09


class ExamsEngine(BaseEngine):
    def rerank(self, q: str, ms: list[Match]) -> list[Match]:
        ql = q.lower()
        def score(m: Match) -> float:
            t = m.text.lower(); s = m.score
            if "backlog" in ql and "backlog" in t:         s += 1.5
            if "revaluation" in ql and "revaluation" in t: s += 1.5
            if "grade" in ql and "grade" in t:             s += 0.8
            if "cgpa" in ql and "cgpa" in t:               s += 0.8
            return s
        return sorted(ms[:3], key=score, reverse=True)

    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        if any(k in ql for k in ("grade", "cgpa", "gpa")):
            return self._first(r"((?:grading|grade point|cgpa|gpa)[^.]{0,250}\.)", ms)
        if any(k in ql for k in ("pass", "minimum", "fail")):
            return self._first(r"((?:minimum|pass|passing)[^.]{0,200}(?:marks?|grade|score)[^.]{0,100}\.)", ms)
        if "backlog" in ql or "arrear" in ql:
            return self._first(r"((?:backlog|arrear|supplementary)[^.]{0,250}\.)", ms)
        if "revaluation" in ql or "recheck" in ql:
            return self._first(r"((?:revaluation|re-checking|recheck)[^.]{0,250}\.)", ms)
        if "schedule" in ql or "hall ticket" in ql:
            return self._first(r"((?:exam schedule|hall ticket|admit card)[^.]{0,200}\.)", ms)
        return None

    def threshold(self) -> float:
        return 0.09


class ContactsEngine(BaseEngine):
    def rerank(self, q: str, ms: list[Match]) -> list[Match]:
        ql = q.lower()
        def score(m: Match) -> float:
            t = m.text.lower(); s = m.score
            if "dean" in ql and "dean" in t:         s += 1.0
            if "hod" in ql and "hod" in t:           s += 1.0
            if "email" in ql and "@" in t:            s += 0.8
            if "phone" in ql and re.search(r"\d{10}", t): s += 0.8
            return s
        return sorted(ms[:3], key=score, reverse=True)

    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        role_patterns = [
            (["dean"],             r"(Dean[^:]{0,30}:\s*[A-Za-z .]+)"),
            (["registrar"],        r"(Registrar[^:]{0,30}:\s*[A-Za-z .]+)"),
            (["hod", "head of department"], r"((?:HoD|Head of Department)[^:]{0,30}:\s*[A-Za-z .]+)"),
            (["coordinator"],      r"(Coordinator[^:]{0,30}:\s*[A-Za-z .]+)"),
            (["chairperson"],      r"(Chairperson[^:]{0,30}:\s*[A-Za-z .]+)"),
        ]
        for keys, pattern in role_patterns:
            if any(k in ql for k in keys):
                r = self._first(pattern, ms)
                if r:
                    return r
        if "email" in ql or "mail" in ql:
            for m in ms[:3]:
                emails = re.findall(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", m.text)
                if emails:
                    return self.clean(", ".join(emails[:3]))
        if "phone" in ql or "number" in ql:
            for m in ms[:3]:
                phones = re.findall(r"(?:\+91[-\s]?)?[0-9]{10}", m.text)
                if phones:
                    return self.clean(", ".join(phones[:3]))
        return None

    def threshold(self) -> float:
        return 0.08


class FinanceEngine(BaseEngine):
    def rerank(self, q: str, ms: list[Match]) -> list[Match]:
        ql = q.lower()
        def score(m: Match) -> float:
            t = m.text.lower(); s = m.score
            if "tuition fee" in t:                        s += 1.2
            if "rs." in t:                                s += 1.0
            if "admission fee" in t or "enrollment fee" in t: s += 0.7
            if "refund" in ql and "refund" in t:          s += 1.0
            if "fine" in ql and "fine" in t:              s += 0.8
            if len(m.text.split()) < 20:                  s -= 1.0
            return s
        return sorted(ms[:3], key=score, reverse=True)

    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        if "refund" in ql:
            return self._first(r"((?:refund|cancellation)[^.]{0,300}\.)", ms)
        if any(k in ql for k in ("fine", "penalty", "late")):
            return self._first(r"((?:fine|penalty|late fee)[^.]{0,200}(?:Rs\.|₹|\d+)[^.]{0,100}\.)", ms)
        best_m, best_v = None, -1.0
        for m in ms[:3]:
            t = m.text.lower()
            if "fee" not in t:
                continue
            sv = (
                (2.5 if "tuition fee" in t else 0) +
                (1.5 if "hostel fee" in t else 0) +
                (1.2 if "admission fee" in t else 0) +
                (1.2 if "enrollment fee" in t or "enrolment fee" in t else 0) +
                (1.2 if "per semester" in t or "per year" in t else 0) +
                (2.0 if "rs." in t or "₹" in m.text else 0) +
                (-1.0 if len(m.text.split()) < 20 else 0)
            )
            if m.score + sv > best_v:
                best_v, best_m = m.score + sv, m
        if not best_m:
            return None
        text = re.sub(r"\s+", " ", best_m.text).strip()
        for pattern in [
            r"(For the .*? program, the tuition fee per semester is Rs\..*?applicable\.)",
            r"(For the .*? program, the tuition fee per semester is Rs\..*?\.)",
            r"(The fee structure for this program is as follows\..*?per year\.)",
            r"(For B\.Tech programs.*?tuition fee per semester is Rs\..*?\.)",
            r"(.*?one-time Enrollment Fee.*?Admission Fee.*?\.)",
        ]:
            found = re.search(pattern, text, re.IGNORECASE)
            if found:
                return self.clean(found.group(1).strip())
        return self.clean(text)

    def threshold(self) -> float:
        return 0.09


class PoliciesEngine(BaseEngine):
    def rerank(self, q: str, ms: list[Match]) -> list[Match]:
        ql = q.lower()
        def score(m: Match) -> float:
            t = m.text.lower(); s = m.score
            if "ragging" in ql and "ragging" in t:       s += 1.5
            if "leave" in ql and "leave" in t:            s += 1.0
            if "conduct" in ql and "conduct" in t:        s += 0.8
            if "discipline" in ql and "disciplin" in t:   s += 0.8
            return s
        return sorted(ms[:3], key=score, reverse=True)

    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        if "ragging" in ql:
            return self._first(r"((?:anti-ragging|ragging)[^.]{0,300}\.)", ms)
        if "leave" in ql:
            return self._first(r"((?:leave policy|leave application|casual leave|medical leave)[^.]{0,250}\.)", ms)
        if "grievance" in ql or "complaint" in ql:
            return self._first(r"((?:grievance|complaint)[^.]{0,250}\.)", ms)
        if "conduct" in ql or "discipline" in ql:
            for m in ms[:3]:
                units = self.split_units(m.text)
                picked = [self.clean(u) for u in units
                          if any(k in u.lower() for k in ["conduct", "disciplin", "prohibited", "violation"])]
                if picked:
                    return " ".join(picked[:3])
        return None

    def threshold(self) -> float:
        return 0.09


class StudentLifeEngine(BaseEngine):
    def rerank(self, q: str, ms: list[Match]) -> list[Match]:
        ql = q.lower()
        def score(m: Match) -> float:
            t = m.text.lower(); s = m.score
            if "club" in ql and "club" in t:           s += 1.0
            if "library" in ql and "library" in t:     s += 1.0
            if "transport" in ql and "transport" in t: s += 1.0
            if "sports" in ql and "sports" in t:       s += 0.6
            return s
        return sorted(ms[:3], key=score, reverse=True)

    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        if "library" in ql:
            return self._first(r"(library[^.]{0,250}\.)", ms)
        if any(k in ql for k in ("transport", "bus", "shuttle")):
            return self._first(r"((?:bus|transport|shuttle)[^.]{0,250}\.)", ms)
        if "club" in ql:
            units = []
            for m in ms[:3]:
                units += self.split_units(m.text)
            picked = [self.clean(u) for u in units if "club" in u.lower() and len(u.split()) > 8]
            if picked:
                return " ".join(picked[:3])
        if any(k in ql for k in ("sports", "gym", "playground")):
            return self._first(r"((?:sports|gym|playground|ground)[^.]{0,250}\.)", ms)
        return None

    def threshold(self) -> float:
        return 0.09


class GeneralEngine(BaseEngine):
    def extract(self, q: str, ms: list[Match]) -> str | None:
        ql = q.lower()
        if "faculty mentor" in ql or ("mentor" in ql and "who" in ql):
            for pattern in [
                r"(Faculty Mentor must be from the School of Computer Science & AI)",
                r"(The Mentor shall be a faculty member[^.]*\.)",
                r"(The faculty mentor shall be generally responsible[^.]*\.)",
                r"(Faculty Mentor[^.]{0,120})",
            ]:
                r = self._first(pattern, ms)
                if r:
                    return r
        role_fields: list[str] = []
        if "chairperson" in ql and "vice" not in ql:
            role_fields = ["chairperson"]
        elif any(k in ql for k in ("vice chair", "vice-chair", "vice chairperson")):
            role_fields = ["vice-chair", "vice chair", "vice chairperson"]
        elif "secretary" in ql:
            role_fields = ["secretary"]
        elif "treasurer" in ql:
            role_fields = ["treasurer"]
        if role_fields:
            for m in ms:
                for unit in self.split_units(m.text):
                    lu = unit.lower()
                    for f in role_fields:
                        if f"{f}:" in lu:
                            found = re.search(r"([A-Za-z][A-Za-z &()/.-]+:\s*.+)", unit)
                            if found:
                                return self.clean(found.group(1).strip())
        if "responsibilit" in ql:
            for m in ms[:3]:
                ht = " ".join(m.headings).lower()
                if "responsibilities of clubs" in ht or "responsibilities of clubs" in m.text.lower():
                    units = self.split_units(m.text)
                    picked = [self.clean(u) for u in units
                              if not u.lower().startswith("11. faculty mentor")
                              and any(k in u.lower() for k in [
                                  "club office bearers are charged", "clubs are responsible",
                                  "each club must communicate", "submit annual financial",
                                  "update club member lists",
                              ])]
                    if picked:
                        return " ".join(picked[:3])
                    return self.clean(compact_answer(m.text, max_sentences=3))
        if "objective" in ql:
            for m in ms[:3]:
                ht = " ".join(m.headings).lower()
                if "introduction" in ht or "objective" in m.text.lower():
                    found = re.search(r"(Objective:\s*[^.]+(?:\.[^.]+)?)", m.text, re.IGNORECASE)
                    if found:
                        return self.clean(found.group(1).strip())
                    return self.clean(compact_answer(m.text, max_sentences=2))
        if "recruitment" in ql:
            for m in ms[:3]:
                if "recruitment" in " ".join(m.headings).lower():
                    return self.clean(compact_answer(m.text, max_sentences=3))
        if "executive committee" in ql or "roles" in ql:
            for m in ms[:3]:
                rl = re.search(r"required to have a (.+?)\. The major criteria of Executive Committee",
                               m.text, re.IGNORECASE)
                if rl:
                    return self.clean(rl.group(1).strip())
                items = re.findall(
                    r"(\d+[.)]?\s*(?:Chairperson|Vice-Chair|Secretary|Treasurer|Marketing\s*&\s*PR Secretary|"
                    r"Web Master|Membership Chair|Management Head|Content\s*&\s*Creative Head|"
                    r"Digital\s*&\s*Social Media Head)[^.]*)",
                    m.text, re.IGNORECASE,
                )
                if items:
                    return " ".join(self.clean(i) for i in items[:10])
        return None

    def threshold(self) -> float:
        return 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Engine registry
# ─────────────────────────────────────────────────────────────────────────────

_ENGINES: dict[str, BaseEngine] = {
    "academics":    AcademicsEngine(),
    "placements":   PlacementsEngine(),
    "scholarships": ScholarshipsEngine(),
    "admissions":   AdmissionsEngine(),
    "hostel":       HostelEngine(),
    "exams":        ExamsEngine(),
    "contacts":     ContactsEngine(),
    "finance":      FinanceEngine(),
    "policies":     PoliciesEngine(),
    "student_life": StudentLifeEngine(),
    "general":      GeneralEngine(),
}


def get_engine(domain: str) -> BaseEngine:
    return _ENGINES.get(domain, _ENGINES["general"])


# ─────────────────────────────────────────────────────────────────────────────
# Generator  (LLM layer)
# ─────────────────────────────────────────────────────────────────────────────

class Generator:
    def __init__(self, model_dir: Path) -> None:
        if not _SLM_DEPS:
            raise RuntimeError("SLM dependencies not installed.")
        model_dir = Path(model_dir).resolve()
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model not found: {model_dir}\n"
                "Edit MODEL_DIR at the top of this file, or run train_slm.py first."
            )
        print(f"[INFO] Loading model from {model_dir} …", flush=True)
        d = str(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(d, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(d, local_files_only=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Model ready on {self.device}.", flush=True)

    def _gen(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=120, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = re.sub(r"^(Answer:|Rewritten answer:)\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bSource\s+\d+:.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\bSource:\s*https?://\S+", "", text, flags=re.IGNORECASE).strip()
        return text or "Not found in corpus."

    def answer_with_context(self, question: str, matches: list[Match]) -> str:
        ctx = "\n\n".join(
            f"Source {i+1}: {' > '.join(m.headings)}\n{compact_answer(m.text, max_sentences=4)}"
            for i, m in enumerate(matches)
        )
        return self._gen(
            "Answer the question only from the context. "
            "If the answer is missing, say 'Not found in corpus.' "
            "Return only the answer text.\n\n"
            f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"
        )

    def rewrite(self, question: str, extracted: str) -> str:
        return self._gen(
            "Rewrite the answer into one clean grammatical sentence or short paragraph.\n"
            "Use only the facts already present in the answer.\n"
            "Do not add any new facts. Do not add bullet points.\n"
            "Do not add source names, URLs, headings, or labels.\n"
            "Do not change any numbers, fees, or program names.\n"
            "Return only the rewritten answer text.\n\n"
            f"Question: {question}\nAnswer to rewrite: {extracted}\n\nRewritten answer:"
        )


# =============================================================================
# ── FASTAPI SERVER ────────────────────────────────────────────────────────────
# =============================================================================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Embedding model ───────────────────────────────────────────────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready")

# ── Data files ────────────────────────────────────────────────────────────────
print("Loading data files...")
with open("data/sru_pillars.json",     "r", encoding="utf-8") as f: PILLARS          = json.load(f)["leaders"]
with open("data/faculty_metadata.json","r", encoding="utf-8") as f: faculty_metadata = json.load(f)
with open("data/locations.json",       "r", encoding="utf-8") as f: locations_data   = json.load(f)
with open("data/sru_faqs.json",        "r", encoding="utf-8") as f: FAQS             = json.load(f)
with open("data/chunk_metadata.json",  "r", encoding="utf-8") as f: rag_metadata     = json.load(f)

faculty_index = faiss.read_index("data/faculty_faiss.index")
rag_index     = faiss.read_index("data/sru_index.faiss")

# ── FAQ FAISS index ───────────────────────────────────────────────────────────
_faq_q   = [f["query"] for f in FAQS]
_faq_emb = embed_model.encode(_faq_q, show_progress_bar=False).astype("float32")
faiss.normalize_L2(_faq_emb)
faq_index = faiss.IndexFlatIP(_faq_emb.shape[1])
faq_index.add(_faq_emb)

# ── RAG texts + BM25 ─────────────────────────────────────────────────────────
rag_texts = []
for c in rag_metadata:
    if isinstance(c, dict): rag_texts.append(c.get("text") or c.get("chunk_text") or "")
    elif isinstance(c, str): rag_texts.append(c)
    else: rag_texts.append("")

from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([t.lower().split() for t in rag_texts])

# ── SLM warm-load (background thread so server starts immediately) ────────────
_slm_generator = None
_slm_retriever = None
_slm_ready     = False
_slm_lock      = threading.Lock()

def _load_slm():
    global _slm_generator, _slm_retriever, _slm_ready
    if not _SLM_DEPS:
        print("[SLM] Dependencies not available — SLM fallback disabled.")
        return
    if not CORPUS_PATH.exists():
        print(f"[SLM] Corpus not found at {CORPUS_PATH} — SLM fallback disabled.")
        return
    if not MODEL_DIR.exists():
        print(f"[SLM] Model not found at {MODEL_DIR} — SLM fallback disabled.")
        return
    try:
        print("[SLM] Loading generator and retriever in background…")
        gen = Generator(MODEL_DIR)
        ret = Retriever(CORPUS_PATH)
        with _slm_lock:
            _slm_generator = gen
            _slm_retriever = ret
            _slm_ready     = True
        print("[SLM] ✓ Ready — SLM fallback active.")
    except Exception as e:
        print(f"[SLM] Load failed: {e}")

threading.Thread(target=_load_slm, daemon=True).start()

print(f"CIRA v1.0 online | pillars:{len(PILLARS)}  faculty:{len(faculty_metadata)}  "
      f"locations:{len(locations_data)}  faqs:{len(FAQS)}  rag:{len(rag_texts)}\n")

# =============================================================================
# UTILITIES
# =============================================================================

def _q(s: str) -> str: return s.lower().strip()

_TOC_LINE = re.compile(r'^\s*(\d+\.[\d.]*\s|[•\-–—]\s|[A-Z][A-Z\s]{4,}$)', re.I)

def _clean(text: str, limit: int = 380) -> str:
    if not text: return text
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    good  = [l for l in lines
             if not (len(l) < 80 and _TOC_LINE.match(l))
             and not re.match(r'^\d+(\.\d+)*\s+[A-Z]', l)]
    out = " ".join(good).strip() or text.strip()
    if len(out) > limit:
        cut = out[:limit]
        dot = max(cut.rfind(". "), cut.rfind(".\n"))
        out = (cut[:dot+1] if dot > limit//2 else cut.rstrip() + "…")
    return out

# =============================================================================
# GREETING
# =============================================================================

_GREET  = {"hi","hello","hey","heyy","heya","howdy","sup","yo"}
_BYE    = {"bye","goodbye","see you","see ya","take care","later","cya"}
_THANKS = {"thanks","thank you","ty","thx","thank","thankyou"}
_OK     = {"ok","okay","cool","great","nice","alright","got it","noted"}
_TGREET = {"good morning","good afternoon","good evening","good night"}

def _greeting(query: str) -> str | None:
    q = query.lower().strip().rstrip("!.,?")
    w = q.split()
    if not w or len(w) > 4: return None
    if q in _TGREET:     return "Good day! 😊 I'm CIRA, SR University's campus assistant. How can I help?"
    if w[0] in _GREET:   return "Hi there! 👋 I'm CIRA. Ask me about admissions, fees, hostel, placement, faculty, or campus locations!"
    if any(b in q for b in _BYE): return "Goodbye! 👋 Come back anytime you need campus info."
    if w[0] in _THANKS:  return "You're welcome! 😊 Let me know if you need anything else."
    if q in _OK:         return "Sure! 😊 Feel free to ask me anything about SR University."
    return None

# =============================================================================
# NAME LOOKUP
# =============================================================================

def _name_lookup(query: str) -> dict | None:
    q = _q(query)
    tokens = q.split()
    _SKIP_STARTS = {"what","how","when","where","who is the","list","show",
                    "tell","give","find","does","is there","can i","do you"}
    if any(q.startswith(s) for s in _SKIP_STARTS): return None
    if len(tokens) < 1 or len(tokens) > 5: return None
    if not any(len(t) >= 4 for t in tokens): return None

    for leader in PILLARS:
        name = leader["person"]["name"].lower()
        if all(t in name for t in tokens if len(t) >= 3):
            p = leader["person"]
            return {
                "type": "image_list",
                "answer": f"Here is {p['name']}, {p.get('designation', '')}:",
                "sources": [{
                    "name":         p["name"],
                    "designation":  p.get("designation", ""),
                    "image_link":   p.get("image_url", ""),
                    "profile_link": p.get("profile_link", ""),
                }]
            }

    hits = [f for f in faculty_metadata
            if all(t in f.get("name", "").lower() for t in tokens if len(t) >= 3)]
    if hits:
        return {
            "type":    "image_list",
            "answer":  f"Found {len(hits)} result(s) for '{query}':",
            "sources": hits[:5],
        }
    return None

# =============================================================================
# LEADERSHIP
# =============================================================================

_LEAD_ROLES   = {"chancellor","vice chancellor","vc","pro chancellor",
                 "registrar","provost","controller","treasurer","rector","president"}
_HOD_BLOCK    = {"hod","head of department","dept head","department head",
                 "cse head","ece head","eee head","civil head","mba head","cse hod","ece hod"}
_LEAD_PHRASES = ["who is the chancellor","who is the vc","who is the vice chancellor",
                 "who leads","top leadership","university management"]

def _is_leadership(q: str) -> bool:
    if any(h in q for h in _HOD_BLOCK): return False
    for r in _LEAD_ROLES:
        if re.search(r'\b' + re.escape(r) + r'\b', q): return True
    return any(p in q for p in _LEAD_PHRASES)

def _search_leader(q: str):
    for l in PILLARS:
        if any(kw.lower() in q for kw in l.get("retrieval_keywords", [])): return l
    for l in PILLARS:
        role = l.get("role_type","").lower()
        if role and re.search(r'\b' + re.escape(role) + r'\b', q): return l
    best, bs = None, 0
    for l in PILLARS:
        s = difflib.SequenceMatcher(None, q, l["person"]["name"].lower()).ratio()
        if s > bs: bs, best = s, l
    return best if bs > 0.4 else None

# =============================================================================
# FACULTY
# =============================================================================

DEPT_ALIAS = {
    "cse":"computer science","csai":"computer science and artificial intelligence",
    "cs":"computer science","computer science":"computer science",
    "ai":"artificial intelligence","ece":"electronics and communication",
    "eee":"electrical and electronics","civil":"civil","mechanical":"mechanical",
    "mba":"business","management":"business","math":"mathematics","maths":"mathematics",
    "physics":"physics","chemistry":"chemistry","english":"english",
    "humanities":"humanities","mtech":"m.tech","phd":"ph.d",
}

_FAC_SIGS = {"faculty","professor","prof","dr.","lecturer","instructor",
             "assistant professor","associate professor","department staff",
             "who teaches","subject teacher","course instructor","lab faculty",
             "faculty list","faculties","staff","teachers",
             "hod","head of department","dept head","department head",
             "cse head","ece head","cse hod","ece hod","eee head"}

_DEPT_FAC_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in DEPT_ALIAS) + r')'
    r'\s*(faculty|staff|hod|head|professor|lecturer|teacher|list)\b'
    r'|\b(faculty|staff|hod|head|professor|lecturer|teacher)\s*'
    r'(' + '|'.join(re.escape(k) for k in DEPT_ALIAS) + r')\b'
)

def _is_faculty(q: str) -> bool:
    if any(s in q for s in _FAC_SIGS): return True
    if _DEPT_FAC_RE.search(q): return True
    return any(len(w) > 4 and any(w in f.get("name","").lower() for f in faculty_metadata)
               for w in q.split())

def _search_faculty(q: str, k: int = 5) -> list:
    want_head = any(h in q for h in ("hod","head","dept head","department head"))
    for alias, dept_key in DEPT_ALIAS.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', q):
            matched = [f for f in faculty_metadata
                       if dept_key in f.get("department","").lower()
                       or dept_key in f.get("school","").lower()]
            if matched:
                if want_head:
                    hod = [f for f in matched
                           if any(h in f.get("designation","").lower()
                                  for h in ("head","hod","director"))]
                    if hod: return hod[:k]
                return matched[:k]
    nm = [f for f in faculty_metadata
          if any(w in f.get("name","").lower() for w in q.split() if len(w) > 3)]
    if nm: return nm[:k]
    for dw in ("professor","hod","head","director","dean","lecturer"):
        if dw in q:
            dm = [f for f in faculty_metadata if dw in f.get("designation","").lower()]
            if dm: return dm[:k]
    D, I = faculty_index.search(embed_model.encode([q]).astype("float32"), k)
    return [faculty_metadata[i] for i in I[0] if i < len(faculty_metadata)]

# =============================================================================
# MAP
# =============================================================================

_MAP_PHRASES = {"where is","where's","how to reach","directions to","show me the",
                "navigate to","take me to","location of","find the","show the",
                "show map","campus map","on the map","where can i find","how do i get to"}
_MAP_STRONG  = {"canteen","library","parking","auditorium","amphi","amphitheatre",
                "xerox","temple","ground","indoor games","dance club","memo counter",
                "admin office","srix","checkpoint","cafeteria","gymnasium"}
_MAP_WEAK    = {"hostel","mess","gym","gate","workshop","lab"}
_MAP_NAV     = {"where","location","find","locate","show","go","reach","navigate","directions"}
_FEE_BLOCK   = {"fee","fees","cost","price","charges","amount","how much","rate",
                "rent","pay","placement","package","packages","admission","process",
                "procedure","scholarship","lpa","salary","accommodation charges"}

def _is_map(q: str) -> bool:
    if any(sig in q for sig in _FEE_BLOCK): return False
    for p in _MAP_PHRASES:
        if p in q: return True
    if re.search(r'\bmap\b', q): return True
    for n in _MAP_STRONG:
        if re.search(r'\b' + re.escape(n) + r'\b', q): return True
    for n in _MAP_WEAK:
        if re.search(r'\b' + re.escape(n) + r'\b', q) and any(nw in q for nw in _MAP_NAV):
            return True
    for loc in locations_data:
        lw = [w for w in loc.lower().split() if len(w) > 3]
        if lw and all(w in q for w in lw): return True
    if re.search(r'\bblock\s*[a-z0-9]\b|\b[a-z][0-9]?\s*block\b', q): return True
    return False

_STRIP_NAV = {"where","is","the","location","of","show","me","find","locate",
              "navigate","directions","to","how","reach","where's","map","on",
              "campus","a","an","at","can","i","get"}

def _search_location(query: str):
    q  = _q(query)
    qc = " ".join(w for w in q.split() if w not in _STRIP_NAV).strip() or q
    ws = set(qc.split())
    _FOOD = {"eat","food","dining","canteen","cafeteria","lunch","breakfast",
             "dinner","snack","hungry","mess"}
    if ws & _FOOD:
        target = "HOSTEL MESS" if ("mess" in q or "hostel" in q) else "CANTEEN"
        if target in locations_data:
            return {"name": target, **{k: locations_data[target][k] for k in ("x","y")}}
    for name, loc in locations_data.items():
        if name.lower() == qc: return {"name": name, "x": loc["x"], "y": loc["y"]}
    for name, loc in locations_data.items():
        if name.lower() in qc: return {"name": name, "x": loc["x"], "y": loc["y"]}
    if len(qc) > 3:
        for name, loc in locations_data.items():
            if qc in name.lower(): return {"name": name, "x": loc["x"], "y": loc["y"]}
    for name, loc in locations_data.items():
        tags = [t.strip().lower() for t in str(loc.get("tag","")).split(",")]
        if any(t and len(t) > 3 and t in qc for t in tags):
            return {"name": name, "x": loc["x"], "y": loc["y"]}
    for name, loc in locations_data.items():
        nw = {w for w in name.lower().split() if len(w) > 3}
        if nw and nw.issubset(ws): return {"name": name, "x": loc["x"], "y": loc["y"]}
    if len(qc) > 3:
        best, bs = None, 0
        for name, loc in locations_data.items():
            s = difflib.SequenceMatcher(None, qc, name.lower()).ratio()
            if s > bs: bs, best = s, name
        if bs > 0.55: return {"name": best, **{k: locations_data[best][k] for k in ("x","y")}}
    return None

def _nearby(x, y, limit=5):
    arr = sorted(
        ((math.sqrt((loc["x"]-x)**2 + (loc["y"]-y)**2), n)
         for n, loc in locations_data.items()),
        key=lambda t: t[0]
    )
    return [n for _, n in arr[1:limit+1]]

# =============================================================================
# FAQ  — Exact → FAISS semantic → RapidFuzz (Fuzzy)
# =============================================================================

def _search_faq(query: str) -> dict | None:
    q = _q(query)
    for faq in FAQS:
        fq = faq["query"].lower()
        if q == fq or q in fq or fq in q:
            return faq
    qe = embed_model.encode([q]).astype("float32")
    faiss.normalize_L2(qe)
    D, I = faq_index.search(qe, 3)
    for rank in range(len(I[0])):
        score = float(D[0][rank])
        idx   = I[0][rank]
        if idx >= len(FAQS): continue
        best = FAQS[idx]
        boost = sum(0.05 for kw in best.get("keywords",[]) if kw.lower() in q)
        if (score + boost) >= 0.68:
            return best
    for faq in FAQS:
        if sum(1 for kw in faq.get("keywords",[]) if kw.lower() in q and len(kw) > 3) >= 2:
            return faq
    if HAS_RAPIDFUZZ:
        choices = {faq["query"]: faq for faq in FAQS}
        result  = rfprocess.extractOne(q, choices.keys(),
                                       scorer=fuzz.token_set_ratio,
                                       score_cutoff=72)
        if result:
            matched_q, score, _ = result
            return choices[matched_q]
    best_faq, best_score = None, 0.0
    for faq in FAQS:
        s = difflib.SequenceMatcher(None, q, faq["query"].lower()).ratio()
        if s > best_score:
            best_score, best_faq = s, faq
    if best_score >= 0.62 and best_faq:
        return best_faq
    return None

# =============================================================================
# RAG  — BM25 + FAISS RRF best chunk
# =============================================================================

_PLACE_KW = {"placement","package","recruiter","lpa","offer","job","hired","salary","ctc"}
_HOST_KW  = {"hostel","accommodation","dormitory","room","boarding"}
_FEE_KW   = {"fee","tuition","payment","scholarship","dues","installment"}
_ADM_KW   = {"admission","jee","eamcet","rank","cutoff","apply","eligibility"}
_EXAM_NZ  = {"exam","grading","supplementary","grade awarded","outstanding","below average"}

def _dom_ok(text, q):
    t = text.lower()
    if any(k in q for k in _PLACE_KW):
        if any(k in t for k in _EXAM_NZ): return False
        if not any(k in t for k in _PLACE_KW): return False
    if any(k in q for k in _HOST_KW)  and not any(k in t for k in _HOST_KW):  return False
    if any(k in q for k in _FEE_KW)   and not any(k in t for k in _FEE_KW):   return False
    if any(k in q for k in _ADM_KW)   and not any(k in t for k in _ADM_KW):   return False
    return True

def _best_chunk(query: str) -> str:
    q  = query.lower()
    tk = q.split()
    qe = embed_model.encode([q]).astype("float32")
    D, I = rag_index.search(qe, 20)
    fr = {I[0][i]: i for i in range(len(I[0])) if I[0][i] < len(rag_texts)}
    br_raw = sorted([(i,s) for i,s in enumerate(bm25.get_scores(tk)) if s > 0],
                    key=lambda x: -x[1])[:20]
    br = {idx: r for r, (idx, _) in enumerate(br_raw)}
    K  = 60
    rrf = {i: (1/(K+fr[i]) if i in fr else 0) + (1/(K+br[i]) if i in br else 0)
           for i in set(fr)|set(br)}
    for idx, _ in sorted(rrf.items(), key=lambda x: -x[1]):
        if idx >= len(rag_texts): continue
        txt = rag_texts[idx].strip()
        if txt and _dom_ok(txt, q):
            return _clean(txt, 380)
    return ""

# =============================================================================
# SLM fallback
# =============================================================================

def _slm_answer(question: str) -> str | None:
    with _slm_lock:
        ready = _slm_ready
        gen   = _slm_generator
        ret   = _slm_retriever
    if not ready:
        return None
    try:
        domain    = detect_domain(question)
        matches   = ret.search(question, domain=domain, top_k=6)
        engine    = get_engine(domain)
        top3      = engine.rerank(question, matches)
        extracted = engine.extract(question, top3) if top3 else None

        if top3 and top3[0].score < engine.threshold():
            extracted = None
            top3 = []

        if extracted:
            rewritten = gen.rewrite(question, extracted)
            ans = rewritten if rewritten and rewritten.lower() != "not found in corpus." else extracted
        elif top3:
            ans = gen.answer_with_context(question, top3)
        else:
            return None

        if ans and ans.lower().strip() in ("not found in corpus.", ""):
            return None
        return ans
    except Exception as e:
        print(f"[SLM] Inference error: {e}")
        return None

# =============================================================================
# FAQ keyword gate
# =============================================================================

_FAQ_KW = {
    "what is","what are","how many","when does","when is","how do i","how can i",
    "can i","is there","does sru","tell me about","explain",
    "admission","fee","fees","scholarship","hostel fee","hostel fees","hostel charges",
    "bus","transport","timing","schedule","placement","package","packages",
    "lpa","recruiter","cutoff","exam","syllabus","result","grade","cgpa","credit",
    "internship","backlog","course","program","degree","btech","mtech","mba",
    "phd","lateral","jee","eamcet","club","nss","ncc","sports","how much",
    "process","procedure","eligibility","criteria","requirement","contact",
    "email","phone","number","website","rank","merit","seat","branch","stream",
}

def _has_faq_kw(q: str) -> bool:
    return any(q.startswith(k) or k in q for k in _FAQ_KW)

# =============================================================================
# INTENT ROUTER
# =============================================================================

def _intent(query: str) -> str:
    q = _q(query)
    if _is_leadership(q): return "leadership"
    if _is_faculty(q):    return "faculty"
    if _is_map(q):        return "campus_map"
    if _has_faq_kw(q):    return "faq"
    if len(q.split()) >= 3: return "rag"
    return "unknown"

# =============================================================================
# CHAT ENDPOINT
# =============================================================================

IDK = ("I don't have enough information to answer that confidently. "
       "Please ask your guide, faculty, or campus staff. 🙂")

@app.post("/chat")
async def chat(data: dict):
    msg = data.get("message","").strip()
    if not msg:
        return {"type":"text","answer":"Please ask me something about SR University."}

    # 0. Greeting
    g = _greeting(msg)
    if g: return {"type":"text","answer": g}

    q      = _q(msg)
    intent = _intent(msg)

    # 1. Name lookup
    if intent not in ("campus_map",):
        name_hit = _name_lookup(q)
        if name_hit:
            return name_hit

    # 2. Leadership
    if intent == "leadership":
        l = _search_leader(q)
        if l:
            p = l["person"]
            return {"type":"image_list",
                    "answer": f"Here is {p['name']}, {p.get('designation','')}:",
                    "sources":[{"name":p["name"],"designation":p.get("designation",""),
                                "image_link":p.get("image_url",""),
                                "profile_link":p.get("profile_link","")}]}
        return {"type":"text","answer":"I couldn't find that leadership profile. Please check sru.edu.in."}

    # 3. Faculty
    if intent == "faculty":
        res = _search_faculty(q)
        if res:
            return {"type":"image_list","answer":f"Found {len(res)} faculty member(s):",
                    "sources": res}
        return {"type":"text","answer":"No faculty found. Try 'CSE faculty', 'ECE HOD', or a professor's name."}

    # 4. Map
    if intent == "campus_map":
        loc = _search_location(q)
        if loc:
            nb_names = _nearby(loc["x"], loc["y"])
            nearby   = [{"name":n,"x":locations_data[n]["x"],"y":locations_data[n]["y"]}
                        for n in nb_names if n in locations_data]
            return {"type":"campus_map","name":loc["name"],
                    "answer": f"📍 {loc['name']} is marked on the campus map below.",
                    "map_image":"SRUniversity.png",
                    "x":loc["x"],"y":loc["y"],"nearby":nearby}
        return {"type":"text","answer":"I couldn't find that location. Try names like 'library', 'canteen', or a block name."}

    # 5. KB lookup
    kb = kb_lookup(msg)
    if kb:
        return {"type":"text","answer": kb}

    # 6. FAQ — Exact → FAISS → Fuzzy
    faq = _search_faq(q)
    if faq:
        raw = faq.get("response", faq.get("answer",""))
        if raw: return {"type":"text","answer": _clean(raw)}

    # 7. RAG best chunk
    chunk = _best_chunk(msg)
    if chunk:
        return {"type":"text","answer": chunk}

    # 8. SLM fallback
    slm_ans = _slm_answer(msg)
    if slm_ans:
        return {"type":"text","answer": slm_ans, "source": "slm"}

    return {"type":"text","answer": IDK}


# ── SLM status endpoint ───────────────────────────────────────────────────────
@app.get("/slm-status")
async def slm_status():
    with _slm_lock:
        return {"ready": _slm_ready, "available": _SLM_DEPS}