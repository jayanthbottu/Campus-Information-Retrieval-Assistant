from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ←  edit these two lines to match your machine
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_PATH = Path(r"data\\clean_corpus.txt")
MODEL_DIR   = Path(r"models\\club-slm")
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    sys.exit(
        f"[ERROR] Missing library: {e}\n"
        "Run:  pip install torch transformers scikit-learn"
    )


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
# Text utilities  (inlined from shared.py)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(value: str) -> str:
    # Fix mojibake: multi-byte sequences that appear when UTF-8 is read as latin-1
    value = value.replace("\r", "").replace("\t", " ")
    # apostrophe / right-single-quote
    value = value.replace("\xc3\xa2\xc2\x80\xc2\x99", "'")
    value = value.replace("\xe2\x80\x99", "'")
    # left double-quote
    value = value.replace("\xc3\xa2\xc2\x80\xc2\x9c", '"')
    value = value.replace("\xe2\x80\x9c", '"')
    # right double-quote
    value = value.replace("\xc3\xa2\xc2\x80\xc2\x9d", '"')
    value = value.replace("\xe2\x80\x9d", '"')
    # en-dash / em-dash
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
        # fine-grained keyword boosts
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

    # shared helper: pull first regex match from top-N chunks
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

        # full fee structure extraction
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

        # Faculty mentor
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

        # Named officer roles
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

        # Responsibilities
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

        # Objective
        if "objective" in ql:
            for m in ms[:3]:
                ht = " ".join(m.headings).lower()
                if "introduction" in ht or "objective" in m.text.lower():
                    found = re.search(r"(Objective:\s*[^.]+(?:\.[^.]+)?)", m.text, re.IGNORECASE)
                    if found:
                        return self.clean(found.group(1).strip())
                    return self.clean(compact_answer(m.text, max_sentences=2))

        # Recruitment
        if "recruitment" in ql:
            for m in ms[:3]:
                if "recruitment" in " ".join(m.headings).lower():
                    return self.clean(compact_answer(m.text, max_sentences=3))

        # Executive committee / roles list
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
        model_dir = Path(model_dir).resolve()
        if not model_dir.exists():
            sys.exit(
                f"[ERROR] Model not found: {model_dir}\n"
                f"Edit MODEL_DIR at the top of this file, or run train_slm.py first."
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


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(question: str, debug: bool = False, top_k: int = 6) -> str:
    # validate paths
    if not CORPUS_PATH.exists():
        sys.exit(f"[ERROR] Corpus not found: {CORPUS_PATH}\nEdit CORPUS_PATH at the top of ask.py")

    generator = Generator(MODEL_DIR)
    domain    = detect_domain(question)
    retriever = Retriever(CORPUS_PATH)
    matches   = retriever.search(question, domain=domain, top_k=top_k)
    engine    = get_engine(domain)

    top3      = engine.rerank(question, matches)
    extracted = engine.extract(question, top3) if top3 else None

    # threshold guard – don't hallucinate on weak matches
    if top3 and top3[0].score < engine.threshold():
        extracted = None
        top3 = []

    if extracted:
        rewritten = generator.rewrite(question, extracted)
        answer = rewritten if rewritten and rewritten.lower() != "not found in corpus." else extracted
    elif top3:
        answer = generator.answer_with_context(question, top3)
    else:
        answer = "Not found in corpus."

    if debug:
        print(json.dumps({
            "question": question,
            "domain":   domain,
            "answer":   answer,
            "matches":  [{"chunk_id": m.chunk_id, "score": round(m.score, 4),
                          "headings": m.headings, "text": m.text} for m in matches],
        }, indent=2, ensure_ascii=False))
    return answer


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="University corpus QA  –  standalone")
    ap.add_argument("question",           help="Question to ask")
    ap.add_argument("--top-k",  type=int, default=6,     help="Number of chunks to retrieve")
    ap.add_argument("--debug",  action="store_true",     help="Print full JSON debug output")
    args = ap.parse_args()

    answer = run(args.question, debug=args.debug, top_k=args.top_k)
    if not args.debug:
        print(answer)


if __name__ == "__main__":
    main()