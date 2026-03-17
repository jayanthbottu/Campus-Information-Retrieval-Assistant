import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

INPUT_FILE = "faculty.json"
OUTPUT_FILE = "clean_faculty.json"

# -------------------------------------------------------
# CANONICAL DEPARTMENT MAPPING
# -------------------------------------------------------

DEPT_MAPPING = {
    "cse": "Computer Science and Engineering",
    "computer science & engineering": "Computer Science and Engineering",
    "computer science engineering": "Computer Science and Engineering",

    "cs & ai": "Computer Science and Artificial Intelligence",
    "csai": "Computer Science and Artificial Intelligence",
    "computer science and ai": "Computer Science and Artificial Intelligence",

    "ece": "Electronics and Communication Engineering",
    "electronics & communication engineering": "Electronics and Communication Engineering",

    "electrical & electronics engineering": "Electrical and Electronics Engineering",
    "electrical and electronics engineering": "Electrical and Electronics Engineering",

    "department of physics": "Physics",
    "physics": "Physics",

    "department of chemistry": "Chemistry",
    "chemistry": "Chemistry",

    "department of civil engineering": "Civil Engineering",
    "civil engineering": "Civil Engineering",

    "school of business": "Business",
    "school of sciences & humanities": "Sciences and Humanities",
    "school of sciences and humanities": "Sciences and Humanities"
}


# -------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------

def clean_text(text):
    if not text:
        return None
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def title_case(text):
    if not text:
        return None
    return text.title()


def generate_slug(name):
    name = name.lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "-", name.strip())
    return name


def canonical_department(dept):
    if not dept:
        return None
    d = dept.lower().strip()
    return DEPT_MAPPING.get(d, title_case(clean_text(dept)))


def split_research(text):
    if not text:
        return []

    if isinstance(text, list):
        items = text
    else:
        items = re.split(r",|;|/", text)

    cleaned = []
    for item in items:
        item = clean_text(item)
        if not item:
            continue

        # remove truncated endings
        if item.endswith("..."):
            continue

        if len(item) < 3:
            continue

        cleaned.append(title_case(item))

    return list(dict.fromkeys(cleaned))


def extract_highest_degree(entry):
    degree_block = {
        "degree": None,
        "specialization": None,
        "institution": None,
        "country": None,
        "year": None
    }

    if "highest_degree" in entry and isinstance(entry["highest_degree"], dict):
        deg = entry["highest_degree"]
        degree_block["degree"] = clean_text(deg.get("degree"))
        degree_block["specialization"] = clean_text(deg.get("specialization"))
        degree_block["institution"] = clean_text(deg.get("institution"))
        degree_block["country"] = clean_text(deg.get("country"))
        degree_block["year"] = clean_text(deg.get("year"))

    elif "education" in entry:
        edu = clean_text(entry["education"])
        if edu and "ph" in edu.lower():
            degree_block["degree"] = "PhD"

            inst_match = re.search(r"at (.*)", edu, re.IGNORECASE)
            if inst_match:
                degree_block["institution"] = clean_text(inst_match.group(1))

    return degree_block


# -------------------------------------------------------
# MAIN CLEANING PIPELINE
# -------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned = []
seen_names = set()

for entry in raw_data:

    name = clean_text(entry.get("name"))
    if not name:
        continue

    normalized_name = title_case(name)

    if normalized_name in seen_names:
        continue
    seen_names.add(normalized_name)

    department = canonical_department(entry.get("department"))

    research_raw = (
        entry.get("research_areas") or
        entry.get("research_interests") or
        entry.get("research_interest")
    )

    research_areas = split_research(research_raw)

    degree_block = extract_highest_degree(entry)

    cleaned_entry = {
        "id": generate_slug(normalized_name),
        "name": normalized_name,
        "designation": clean_text(entry.get("designation")),
        "department": department,
        "school": title_case(clean_text(entry.get("school"))),
        "highest_degree": degree_block,
        "research_areas": research_areas,
        "contact": {
            "email": clean_text(
                entry.get("email") or
                (entry.get("contact") or {}).get("email")
            ),
            "phone": clean_text(
                entry.get("phone") or
                (entry.get("contact") or {}).get("phone")
            )
        },
        "profile_link": clean_text(
            entry.get("profile_link") or entry.get("profile_url")
        ),
        "image_link": clean_text(
            entry.get("image_link") or entry.get("image_url")
        ),
        "source": "SR University Official Website"
    }

    cleaned.append(cleaned_entry)


# -------------------------------------------------------
# FINAL VALIDATION CHECK
# -------------------------------------------------------

for obj in cleaned:
    required_keys = [
        "id", "name", "designation", "department",
        "school", "highest_degree", "research_areas",
        "contact", "profile_link", "image_link", "source"
    ]
    for key in required_keys:
        if key not in obj:
            raise ValueError(f"Missing key {key} in {obj['name']}")

# -------------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print("Production-ready faculty JSON generated successfully.")