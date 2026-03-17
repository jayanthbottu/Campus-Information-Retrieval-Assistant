# ============================================================
# SRU_KB.PY  —  SR University Knowledge Base
# Built directly from official corpus: SR_University_Handbook_RAG_2026.pdf
# and other uploaded documents. Every fact is sourced from the corpus.
# Format: each entry has "answer" (clean, direct) and "keywords" (trigger phrases)
# ============================================================

KB = [

    # ── ABOUT SRU ──────────────────────────────────────────────────────────────

    {
        "answer": (
            "SR University is the first private university established in Telangana. "
            "It is located at Ananthasagar, Hasanparthy, Warangal – 506371, Telangana. "
            "It is governed by Sri Rajeshwara Educational Society, a 50-year-old conglomerate "
            "that oversees 95 institutions with 90,000+ students and 17,000 staff."
        ),
        "keywords": [
            "about sru", "about sr university", "what is sr university", "what is sru",
            "history of sru", "background of sru", "tell me about sru", "sr university overview",
            "who founded sru", "when was sru founded", "first private university telangana",
            "sru history",
        ],
    },

    {
        "answer": (
            "SR University is ranked 91st in NIRF India Rankings, 7th among private universities "
            "in India in THE World University Rankings 2026 (global band 801–1000), 23rd in "
            "UI GreenMetric World University Rankings, and 50th in THE Impact Rankings. "
            "It holds NBA Tier-1 accreditation and is UGC approved."
        ),
        "keywords": [
            "ranking", "rankings", "nirf rank", "nirf ranking", "the ranking",
            "world ranking", "sru rank", "sru ranking", "nba accreditation",
            "ugc approved", "accreditation", "naac", "recognized", "approved",
            "nirf", "private university ranking", "india ranking",
        ],
    },

    {
        "answer": (
            "SR University has over 10,000 students from 16 countries, 80+ global academic "
            "tie-ups, 150+ recruiters, and a 97% employment rate. The EAMCET code is SRHP."
        ),
        "keywords": [
            "how many students", "student strength", "total students", "eamcet code",
            "sru eamcet code", "students from other countries", "international students",
            "global tie ups", "tie ups", "employment rate",
        ],
    },

    # ── FEES ───────────────────────────────────────────────────────────────────

    {
        "answer": (
            "B.Tech CSE fee at SRU: ₹1,47,500/semester (Year 1), ₹1,55,000 (Year 2), "
            "₹1,63,000 (Year 3), ₹1,71,000 (Year 4). One-time Enrollment Fee ₹5,000 "
            "and Admission Fee ₹25,000 also apply. Specialisation fee ₹30,000/year from Year 2."
        ),
        "keywords": [
            "cse fee", "btech cse fee", "computer science fee", "cse fees",
            "fee structure", "fee per semester", "semester fee", "btech fee",
            "fees", "tuition fee", "annual fee", "fee structure btech",
            "how much fee", "btech fees", "total fee", "college fee",
        ],
    },

    {
        "answer": (
            "B.Tech ECE fee: ₹1,20,000/semester. One-time Enrollment Fee ₹5,000 and "
            "Admission Fee ₹25,000. Specialisation fee ₹30,000/year from Year 2 for "
            "AI & ML or VLSI Design specialisations."
        ),
        "keywords": [
            "ece fee", "btech ece fee", "electronics fee", "ece fees",
            "ece fee structure", "fee for ece",
        ],
    },

    {
        "answer": (
            "B.Tech EEE, Mechanical, and Civil fee: ₹1,00,000/semester. One-time Enrollment Fee "
            "₹5,000 and Admission Fee ₹25,000. Specialisation fee ₹30,000/year from Year 2."
        ),
        "keywords": [
            "eee fee", "mechanical fee", "civil fee", "eee fees", "mech fee",
            "btech eee fee", "civil engineering fee", "mechanical engineering fee",
        ],
    },

    {
        "answer": (
            "B.Tech CSE scholarship: 20% for EAMCET rank 16001–18000 (fee ₹1,18,000), "
            "30% for rank 15001–16000 (₹1,04,000), 40% for rank 12001–15000 (₹89,000), "
            "50% for rank 10001–12000 (₹74,000), 60% for rank 1–10000 (₹60,000). "
            "JEE Mains and CBSE/State Board percentages have corresponding slabs."
        ),
        "keywords": [
            "scholarship", "scholarships", "fee waiver", "merit scholarship",
            "scholarship cse", "scholarship for eamcet", "scholarship for jee",
            "scholarship structure", "how much scholarship", "discount on fees",
            "eamcet scholarship", "jee scholarship",
        ],
    },

    {
        "answer": (
            "SRU offers a First Arrival Admission Benefit: confirm admission before 28 Feb 2026 "
            "and the one-time Admission Fee of ₹25,000 is completely waived off. "
            "Confirming before 30 Nov 2025 also gave an extra 10% scholarship on tuition."
        ),
        "keywords": [
            "first arrival", "admission benefit", "early admission", "admission fee waiver",
            "waiver", "25000 waiver", "first arrival benefit",
        ],
    },

    {
        "answer": (
            "Fee payment can be made online at https://sru.edu.in/online_payment. "
            "You can also pay at the accounts office on campus."
        ),
        "keywords": [
            "fee payment", "pay fee", "pay fees", "how to pay fee",
            "online fee payment", "fee payment link", "pay online",
        ],
    },

    # ── ADMISSIONS ─────────────────────────────────────────────────────────────

    {
        "answer": (
            "Apply for UG/PG admission at https://sru.edu.in/admission/. "
            "International students apply at https://sru.edu.in/international. "
            "Ph.D. (Chancellor Fellowship) at https://sraap.in/cphd/, "
            "other Ph.D. at https://sraap.in/phd/."
        ),
        "keywords": [
            "how to apply", "apply for admission", "admission portal", "application portal",
            "admission link", "apply online", "admission form", "apply",
            "how to get admission", "admission process", "join sru", "admission procedure",
            "admission 2026", "admission 2025",
        ],
    },

    {
        "answer": (
            "B.Tech eligibility: Telangana students need 55% in 10+2 with PCM; "
            "other state students need 50% in 10+2 with PCM. "
            "Must qualify in SRSAT, JEE Main, EAPCET/EAMCET, or equivalent state entrance exam, "
            "or have merit in Sports/Cultural activities."
        ),
        "keywords": [
            "eligibility", "btech eligibility", "admission eligibility", "minimum marks",
            "eligibility criteria", "10+2 marks", "pcm marks", "12th marks",
            "qualification for btech", "criteria for admission",
        ],
    },

    {
        "answer": (
            "SRU accepts: SRSAT (own entrance test), JEE Main, EAPCET/EAMCET (code: SRHP), "
            "state-level engineering entrance exams, SAT (for international students), "
            "IB scores, and merit in Sports or Cultural activities."
        ),
        "keywords": [
            "entrance exam", "entrance examination", "which exam", "accepted exams",
            "jee", "eamcet", "srsat", "sat", "entrance test", "what exam for sru",
        ],
    },

    {
        "answer": (
            "For B.Tech admissions, contact: 833 100 3030, 833 100 4040, "
            "837 403 9180, 837 403 9104. Email: info@sru.edu.in."
        ),
        "keywords": [
            "admission contact", "admission helpline", "admission phone", "admission number",
            "contact for admission", "admission enquiry", "admission office contact",
        ],
    },

    {
        "answer": (
            "MBA eligibility: 50% in any graduation degree. "
            "M.Tech eligibility: 55% in B.Tech in the corresponding discipline. "
            "MCA eligibility: graduation with 55% marks. "
            "Ph.D. eligibility: Master's degree with 55% or 6.0 CGPA."
        ),
        "keywords": [
            "mba eligibility", "mtech eligibility", "mca eligibility", "pg eligibility",
            "postgraduate eligibility", "masters eligibility",
        ],
    },

    {
        "answer": (
            "Ph.D. admissions 2025 cycle: last date was 14 Nov 2025; online exam and "
            "interview on 24–25 Nov 2025. Chancellor Fellowship (CSE/AI only): ₹1,00,000/month. "
            "Regular Fellowship (all disciplines): ₹40,000/month. Apply at https://sraap.in/phd/."
        ),
        "keywords": [
            "phd", "phd admission", "phd eligibility", "phd fellowship", "phd stipend",
            "chancellor fellowship", "research program", "doctorate", "phd dates",
        ],
    },

    # ── PROGRAMS / COURSES ─────────────────────────────────────────────────────

    {
        "answer": (
            "SRU offers B.Tech in CSE (with specialisations in AI & ML, Cyber Security, "
            "Data Science, Gaming & ARVR, Robotics & Automation), ECE, EEE, Mechanical, Civil. "
            "Also BBA, MBA, MCA, B.Sc. (Hons.), B.Sc. Agriculture, M.Tech, M.Sc., and Ph.D. programs."
        ),
        "keywords": [
            "courses offered", "programs offered", "what courses", "all courses",
            "list of courses", "available programs", "branches", "btech branches",
            "what branches", "engineering branches", "departments", "available branches",
            "streams", "specialisations", "cse specialisations",
        ],
    },

    {
        "answer": (
            "CSE specialisations: Artificial Intelligence and Machine Learning, Cyber Security, "
            "Data Science, Gaming and ARVR, and Robotics and Automation."
        ),
        "keywords": [
            "cse specialisation", "cse specializations", "ai ml specialisation",
            "cyber security course", "data science specialisation", "gaming arvr",
            "robotics specialisation", "cse streams",
        ],
    },

    {
        "answer": (
            "The International Engineering Program (IEP) requires 70% in 10+2 with PCM. "
            "The 2+2 Study Abroad program with University of Melbourne: 2 years at SRU + "
            "2 years at Melbourne. Tuition ₹2,50,000/year, hostel ₹1,00,000/year at SRU."
        ),
        "keywords": [
            "iep", "international engineering program", "2+2 program", "melbourne program",
            "study abroad", "university of melbourne", "international program",
        ],
    },

    {
        "answer": (
            "Minor Degree: earn 16 additional credits from a different discipline. "
            "Eligible if CGPA ≥ 7.0 after 4th semester. Fee: ₹20,000/year (₹10,000 × 2 installments). "
            "Students with CGPA ≥ 7.5 get ₹5,000 scholarship, reducing fee to ₹15,000."
        ),
        "keywords": [
            "minor", "minor degree", "minor program", "minor eligibility", "minor fee",
            "minor subjects", "how to do minor", "secondary degree",
        ],
    },

    {
        "answer": (
            "Branch change is available for B.Tech students after Year 1, subject to academic "
            "performance, seat availability, and university policy. Contact the Dean Academics Office."
        ),
        "keywords": [
            "branch change", "change branch", "switch branch", "transfer branch",
        ],
    },

    # ── PLACEMENTS ─────────────────────────────────────────────────────────────

    {
        "answer": (
            "2024–25 placement highlights: highest package 51 LPA (CISCO), "
            "top 10-percentile average 21 LPA, overall average 6.5 LPA, "
            "97% employment rate, 1,200+ job offers, 150+ recruiters."
        ),
        "keywords": [
            "placement", "placements", "placement packages", "highest package",
            "average package", "placement statistics", "how many lpa", "package",
            "packages", "ctc", "lpa", "salary", "placement record", "placement 2025",
            "placement 2024", "max package", "top package",
        ],
    },

    {
        "answer": (
            "Top recruiters at SRU (2024–25): CISCO (51 LPA), Salesforce (44 LPA), "
            "ServiceNow (44 LPA), PayPal (34.5 LPA), Accenture (380 offers), "
            "Capgemini (48 offers), Infosys (65 offers), Cognizant (41 offers), "
            "Flipkart, Microsoft, Amazon, and 150+ more companies."
        ),
        "keywords": [
            "recruiters", "companies", "top companies", "which companies visit",
            "company list", "top recruiters", "mnc", "it companies", "who recruits",
            "placement companies", "cisco", "accenture", "infosys recruiter",
        ],
    },

    {
        "answer": (
            "SRU's placement cell provides pre-placement training, soft skills development, "
            "mock interviews, and aptitude preparation. Both on-campus and off-campus "
            "placements are supported. International placement opportunities are also facilitated."
        ),
        "keywords": [
            "placement process", "placement preparation", "placement training",
            "placement cell", "how placement works", "campus recruitment",
            "placement support", "placement drive", "how to get placed",
        ],
    },

    {
        "answer": (
            "Highest internship stipend: ₹1,25,000/month offered by CISCO (2024–25 batch). "
            "SRiX facilitates 150+ internships annually through the incubation centre."
        ),
        "keywords": [
            "internship", "internship stipend", "internship salary", "highest internship",
            "internship package", "cisco internship", "internship at sru",
        ],
    },

    # ── HOSTEL ─────────────────────────────────────────────────────────────────

    {
        "answer": (
            "SRU hostels are air-conditioned with Wi-Fi in every room. Facilities include "
            "TV Room, Visitor Rooms, Game Room, Study Room, 24/7 medical facility, gymnasium, "
            "and USB charging outlets. Both boys and girls hostels are available on campus."
        ),
        "keywords": [
            "hostel", "hostel facilities", "hostel amenities", "boys hostel", "girls hostel",
            "hostel rooms", "hostel wifi", "hostel ac", "about hostel", "hostel info",
            "hostel details", "hostel information", "hostel accommodation",
        ],
    },

    # ── SPORTS ─────────────────────────────────────────────────────────────────

    {
        "answer": (
            "SRU sports facilities: Cricket, Basketball, Football, Volleyball, Throwball, "
            "Kho-Kho, Kabaddi, Table Tennis, Badminton, Chess, Carrom, and a Gymnasium. "
            "SRU hosts the SR Cricket Champions Trophy annually for 16+ consecutive years."
        ),
        "keywords": [
            "sports", "sports facilities", "games", "sports complex", "cricket",
            "football", "basketball", "gymnasium", "gym", "sports ground",
            "indoor games", "outdoor sports", "sports at sru",
        ],
    },

    # ── CLUBS ──────────────────────────────────────────────────────────────────

    {
        "answer": (
            "SRU student clubs include: Coding Club, IEEE Computer Society, ACM Chapter, "
            "CSI Chapter, GDSC, IEEE WIE, Garden Club, Yoga Club, Media & Photography Club, "
            "Painting & Sketching Club, Dance & Music Club, Drama & Theatre Club, "
            "NSS, NCC, Master Communicators Club, Martial Arts Club, and Sankhya (Maths) Club."
        ),
        "keywords": [
            "clubs", "student clubs", "club list", "what clubs", "technical clubs",
            "extracurricular", "activities", "club activities", "student activities",
        ],
    },

    {
        "answer": (
            "Coding Club coordinator: Dr. P. Pramod Kumar. "
            "IEEE Computer Society coordinator: Dr. Mohammad Ali. "
            "ACM Chapter coordinator: Dr. Raghvendra Kishore Singh. "
            "CSI Chapter coordinator: Ms. Neelima Gurrapu. "
            "GDSC coordinator: Dr. Rajchandar K. "
            "IEEE WIE coordinator: Ms. Faiza Iram."
        ),
        "keywords": [
            "club coordinator", "coding club coordinator", "ieee coordinator",
            "acm coordinator", "csi coordinator", "gdsc coordinator",
            "club faculty", "club mentor",
        ],
    },

    {
        "answer": (
            "NSS (National Service Scheme) and NCC (National Cadet Corps) are both active at SRU. "
            "Register through the student affairs office at the start of each academic year."
        ),
        "keywords": [
            "nss", "ncc", "national service scheme", "national cadet corps",
            "join nss", "join ncc", "social service", "cadet",
        ],
    },

    # ── ACADEMICS / EXAMS ──────────────────────────────────────────────────────

    {
        "answer": (
            "Minimum attendance to sit for mid exams is 60% per subject. "
            "A grace of up to 5% may be granted for documented medical emergencies (min 55%). "
            "Students below 55% may be permitted only by the School Head or Dean."
        ),
        "keywords": [
            "attendance", "minimum attendance", "attendance policy", "attendance percentage",
            "attendance for exam", "how much attendance", "attendance rule",
        ],
    },

    {
        "answer": (
            "Mid exam rules: arrive 15 minutes early, carry ID card, no mobile phones or "
            "electronic devices. Late entry only within first 15 minutes. "
            "Students can leave only after 30 minutes from start."
        ),
        "keywords": [
            "mid exam", "mid exam rules", "exam rules", "exam guidelines",
            "examination rules", "exam policy", "exam conduct",
        ],
    },

    {
        "answer": (
            "Examination process: register for end-semester exams after paying tuition. "
            "Pass mark is 40. Grading: O, A, B, C, D. "
            "Students with backlogs can appear for supplementary exams. "
            "Summer semester available for grade improvement (B to D grades)."
        ),
        "keywords": [
            "exam process", "examination process", "pass mark", "grading",
            "grade", "cgpa", "how grading works", "semester exam",
            "backlog", "supplementary exam", "grade improvement", "summer semester",
        ],
    },

    {
        "answer": (
            "Academic calendar for Even Semester 2025-26 is available on the SRU website. "
            "Contact the exam branch at https://sru.edu.in/examination_branch."
        ),
        "keywords": [
            "academic calendar", "exam dates", "semester dates", "exam schedule",
            "when exams", "exam timetable", "calendar",
        ],
    },

    # ── RESEARCH ───────────────────────────────────────────────────────────────

    {
        "answer": (
            "SRU research stats: 4,700+ publications, 850+ patents filed, 50+ sponsored projects, "
            "₹39 Crore+ in research funding. Collaborates with Missouri S&T (USA) and has "
            "interdisciplinary research centres across technology, engineering, and agriculture."
        ),
        "keywords": [
            "research", "research at sru", "publications", "patents", "research funding",
            "sponsored research", "research stats", "research overview",
        ],
    },

    # ── SRIX ───────────────────────────────────────────────────────────────────

    {
        "answer": (
            "SRiX (SR Innovation Exchange) is SRU's DST-recognised Technology Business Incubator. "
            "It has supported 171+ startups, created 741+ jobs, and funded 103 startups with "
            "₹13.57 Crore. Startups have collectively raised ₹76 Crore from VCs and angel investors."
        ),
        "keywords": [
            "srix", "incubator", "sr innovation exchange", "startup", "incubation",
            "startup support", "srix startups", "innovation exchange",
        ],
    },

    {
        "answer": (
            "SRiX programs: NIDHI PRAYAS (proof-of-concept), NIDHI SSS (seed funding), "
            "NIDHI EiR (entrepreneur-in-residence), NIDHI Accelerator, TIDE 2.0 (MeitY), "
            "Startup India Seed Fund. Pre-Incubation, InnovationX, SPRINT, and Women Entrepreneurship programs also available."
        ),
        "keywords": [
            "srix programs", "incubation programs", "funding programs", "nidhi",
            "startup funding", "tide 2.0", "seed fund", "women entrepreneurship",
            "sprint program", "innovationx",
        ],
    },

    # ── INTERNATIONAL ──────────────────────────────────────────────────────────

    {
        "answer": (
            "SRU has 80+ international academic collaborations with universities in USA, "
            "Australia, France, Spain, Iraq, Taiwan, Vietnam, and others. Key partners: "
            "University of Massachusetts Lowell, University of Melbourne, Missouri S&T, "
            "FPT University (Vietnam), and more."
        ),
        "keywords": [
            "international collaborations", "international partners", "global tie ups",
            "foreign university", "study abroad", "international", "partner universities",
        ],
    },

    # ── CONTACT ────────────────────────────────────────────────────────────────

    {
        "answer": (
            "SR University contact: Phone: 0870-281-8333 / 0870-281-8311. "
            "Admission helpline: 833 100 3030, 833 100 4040, 837 403 9180, 837 403 9104. "
            "Email: info@sru.edu.in. Website: https://sru.edu.in. "
            "Address: Ananthasagar, Hasanparthy, Warangal – 506371, Telangana."
        ),
        "keywords": [
            "contact", "phone number", "contact number", "email", "helpline",
            "university phone", "university email", "how to contact", "contact sru",
            "sru phone", "sru email", "reach sru", "address", "sru address",
            "university address", "where is sru", "location",
        ],
    },

    {
        "answer": (
            "Key SRU portals: Main site: sru.edu.in | Admissions: sru.edu.in/admission | "
            "Fee payment: sru.edu.in/online_payment | Ph.D. portal: sraap.in | "
            "Library: sru.edu.in/library | Alumni: alumni.sru.edu.in | SRiX: srix.in"
        ),
        "keywords": [
            "website", "portal", "sru website", "student portal", "online portal",
            "library portal", "alumni portal", "fee portal",
        ],
    },

]


# ============================================================
# Flat keyword → answer index  (built once at import)
# ============================================================

_IDX: dict[str, str] = {}
for _entry in KB:
    for _kw in _entry["keywords"]:
        _IDX[_kw.lower().strip()] = _entry["answer"]


def kb_lookup(query: str) -> str | None:
    """
    Three-pass lookup:
    1. Exact key match
    2. Key is substring of query (longest wins)
    3. All words in key are present in query (longest wins)
    Returns clean answer string or None.
    """
    q = query.lower().strip().rstrip("?!.")

    # Pass 1 — exact
    if q in _IDX:
        return _IDX[q]

    # Pass 2 — key substring of query (prefer longest key)
    best_ans, best_len = None, 0
    for key, ans in _IDX.items():
        if key in q and len(key) > best_len and len(key) >= 4:
            best_len, best_ans = len(key), ans

    if best_ans:
        return best_ans

    # Pass 3 — all key words present in query (prefer longest key)
    q_words = set(q.split())
    best_ans, best_len = None, 0
    for key, ans in _IDX.items():
        kw = set(key.split())
        if len(kw) >= 2 and kw.issubset(q_words) and len(key) > best_len:
            best_len, best_ans = len(key), ans

    return best_ans