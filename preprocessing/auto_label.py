"""
auto_label.py
=============
Automatically labels cleaned complaints with:
  - department   (Water Supply, Engineering, SWM, Health, etc.)
  - category     (Water Shortage, Road Pothole, Garbage Not Collected, etc.)
  - language     (en, hi, hinglish)
  - severity     (HIGH, MEDIUM, LOW) based on urgency keywords
  - confidence   (0.0 – 1.0) of the label assignment

Strategy:
  1. Keyword-based rule matching (fast, transparent)
  2. Conflict resolution (multiple departments match → pick best)
  3. Low-confidence records flagged for manual review

Input:  data/processed/complaints_all.jsonl
Output:
  data/processed/complaints_labeled.csv      ← high-confidence labeled
  data/processed/complaints_review.csv       ← needs manual review

Run: python preprocessing/auto_label.py
"""

import csv
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IN_FILE = "data/processed/complaints_all.jsonl"
OUT_LABELED = "data/processed/complaints_labeled.csv"
OUT_REVIEW = "data/processed/complaints_review.csv"
OUT_DIR = "data/processed"

# ─────────────────────────────────────────────────────────────────────────────
# DEPARTMENT & CATEGORY KEYWORD MAP
# Format: department → {category → [keywords (EN + Hinglish + Devanagari)]}
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "Water Supply": {
        "Water Shortage": [
            "water", "paani", "पानी", "no water", "water not coming",
            "water supply", "nali", "nalka", "tap", "borewell",
            "water cut", "paani nahi", "जल", "पेयजल", "paani band",
            "water problem", "pani ki problem", "water shortage",
        ],
        "Water Leakage": [
            "water leak", "pipe burst", "pipeline broke", "paani behna",
            "nali toot", "leakage", "flooding road water", "main burst",
            "pipe leakage", "water wastage", "paani phoot",
        ],
        "Contaminated Water": [
            "dirty water", "contaminated", "colour water", "smelly water",
            "ganda paani", "गंदा पानी", "yellow water", "muddy water",
            "brown water", "impure water", "water quality",
        ],
    },
    "Engineering": {
        "Road Pothole": [
            "pothole", "khaDDa", "road damage", "road repair",
            "broken road", "sadak toot", "बड़ा गड्ढा", "गड्ढा",
            "road bad", "road condition", "road hole", "road problem",
            "khaDDe sadak", "road not repaired", "road digging",
        ],
        "Road Construction": [
            "road construction", "tarring", "road work", "naya road",
            "new road", "road laying", "bitumen", "asphalt need",
        ],
        "Street Light": [
            "street light", "light not working", "dark road", "light band",
            "bijli light", "pole light", "street lamp", "andhera",
            "अंधेरा", "streetlight not working", "light out",
        ],
        "Drainage": [
            "drain", "nullah", "drainage", "choked drain", "sewage line",
            "manhole open", "overflow drain", "nali choke", "नाली",
            "drain overflow", "water logging", "flooding", "barsat paani",
        ],
    },
    "SWM": {
        "Garbage Not Collected": [
            "garbage", "kachra", "कचरा", "waste", "dustbin", "garbage not",
            "garbage pickup", "kachra nahi", "कचरा नहीं", "sweeping",
            "solid waste", "trash", "rubbish", "litter", "cleaning not",
            "kachrawala nahi", "garbage vehicle", "garbage man",
        ],
        "Illegal Dumping": [
            "dumping", "garbage dump", "illegal dump", "garbage everywhere",
            "roadside garbage", "open garbage", "waste dumped",
        ],
        "Stray Animals": [
            "stray dog", "dogs biting", "dog menace", "awaara kutte",
            "आवारा कुत्ते", "stray cattle", "cow road", "bull attack",
        ],
    },
    "Public Health": {
        "Dengue/Mosquito": [
            "dengue", "mosquito", "malaria", "mosquitoes", "fogging",
            "larva", "mosquito breeding", "stagnant water", "मच्छर",
            "डेंगू", "मलेरिया", "mosquito bite",
        ],
        "Food Safety": [
            "food poison", "bad food", "hotel dirty", "stale food",
            "food adulteration", "food quality", "खाद्य", "restaurant dirty",
        ],
        "Sanitation": [
            "toilet", "open defecation", "OD", "sanitation", "washroom",
            "shauchalay", "शौचालय", "public toilet", "clean toilet",
        ],
    },
    "Revenue/Tax": {
        "Property Tax": [
            "property tax", "house tax", "tax notice", "tax bill",
            "tax payment", "tax receipt", "assessment", "demand notice",
            "sampatti kar", "संपत्ति कर", "property assessment",
        ],
        "Mutation": [
            "mutation", "name transfer", "ownership change", "khata",
            "property transfer", "registration",
        ],
    },
    "Town Planning": {
        "Illegal Construction": [
            "illegal construction", "unauthorized building", "encroachment",
            "demolition", "illegal structure", "अवैध निर्माण",
            "building violation", "construction without permission",
        ],
        "Building Permission": [
            "building permission", "plan approval", "FSI", "construction",
            "commencement certificate", "OC", "occupation certificate",
            "CC letter", "plan sanction",
        ],
    },
    "Registration": {
        "Birth Certificate": [
            "birth certificate", "janma praman patra", "जन्म प्रमाण",
            "birth registration", "child registration",
        ],
        "Death Certificate": [
            "death certificate", "mrityu praman patra", "मृत्यु प्रमाण",
            "death registration",
        ],
    },
    "Licensing": {
        "Trade License": [
            "trade license", "shop license", "business license",
            "व्यापार लाइसेंस", "dukan license", "noc",
        ],
    },
    "Fire Services": {
        "Fire Incident": [
            "fire", "aag", "आग", "burning", "caught fire", "smoke",
            "fire brigade", "fire station", "cylinder blast",
        ],
        "Fire NOC": [
            "fire noc", "fire safety", "fire certificate",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────
HIGH_SEVERITY = [
    "urgent", "emergency", "asap", "immediately", "serious", "jaldi",
    "जरूरी", "crisis", "danger", "burst", "fire", "death",
    "accident", "flood", "आग", "tatkaal", "तत्काल",
]
LOW_SEVERITY = [
    "request", "please", "kindly", "please look into",
    "baad mein", "whenever possible", "if possible",
]


def score_record(text: str) -> tuple[str, str, float]:
    """
    Returns (department, category, confidence).
    Confidence is fraction of matched keywords.
    """
    text_lower = text.lower()
    scores = {}  # (dept, cat) → match_count

    for dept, categories in LABEL_MAP.items():
        for cat, keywords in categories.items():
            count = 0
            for kw in keywords:
                if kw.lower() in text_lower:
                    count += 1
            if count > 0:
                scores[(dept, cat)] = count

    if not scores:
        return "Unknown", "Unknown", 0.0

    # Pick best match
    best = max(scores, key=lambda k: scores[k])
    best_count = scores[best]

    # Confidence: how many keywords matched vs total for that category
    total_kw = len(LABEL_MAP[best[0]][best[1]])
    confidence = min(best_count / max(total_kw * 0.2, 1), 1.0)  # normalized

    # Penalize if multiple departments match (ambiguous)
    dept_matches = len(set(d for d, c in scores.keys()))
    if dept_matches > 2:
        confidence *= 0.7

    return best[0], best[1], round(confidence, 3)


def get_severity(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in HIGH_SEVERITY):
        return "HIGH"
    elif any(kw in text_lower for kw in LOW_SEVERITY):
        return "LOW"
    return "MEDIUM"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(IN_FILE):
        log.error("Input file not found: %s — run clean_pipeline.py first", IN_FILE)
        return

    labeled = []
    review = []

    with open(IN_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
            except Exception:
                continue

            text = r.get("text", "").strip()
            if not text:
                continue

            dept, cat, conf = score_record(text)
            severity = get_severity(text)
            language = r.get("language", "en")

            record = {
                "id": i,
                "text": text,
                "clean_text": re.sub(r"\s+", " ", text).strip(),
                "department": dept,
                "category": cat,
                "severity": severity,
                "language": language,
                "confidence": conf,
                "source": r.get("source", ""),
            }

            # High confidence → labeled dataset; low → review
            if conf >= 0.3 and dept != "Unknown":
                labeled.append(record)
            else:
                review.append(record)

    log.info("Labeled (auto): %d records", len(labeled))
    log.info("Need review:    %d records", len(review))

    # CSV columns
    fieldnames = ["id", "text", "clean_text", "department", "category",
                  "severity", "language", "confidence", "source"]

    with open(OUT_LABELED, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)
    log.info("✓ Saved labeled dataset → %s", OUT_LABELED)

    with open(OUT_REVIEW, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review)
    log.info("✓ Saved review dataset  → %s", OUT_REVIEW)

    # Label distribution
    dept_dist = {}
    for r in labeled:
        d = r["department"]
        dept_dist[d] = dept_dist.get(d, 0) + 1
    log.info("Department distribution: %s", dept_dist)

    lang_dist = {}
    for r in labeled:
        l = r["language"]
        lang_dist[l] = lang_dist.get(l, 0) + 1
    log.info("Language distribution: %s", lang_dist)


if __name__ == "__main__":
    main()
