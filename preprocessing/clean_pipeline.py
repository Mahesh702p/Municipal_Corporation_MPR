"""
clean_pipeline.py
=================
Unified data cleaning and normalization pipeline for all scraped data.

Steps:
  1. Load all JSONL/CSV files from raw/ folders
  2. Merge into one stream
  3. Deduplicate (exact + near-duplicate via MinHash)
  4. Language detection (English / Hindi / Marathi / Hinglish)
  5. Length filtering (5–150 words)
  6. PII masking (phone, email, Aadhaar)
  7. Text normalization (via abctokz DevanagariNormalizer)
  8. Export:
       - data/processed/complaints_all.jsonl    (all cleaned records)
       - data/processed/pretrain_corpus.txt     (all text, one line per record)

Run: python preprocessing/clean_pipeline.py
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Add abctokz to path
sys.path.insert(0, str(Path(__file__).parent.parent / "abctokz_repo" / "src"))

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

# ─────────────────────────────────────────
# PII patterns to mask
# ─────────────────────────────────────────
PII_PATTERNS = [
    (re.compile(r"\b[6-9]\d{9}\b"), "<PHONE>"),                         # Indian mobile
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "<AADHAAR>"),           # Aadhaar
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "<PAN>"),                  # PAN card
    (re.compile(r"\S+@\S+\.\S+"), "<EMAIL>"),                           # Email
    (re.compile(r"\b\d{6}\b"), "<PINCODE>"),                            # 6-digit PIN (may also be plot no.)
    (re.compile(r"https?://\S+"), "<URL>"),                             # URLs
]

# ─────────────────────────────────────────
# Language detection labels
# ─────────────────────────────────────────
DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
HINGLISH_WORDS = {
    "nahi", "hai", "hain", "tha", "thi", "paani", "sadak", "kachra",
    "bijli", "nagar", "ward", "ghar", "raha", "rahi", "nahi", "bahut",
    "kal", "aaj", "karo", "karo", "bhi", "aur", "se", "ko", "ka", "ki",
    "kab", "tak", "dena", "lena", "aaega", "milega", "gaya", "gayi",
}


def detect_language(text: str) -> str:
    """Detect primary language of text."""
    text_lower = text.lower()
    words = set(text_lower.split())

    has_devanagari = bool(DEVANAGARI_RANGE.search(text))
    hinglish_count = len(words & HINGLISH_WORDS)

    if has_devanagari:
        return "hi"  # Hindi/Marathi Devanagari
    elif hinglish_count >= 2:
        return "hinglish"
    else:
        try:
            from langdetect import detect, LangDetectException
            lang = detect(text)
            return lang
        except Exception:
            return "en"


def mask_pii(text: str) -> str:
    """Replace PII patterns with placeholder tokens."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def normalize_text(text: str) -> str:
    """
    Normalize text using abctokz's multilingual normalizer.
    Falls back to basic normalization if abctokz not available.
    """
    try:
        from abctokz.normalizers.devanagari import DevanagariNormalizer
        norm = DevanagariNormalizer(nfc_first=True, strip_zero_width=False)
        text = norm.normalize(text)
    except ImportError:
        import unicodedata
        text = unicodedata.normalize("NFC", text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def is_valid(text: str, min_words: int = 5, max_words: int = 150) -> bool:
    """Check if text passes quality filters."""
    if not text or not text.strip():
        return False
    word_count = len(text.split())
    if word_count < min_words or word_count > max_words:
        return False
    # Skip if mostly numbers/symbols
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return False
    return True


def minhash_dedup(records: list[dict], threshold: float = 0.85) -> list[dict]:
    """Near-duplicate detection using MinHash LSH."""
    try:
        from datasketch import MinHash, MinHashLSH
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        unique = []
        seen_keys = {}

        for i, r in enumerate(records):
            text = r.get("text", "")
            m = MinHash(num_perm=128)
            for word in text.lower().split():
                m.update(word.encode("utf-8"))

            key = f"rec_{i}"
            try:
                result = lsh.query(m)
                if not result:
                    lsh.insert(key, m)
                    unique.append(r)
                    seen_keys[key] = i
            except Exception:
                unique.append(r)

        log.info("MinHash dedup: %d → %d records (removed %d duplicates)",
                 len(records), len(unique), len(records) - len(unique))
        return unique

    except ImportError:
        log.warning("datasketch not installed — using exact dedup only")
        seen = set()
        unique = []
        for r in records:
            text = r.get("text", "").strip()
            if text not in seen:
                seen.add(text)
                unique.append(r)
        return unique


def load_all_raw() -> list[dict]:
    """Load all raw scraped data from all sources."""
    all_records = []

    # Twitter
    twitter_file = os.path.join(RAW_DIR, "twitter", "tweets.jsonl")
    if os.path.exists(twitter_file):
        with open(twitter_file, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    all_records.append({
                        "text": r.get("text", ""),
                        "source": r.get("source", "twitter"),
                        "date": r.get("date", ""),
                        "lang_raw": r.get("lang", ""),
                        "reply_to": r.get("reply_to", ""),
                    })
                except Exception:
                    pass
        log.info("Loaded Twitter: %d records", len(all_records))

    # PG Portal
    pg_file = os.path.join(RAW_DIR, "pgportal", "pgportal_grievances.jsonl")
    if os.path.exists(pg_file):
        before = len(all_records)
        with open(pg_file, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    all_records.append({
                        "text": r.get("text", ""),
                        "department": r.get("department", ""),
                        "source": "pgportal",
                        "date": r.get("date", ""),
                    })
                except Exception:
                    pass
        log.info("Loaded PG Portal: %d records", len(all_records) - before)

    # MC Websites
    mc_dir = os.path.join(RAW_DIR, "mc_websites")
    if os.path.exists(mc_dir):
        before = len(all_records)
        for fname in os.listdir(mc_dir):
            if fname.endswith(".jsonl") and not fname.endswith("_visited.txt"):
                with open(os.path.join(mc_dir, fname), encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            all_records.append({
                                "text": r.get("text", ""),
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "source": r.get("source", "mc_website"),
                            })
                        except Exception:
                            pass
        log.info("Loaded MC Websites: %d records", len(all_records) - before)

    return all_records


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log.info("Loading all raw data...")
    records = load_all_raw()
    log.info("Total raw records: %d", len(records))

    # Step 1: Exact dedup by text
    log.info("Step 1: Exact deduplication...")
    seen_texts = set()
    deduped = []
    for r in records:
        text = r.get("text", "").strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            deduped.append(r)
    log.info("After exact dedup: %d records", len(deduped))

    # Step 2: Quality filter
    log.info("Step 2: Quality filtering...")
    filtered = [r for r in deduped if is_valid(r.get("text", ""))]
    log.info("After quality filter: %d records", len(filtered))

    # Step 3: Near-dedup with MinHash
    log.info("Step 3: Near-duplicate detection...")
    filtered = minhash_dedup(filtered)

    # Step 4: PII masking + normalization + language detection
    log.info("Step 4: PII masking, normalization, language detection...")
    processed = []
    for r in filtered:
        text = r.get("text", "")
        text = mask_pii(text)
        text = normalize_text(text)
        lang = detect_language(text)
        r["text"] = text
        r["language"] = lang
        processed.append(r)

    log.info("Processed %d records", len(processed))

    # Step 5: Save cleaned complaints
    out_complaints = os.path.join(OUT_DIR, "complaints_all.jsonl")
    with open(out_complaints, "w", encoding="utf-8") as f:
        for r in processed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved complaints → %s", out_complaints)

    # Step 6: Build pretrain corpus (one line per record)
    out_corpus = os.path.join(OUT_DIR, "pretrain_corpus.txt")
    with open(out_corpus, "w", encoding="utf-8") as f:
        for r in processed:
            text = r.get("text", "").replace("\n", " ").strip()
            if text:
                f.write(text + "\n")

    # Also add PDF text to pretrain corpus
    pdf_dir = os.path.join(RAW_DIR, "pdfs")
    if os.path.exists(pdf_dir):
        with open(out_corpus, "a", encoding="utf-8") as f:
            for fname in os.listdir(pdf_dir):
                if fname.endswith(".txt"):
                    with open(os.path.join(pdf_dir, fname), encoding="utf-8") as pf:
                        for line in pf:
                            line = line.strip()
                            if len(line.split()) >= 5:
                                f.write(line + "\n")
        log.info("Added PDF text to pretrain corpus")

    total_lines = sum(1 for _ in open(out_corpus, encoding="utf-8"))
    log.info("Pretrain corpus: %d lines → %s", total_lines, out_corpus)

    # Summary
    lang_dist = {}
    for r in processed:
        lg = r.get("language", "unknown")
        lang_dist[lg] = lang_dist.get(lg, 0) + 1
    log.info("Language distribution: %s", lang_dist)

    source_dist = {}
    for r in processed:
        src = r.get("source", "unknown")
        source_dist[src] = source_dist.get(src, 0) + 1
    log.info("Source distribution: %s", source_dist)


if __name__ == "__main__":
    main()
