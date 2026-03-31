"""
generate_intents.py
===================
Generates multi-intent dataset for the Municipal Corporation MLM.
Adds 4 new intent types alongside existing complaints:
  - query        : "birth certificate kaise milega?"
  - status_check : "meri complaint ka kya hua?"
  - emergency    : "aag lagi hai!"
  - service_request : "naya water connection chahiye"

Output appended to data/processed/complaints_labeled.csv
Also updates pretrain_corpus.txt

Run AFTER generate_synthetic_data.py:
  python preprocessing/generate_intents.py
"""

import csv
import os
import random
import re

random.seed(99)

OUT_DIR = "data/processed"
OUT_LABELED = os.path.join(OUT_DIR, "complaints_labeled.csv")
OUT_CORPUS = os.path.join(OUT_DIR, "pretrain_corpus.txt")

WARDS = [f"ward {i}" for i in range(1, 50)] + ["Andheri", "Bandra", "Koramangala", "Pimpri"]
DEPTS = ["water supply", "engineering", "swm", "public health", "revenue", "town planning", "registration", "licensing", "fire"]
APP_IDS = [f"MC{random.randint(10000,99999)}" for _ in range(200)]
COMPLAINT_IDS = [f"CMP{random.randint(1000,9999)}" for _ in range(200)]

NOISE_CASES = [str.lower, str.upper, lambda x: x]
FILLER = ["", "", "", "bhai ", "sir ", "hello ", "plz ", "kindly "]

def noisify(text):
    fn = random.choice(NOISE_CASES)
    text = fn(text)
    if random.random() < 0.3:
        text = text.rstrip("?.!")
    return random.choice(FILLER) + text.strip()


# ─────────────────────────────────────────
# INTENT: QUERY
# Questions about procedures/policies — answered by RAG
# ─────────────────────────────────────────
QUERY_TEMPLATES = {
    "Water Supply": [
        "how to apply for new water connection",
        "new water connection ke liye kya karna padta hai",
        "नया पानी कनेक्शन कैसे मिलेगा",
        "water connection transfer process kya hai",
        "paani ka bil online kaise bhare",
        "how to pay water bill online",
        "water meter complaint kaise kare",
        "paani ka meter kharab hai kya karna hai",
    ],
    "Revenue/Tax": [
        "property tax kaise calculate hota hai",
        "how is property tax calculated",
        "संपत्ति कर की गणना कैसे होती है",
        "property tax exemption ke liye kaun eligible hai",
        "self assessment form kaise bhare",
        "property tax online payment kaise kare",
        "mera property tax account number kaise milega",
        "what documents needed for property tax transfer",
    ],
    "Registration": [
        "birth certificate ke liye kya documents chahiye",
        "documents required for birth certificate",
        "जन्म प्रमाण पत्र के लिए क्या चाहिए",
        "death certificate kaise apply kare",
        "marriage certificate process kya hai",
        "birth certificate correction kaise kare",
        "how many days to get birth certificate",
        "death certificate fees kitni hai",
    ],
    "Town Planning": [
        "building permission ke liye kya chahiye",
        "ghar banane ke liye kya permission chahiye",
        "मकान बनाने की अनुमति कैसे मिलती है",
        "completion certificate kaise milega",
        "what is the process for OC certificate",
        "building plan approval kitne din lagta hai",
        "floor space index kya hota hai",
        "set back rules kya hain",
    ],
    "Licensing": [
        "trade license ke liye kya documents chahiye",
        "dukan ka license kaise banwaye",
        "व्यापार लाइसेंस के लिए क्या करें",
        "fire noc kaise milega",
        "food license kaise apply kare",
        "trade license renewal process kya hai",
        "how long is trade license valid",
    ],
    "SWM": [
        "garbage collection timings kya hain",
        "kachra gaadi kab aati hai",
        "कचरा गाड़ी कब आती है",
        "how to report illegal dumping",
        "wet waste dry waste separation kaise kare",
        "composting unit kaise lagwaye",
    ],
    "Public Health": [
        "fogging ke liye kaise request kare",
        "dengue test kahan hota hai",
        "डेंगू का टेस्ट कहाँ होता है",
        "municipal hospital timings kya hain",
        "health card kaise banwaye",
        "vaccination camp kab hai",
    ],
}

# ─────────────────────────────────────────
# INTENT: STATUS CHECK
# ─────────────────────────────────────────
STATUS_TEMPLATES = [
    "meri complaint {cid} ka status kya hai",
    "complaint {cid} ka kya hua",
    "my complaint {cid} still pending",
    "I filed complaint {cid} {dur} ago no update",
    "application {aid} status batao",
    "when will application {aid} be processed",
    "mera application {aid} kahin khoya to nahi",
    "मेरी शिकायत {cid} का क्या हुआ",
    "application {aid} ka status nahi pata",
    "complaint number {cid} check karo please",
    "{cid} complaint registered but no action taken",
    "2 weeks ago filed {cid} nothing happened",
    "please tell me status of {cid}",
    "kab tak resolve hoga {cid}",
    "water connection application {aid} ka status",
    "birth certificate application {aid} ready hai kya",
]

# ─────────────────────────────────────────
# INTENT: EMERGENCY
# ─────────────────────────────────────────
EMERGENCY_TEMPLATES = [
    "aag lag gayi {ward} mein help karo",
    "fire broke out {ward} call fire brigade",
    "{ward} में आग लगी है तुरंत मदद करो",
    "manhole mein bacha gira {ward} please help",
    "child fell in open drain {ward} emergency",
    "tree fell on road {ward} blocking traffic",
    "electric wire on road {ward} live wire dangerous",
    "bijli ka taar gira {ward} log shock kha sakte hain",
    "{ward} में बिजली का तार गिरा है",
    "gas leakage {ward} please evacuate",
    "building collapse {ward}",
    "flood in {ward} people stuck please help",
    "road cave in {ward} car fell inside",
    "pipeline blast {ward} water everywhere road closed",
    "HELP aag lag gayi {ward} fire engine bhejo",
    "SOS open manhole {ward} no cover",
]

# ─────────────────────────────────────────
# INTENT: SERVICE REQUEST
# ─────────────────────────────────────────
SERVICE_TEMPLATES = {
    "Water Supply": [
        "naya water connection chahiye {ward} mein",
        "I want to apply for new water connection {ward}",
        "पानी का नया कनेक्शन चाहिए {ward} में",
        "water meter change karna hai {ward}",
        "water connection transfer karna hai {ward}",
    ],
    "Revenue/Tax": [
        "property tax self assessment karna hai",
        "property tax naam transfer karna hai",
        "संपत्ति कर जमा करना है ऑनलाइन",
        "property tax objection file karna hai",
    ],
    "Registration": [
        "birth certificate apply karna hai {ward}",
        "death certificate apply karna hai",
        "जन्म प्रमाण पत्र के लिए आवेदन करना है",
        "marriage certificate chahiye",
    ],
    "Licensing": [
        "new trade license apply karna hai {ward}",
        "trade license renewal karna hai",
        "नया व्यापार लाइसेंस चाहिए {ward} में",
        "food license apply karna hai",
    ],
    "Town Planning": [
        "building plan approval ke liye apply karna hai",
        "OC certificate apply karna hai",
        "भवन निर्माण अनुमति के लिए आवेदन करना है",
        "encroachment removal request dena hai",
    ],
    "SWM": [
        "extra dustbin chahiye {ward} colony mein",
        "composting unit ke liye apply karna hai",
        "bulk waste pickup schedule karna hai",
    ],
}


def generate_records():
    records = []
    rid = 0
    REPEATS = 40

    # QUERY intent
    for dept, templates in QUERY_TEMPLATES.items():
        for _ in range(REPEATS):
            for tmpl in templates:
                text = noisify(tmpl)
                records.append({
                    "id": rid, "text": text, "clean_text": text,
                    "intent": "query", "department": dept,
                    "category": "Information Query", "severity": "LOW",
                    "language": "en" if all(ord(c) < 128 for c in tmpl) else "hi",
                    "confidence": round(random.uniform(0.8, 1.0), 2),
                    "source": "synthetic_intent",
                })
                rid += 1

    # STATUS CHECK intent
    for _ in range(REPEATS * 3):
        for tmpl in STATUS_TEMPLATES:
            dur = random.choice(["3 days", "1 week", "2 weeks", "one month"])
            cid = random.choice(COMPLAINT_IDS)
            aid = random.choice(APP_IDS)
            try:
                text = noisify(tmpl.format(cid=cid, aid=aid, dur=dur))
            except Exception:
                text = noisify(tmpl)
            records.append({
                "id": rid, "text": text, "clean_text": text,
                "intent": "status_check", "department": "General",
                "category": "Status Check", "severity": "LOW",
                "language": "hinglish",
                "confidence": round(random.uniform(0.8, 1.0), 2),
                "source": "synthetic_intent",
            })
            rid += 1

    # EMERGENCY intent
    for _ in range(REPEATS * 2):
        for tmpl in EMERGENCY_TEMPLATES:
            ward = random.choice(WARDS)
            try:
                text = noisify(tmpl.format(ward=ward))
            except Exception:
                text = noisify(tmpl)
            records.append({
                "id": rid, "text": text, "clean_text": text,
                "intent": "emergency", "department": "Fire Services",
                "category": "Emergency", "severity": "HIGH",
                "language": "hinglish",
                "confidence": 1.0,
                "source": "synthetic_intent",
            })
            rid += 1

    # SERVICE REQUEST intent
    for dept, templates in SERVICE_TEMPLATES.items():
        for _ in range(REPEATS):
            for tmpl in templates:
                ward = random.choice(WARDS)
                try:
                    text = noisify(tmpl.format(ward=ward))
                except Exception:
                    text = noisify(tmpl)
                records.append({
                    "id": rid, "text": text, "clean_text": text,
                    "intent": "service_request", "department": dept,
                    "category": "Service Request", "severity": "LOW",
                    "language": "hinglish",
                    "confidence": round(random.uniform(0.8, 1.0), 2),
                    "source": "synthetic_intent",
                })
                rid += 1

    return records


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load existing complaint records and patch intent label
    existing = []
    if os.path.exists(OUT_LABELED):
        with open(OUT_LABELED, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["intent"] = "complaint"  # all existing are complaints
                existing.append(row)
        print(f"Loaded {len(existing):,} existing complaint records")

    new_records = generate_records()
    print(f"Generated {len(new_records):,} new intent records")

    # Combine and shuffle
    all_records = existing + new_records
    random.shuffle(all_records)

    # Determine columns (union of all keys)
    fieldnames = ["id", "text", "clean_text", "intent", "department",
                  "category", "severity", "language", "confidence", "source"]

    with open(OUT_LABELED, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(all_records):
            row["id"] = i
            writer.writerow(row)

    # Update corpus
    with open(OUT_CORPUS, "a", encoding="utf-8") as f:
        for r in new_records:
            f.write(r["text"] + "\n")

    # Stats
    intent_dist = {}
    for r in all_records:
        k = r.get("intent", "complaint")
        intent_dist[k] = intent_dist.get(k, 0) + 1

    print(f"\nTotal: {len(all_records):,} records")
    print("Intent distribution:")
    for k, v in sorted(intent_dist.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v:>6}")

    print(f"\n✓ Saved → {OUT_LABELED}")
    print(f"✓ Corpus appended → {OUT_CORPUS}")


if __name__ == "__main__":
    main()
