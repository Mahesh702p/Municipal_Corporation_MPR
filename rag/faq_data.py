"""
faq_data.py
===========
Generates municipal FAQ (Question-Answer) pairs for the RAG pipeline.
These are chunked and indexed into FAISS for retrieval.

Covers: Water Supply, Property Tax, Registration, Building Permission,
        Trade License, SWM, Health, Emergency procedures.

Output: data/rag/faq_chunks.jsonl
"""

import json
import os

OUT_DIR = "data/rag"
OUT_FILE = os.path.join(OUT_DIR, "faq_chunks.jsonl")

FAQ_DATA = [
    # ── WATER SUPPLY ─────────────────────────────────────────────────────────
    {
        "id": "ws_001",
        "department": "Water Supply",
        "question": "How to apply for a new water connection?",
        "answer": (
            "To apply for a new water connection: "
            "1. Visit the municipal water department office or online portal. "
            "2. Fill Form WC-1 (New Connection Application). "
            "3. Submit documents: property ownership proof, ID proof, site plan. "
            "4. Pay the connection fee (varies by pipe diameter: ½ inch = ₹2000, ¾ inch = ₹3500). "
            "5. Inspection will be done within 7 working days. "
            "6. Connection provided within 15 working days after inspection approval."
        ),
        "keywords": ["water connection", "apply", "naya connection", "पानी कनेक्शन"],
    },
    {
        "id": "ws_002",
        "department": "Water Supply",
        "question": "How to pay water bill online?",
        "answer": (
            "Water bill can be paid online through: "
            "1. Municipal corporation official website → 'Pay Water Bill' section. "
            "2. Enter Consumer Number (found on your bill). "
            "3. View outstanding amount and pay via UPI, Net Banking, or Debit Card. "
            "4. Download receipt immediately. "
            "Payment can also be done at any CSC (Common Service Centre) or authorized bank."
        ),
        "keywords": ["water bill", "online pay", "paani bill", "consumer number"],
    },
    {
        "id": "ws_003",
        "department": "Water Supply",
        "question": "Water connection transfer process?",
        "answer": (
            "To transfer a water connection to a new owner: "
            "Submit Form WC-3 with: sale deed / property transfer documents, "
            "ID proof of new owner, last paid water bill, NOC from previous owner. "
            "Processing time: 15–30 working days. Fee: ₹500."
        ),
        "keywords": ["water connection transfer", "naam transfer", "new owner"],
    },

    # ── PROPERTY TAX ─────────────────────────────────────────────────────────
    {
        "id": "pt_001",
        "department": "Revenue/Tax",
        "question": "How is property tax calculated?",
        "answer": (
            "Property tax = Annual Rateable Value × Tax Rate. "
            "Annual Rateable Value (ARV) = Monthly Rent × 12. "
            "For self-occupied residential properties: ARV is based on carpet area × zone rate. "
            "Tax rates: Residential = 0.5–1%, Commercial = 1–2%, Industrial = 1.5–2.5%. "
            "10% discount for early payment (before June 30). "
            "Penalty of 2% per month for late payment."
        ),
        "keywords": ["property tax calculation", "tax rate", "ARV", "संपत्ति कर"],
    },
    {
        "id": "pt_002",
        "department": "Revenue/Tax",
        "question": "Who is eligible for property tax exemption?",
        "answer": (
            "Property tax exemption is available for: "
            "1. Ex-servicemen (50% exemption). "
            "2. Senior citizens above 65 years (25% exemption for self-occupied residential). "
            "3. Physically handicapped persons with 40%+ disability (50% exemption). "
            "4. Religious/charitable institutions (100% if not used for profit). "
            "Application required at Revenue Department with supporting documents."
        ),
        "keywords": ["property tax exemption", "senior citizen", "disability", "ex-servicemen"],
    },
    {
        "id": "pt_003",
        "department": "Revenue/Tax",
        "question": "How to file property tax self-assessment?",
        "answer": (
            "Self-assessment can be done online: "
            "1. Login to municipal portal with property ID. "
            "2. Fill self-assessment form with property details (area, usage, zone). "
            "3. System calculates tax automatically. "
            "4. Review and submit. Pay online or download challan for bank payment. "
            "Self-assessment can also be done at any municipal ward office."
        ),
        "keywords": ["self assessment", "property tax online", "form fill"],
    },

    # ── BIRTH / DEATH CERTIFICATE ─────────────────────────────────────────────
    {
        "id": "reg_001",
        "department": "Registration",
        "question": "Documents required for birth certificate?",
        "answer": (
            "Birth Certificate application documents: "
            "1. Hospital discharge summary / birth report (from hospital where birth occurred). "
            "2. Parents' ID proof (Aadhaar, PAN, Voter ID). "
            "3. Parents' address proof. "
            "4. Marriage certificate of parents. "
            "5. Form No. 1 (Birth Registration Form) — available at ward office. "
            "Registration must be done within 21 days of birth. "
            "After 21 days: additional affidavit required. After 1 year: court order needed. "
            "Fee: Free within 21 days. ₹50 after 21 days."
        ),
        "keywords": ["birth certificate", "documents", "janam praman", "registration"],
    },
    {
        "id": "reg_002",
        "department": "Registration",
        "question": "How to apply for death certificate?",
        "answer": (
            "Death Certificate documents required: "
            "1. Death report from hospital / cremation ground certificate. "
            "2. Deceased's ID proof (Aadhaar, Voter ID). "
            "3. Applicant's ID and relationship proof. "
            "4. Address proof. "
            "Must be registered within 21 days. Fee: Free within 21 days, ₹50 thereafter. "
            "Issued within 7 working days at ward office."
        ),
        "keywords": ["death certificate", "mrityu praman patra", "documents"],
    },
    {
        "id": "reg_003",
        "department": "Registration",
        "question": "How to correct a birth certificate?",
        "answer": (
            "To correct errors in birth certificate: "
            "Submit application at ward office with: Original certificate, "
            "Affidavit stating the error and correct information, "
            "Documentary proof of correct information (hospital records, school records). "
            "Fee: ₹100. Processing time: 15–30 working days. "
            "For name addition after 1 year: magistrate affidavit required."
        ),
        "keywords": ["birth certificate correction", "name change", "correction"],
    },

    # ── BUILDING PERMISSION ───────────────────────────────────────────────────
    {
        "id": "tp_001",
        "department": "Town Planning",
        "question": "What documents are needed for building permission?",
        "answer": (
            "Documents required for building plan approval: "
            "1. Ownership documents (7/12 extract, property card, title deed). "
            "2. Site plan and building plan drawn by licensed architect. "
            "3. Structural stability certificate by licensed engineer. "
            "4. No Objection Certificate from Fire Dept (for buildings >15m). "
            "5. NOC from Airport Authority (if near airport zone). "
            "6. Form IV application. "
            "Timeline: Deemed approval in 30 days if documents complete. "
            "Fees: Based on built-up area (₹50–200 per sq.m depending on zone)."
        ),
        "keywords": ["building permission", "plan approval", "documents", "architect"],
    },
    {
        "id": "tp_002",
        "department": "Town Planning",
        "question": "What is FSI (Floor Space Index)?",
        "answer": (
            "FSI (Floor Space Index) = Total built-up area / Plot area. "
            "It determines how much construction is allowed on a plot. "
            "FSI varies by zone: Residential = 1.0–2.5, Commercial = 2.0–4.0. "
            "Example: Plot of 100 sq.m with FSI 2.5 allows 250 sq.m total construction. "
            "Check your local zoning to know your FSI. Available on municipal website → Land Use Plan."
        ),
        "keywords": ["FSI", "floor space index", "FAR", "construction limit"],
    },

    # ── TRADE LICENSE ─────────────────────────────────────────────────────────
    {
        "id": "lic_001",
        "department": "Licensing",
        "question": "How to apply for a trade/shop license?",
        "answer": (
            "Trade License application process: "
            "1. Apply online at municipal portal or at licensing department. "
            "2. Documents: Business address proof, Owner ID proof, "
            "Lease/ownership proof of premises, "
            "NOC from property owner (if rented), Fire NOC (for food/hazardous businesses). "
            "3. Fee: ₹500–₹5000 depending on business type and area. "
            "4. Physical inspection within 7 days. "
            "5. License issued within 30 days. Valid for 1 year, renewable annually."
        ),
        "keywords": ["trade license", "shop license", "dukan license", "business license"],
    },
    {
        "id": "lic_002",
        "department": "Licensing",
        "question": "How to renew trade license?",
        "answer": (
            "Trade license renewal: "
            "Apply 30 days before expiry. "
            "Submit: Renewal form, Previous year license copy, Updated address/ID proof, "
            "Paid property tax receipt for the premises. "
            "Fee: Same as original license fee. "
            "Online renewal available on municipal portal. No physical inspection if no change in business."
        ),
        "keywords": ["trade license renewal", "license renew", "shop renew"],
    },

    # ── SWM / GARBAGE ─────────────────────────────────────────────────────────
    {
        "id": "swm_001",
        "department": "SWM",
        "question": "What are garbage collection timings?",
        "answer": (
            "Garbage collection schedule: "
            "Residential areas: 6:00 AM – 10:00 AM daily. "
            "Commercial areas: 9:00 AM – 1:00 PM daily. "
            "Wet waste (kitchen waste/green bin): Collected daily. "
            "Dry waste (plastic, paper/blue bin): Collected alternate days. "
            "Bulk waste (furniture, construction): Call 1800-XXX-XXXX for scheduled pickup. "
            "If collection missed: Complain on municipal app or call ward office."
        ),
        "keywords": ["garbage timing", "collection time", "kachra time", "waste pickup"],
    },
    {
        "id": "swm_002",
        "department": "SWM",
        "question": "How to segregate wet and dry waste?",
        "answer": (
            "Waste segregation rules: "
            "GREEN BIN (Wet/Organic waste): Food scraps, vegetable peels, cooked food leftovers, "
            "garden waste, tea leaves, eggshells. "
            "BLUE BIN (Dry/Recyclable waste): Paper, plastic bottles, glass, metal, cardboard, "
            "cloth, rubber. "
            "RED BIN (Hazardous waste): Medicines, batteries, bulbs, paint, syringes. "
            "Segregation is mandatory. Fine of ₹500 for mixing waste."
        ),
        "keywords": ["waste segregation", "wet dry waste", "green bin blue bin", "kachra alag"],
    },

    # ── PUBLIC HEALTH ─────────────────────────────────────────────────────────
    {
        "id": "ph_001",
        "department": "Public Health",
        "question": "How to request mosquito fogging?",
        "answer": (
            "Mosquito fogging request: "
            "1. Call your ward office or health department helpline. "
            "2. File request on municipal app under 'Public Health → Mosquito Control'. "
            "3. Provide exact location (ward, area, landmark). "
            "Fogging is typically scheduled: Evenings between 6–8 PM. "
            "Response time: Within 48–72 hours of complaint. "
            "Ensure residents close windows during fogging."
        ),
        "keywords": ["fogging", "mosquito control", "dengue prevention", "spray request"],
    },

    # ── GENERAL / COMPLAINTS ──────────────────────────────────────────────────
    {
        "id": "gen_001",
        "department": "General",
        "question": "How to file a complaint with the municipal corporation?",
        "answer": (
            "File a complaint through: "
            "1. Municipal Corporation App (available on Play Store / App Store). "
            "2. Official website → 'Citizen Complaints' portal. "
            "3. Visit your Ward Office directly. "
            "4. Call: 1800-XXX-XXXX (toll-free, 24x7 for emergencies). "
            "5. Twitter/X: @YourMC (mention ward and issue). "
            "After filing: You receive a complaint ID. Track status online or via SMS. "
            "Resolution timeline: Emergency = 24 hours, High = 72 hours, Medium = 7 days."
        ),
        "keywords": ["file complaint", "complain kaise kare", "complaint number", "grievance"],
    },
    {
        "id": "gen_002",
        "department": "General",
        "question": "What are the emergency contact numbers?",
        "answer": (
            "Municipal Corporation Emergency Contacts: "
            "Fire Brigade: 101, "
            "Ambulance: 108, "
            "Police: 100, "
            "Municipal Helpline: 1800-XXX-XXXX (24x7), "
            "Water Emergency: 1916, "
            "Electricity Emergency: 1912, "
            "Disaster Management: 1077. "
            "For non-emergency complaints: Use the municipal app or visit ward office."
        ),
        "keywords": ["emergency number", "helpline", "contact", "phone number"],
    },
    {
        "id": "gen_003",
        "department": "General",
        "question": "What is the complaint resolution timeline?",
        "answer": (
            "Complaint resolution timelines (as per citizen charter): "
            "Emergency (fire, open manhole, live wire): 24 hours. "
            "High priority (no water supply, road accident risk): 48–72 hours. "
            "Medium priority (garbage, potholes, drainage): 7 working days. "
            "Low priority (street light, minor repairs): 15 working days. "
            "If not resolved in time: Escalate to Deputy Commissioner or use PG Portal."
        ),
        "keywords": ["complaint timeline", "resolution time", "kitne din mein", "escalate"],
    },
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    chunks = []
    for faq in FAQ_DATA:
        # Main Q+A chunk
        chunk_text = f"Q: {faq['question']}\nA: {faq['answer']}"
        chunks.append({
            "chunk_id": faq["id"],
            "department": faq["department"],
            "text": chunk_text,
            "question": faq["question"],
            "answer": faq["answer"],
            "keywords": faq["keywords"],
            "source": "synthetic_faq",
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"✓ {len(chunks)} FAQ chunks written → {OUT_FILE}")
    dept_dist = {}
    for c in chunks:
        dept_dist[c["department"]] = dept_dist.get(c["department"], 0) + 1
    for d, n in sorted(dept_dist.items()):
        print(f"  {d}: {n} chunks")


if __name__ == "__main__":
    main()
