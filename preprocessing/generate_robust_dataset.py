import os
import random
import pandas as pd
from itertools import product
from pathlib import Path

random.seed(42)

DEPARTMENTS = ["water_supply", "solid_waste", "roads", "health", "electricity", "disaster_management", "revenue", "parks", "sewerage"]
INTENTS = ["emergency", "status_check", "query", "service_request", "complaint"]

# ==========================================
# 1. VERY STRONG FORMAL ENGLISH
# ==========================================
def generate_formal_english():
    res = []
    
    # 1.1 Complaints
    intros = [
        "I am writing to formally register a grievance regarding",
        "This communication serves as an official complaint about",
        "It is with profound disappointment that I report",
        "I wish to bring to the immediate attention of the municipal authorities",
        "Please consider this a formal notification regarding the persistent issue of"
    ]
    issues = [
        ("the severe degradation of the local road infrastructure", "roads"),
        ("the catastrophic failure of the municipal drainage system", "sewerage"),
        ("the perpetual lack of potable water supply", "water_supply"),
        ("the egregious accumulation of solid waste and refuse", "solid_waste"),
        ("the complete absence of functional street illumination", "electricity"),
        ("the unauthorized encroachment upon civic pedestrian pathways", "revenue"),
        ("the proliferation of disease-carrying vectors due to stagnant water", "health"),
        ("the deplorable state of maintenance of the public recreational parks", "parks")
    ]
    closings = [
        "I anticipate a prompt and commensurate response.",
        "Kindly expedite the resolution of this matter.",
        "Your immediate intervention is highly solicited.",
        "I expect this grievance to be addressed with the utmost urgency."
    ]
    for i, (iss, dept), c in product(intros, issues, closings):
        res.append((f"{i} {iss}. {c}", dept, "complaint"))
        
    # 1.2 Queries
    q_intros = ["Could you kindly elucidate", "I require comprehensive information regarding", "Please detail the standard operating procedure for"]
    q_topics = [
        ("the remittance of property tax arrears", "revenue"),
        ("the acquisition of a duplicated death registry certificate", "health"),
        ("the spatial zoning parameters for commercial establishment", "revenue"),
        ("the scheduled deployment of waste management vehicles", "solid_waste")
    ]
    for i, (top, dept) in product(q_intros, q_topics):
        res.append((f"{i} {top}?", dept, "query"))
        
    return res

# ==========================================
# 2. WEAK / BROKEN SENTENCES (Short/Abrupt)
# ==========================================
def generate_broken_english():
    res = []
    issues = [
        ("no water coming", "water_supply"),
        ("road very bad break car", "roads"),
        ("garbage everywhere smell", "solid_waste"),
        ("light gone full dark", "electricity"),
        ("drain block dirty water", "sewerage"),
        ("big tree fall danger", "disaster_management"),
        ("tax portal not working", "revenue"),
        ("dog bite many people", "health"),
        ("park grass too long snakes", "parks")
    ]
    prefixes = ["pls fix", "very urgent:", "bmc please", "sir", ""]
    suffixes = ["do fast", "help", "very sad", "fix it", ""]
    
    for p, (iss, dept), s in product(prefixes, issues, suffixes):
        res.append((f"{p} {iss} {s}".strip(), dept, "complaint"))
        res.append((f"{iss} when fix?", dept, "status_check")) # Broken status
        res.append((f"need {iss.split()[-1]} department", dept, "service_request")) # Broken service
    return res

# ==========================================
# 3. HALF-HINDI HALF-ENGLISH (Code-Mixing)
# ==========================================
def generate_code_mixed():
    res = []
    # Mixes of English subjects/verbs with Hindi descriptors
    situations = [
        ("The transformer is sparking, aag lagne ka darr hai", "electricity", "emergency"),
        ("Nobody is listening to us, kab tak wait kare for water supply", "water_supply", "complaint"),
        ("Garbage truck is not coming, kachra ghar ke bahar pada hai", "solid_waste", "complaint"),
        ("Potholes are so big, gaadi chalana muskil ho gaya hai", "roads", "complaint"),
        ("Tax portal is down, payment kaise kare", "revenue", "query"),
        ("Need a fogging machine immediately, machhar bohot badh gaye hai", "health", "service_request"),
        ("Sewer water is overflowing, pura rasta kharab ho rakha hai", "sewerage", "complaint"),
        ("Complaint ID 4059 is showing pending, koi action kyu nahi le raha", "roads", "status_check"),
        ("Building is collapsing, jaldi fire brigade bhejo", "disaster_management", "emergency")
    ]
    
    variations = ["@bmc", "Please see this,", "Urgent:", "Sir,", ""]
    for v in variations:
        for text, dept, intent in situations:
            res.append((f"{v} {text}".strip(), dept, intent))
            
    return res

# ==========================================
# 4. BIG NARRATIVE SENTENCES (Long winded)
# ==========================================
def generate_long_narratives():
    res = []
    templates = [
        ("water_supply", "complaint", "Sir I have been living in this residential society for over 20 years and I have never seen such the worst condition of municipal provisions where we have been completely deprived of basic drinking water for 5 consecutive days despite paying our taxes on time and making multiple calls to the local ward office with absolutely zero response from any official."),
        ("roads", "complaint", "It is extremely frustrating to drive on the main arterial road connecting the highway because there are massive craters and potholes every two meters which completely destroyed my car's suspension yesterday and causes massive traffic jams every single morning when people are just trying to get to their workplace."),
        ("health", "emergency", "There is an extremely urgent situation developing in our sector right now where a pack of highly aggressive stray dogs has already bitten three innocent children playing in the park and they are still roaming freely attacking anyone who walks by so we need the animal control team deployed immediately before someone is fatally injured."),
        ("solid_waste", "service_request", "We are a group of concerned citizens writing to formally request the immediate deployment of municipal waste clearing trucks to our alleyway because the accumulated construction debris and household garbage has formed a literal mountain blocking the pedestrian pathway and creating an unbearable stench across the entire neighborhood."),
        ("revenue", "query", "I would like to understand the detailed step by step procedure required for a senior citizen to claim the promised rebate on their annual residential property tax calculation through the new online portal because the interface is extremely confusing and keeps rejecting my uploaded documents without specifying the exact format required by the municipal guidelines.")
    ]
    for dept, intent, text in templates:
        res.append((text, dept, intent))
    return res

# ==========================================
# 5. VERY SHORT SENTENCES (1-3 words)
# ==========================================
def generate_micro_sentences():
    res = []
    words = [
        ("fire help", "disaster_management", "emergency"),
        ("need ambulance", "health", "emergency"),
        ("water leak", "water_supply", "complaint"),
        ("fix road", "roads", "service_request"),
        ("trash pickup", "solid_waste", "service_request"),
        ("tax query", "revenue", "query"),
        ("status check", "revenue", "status_check"),
        ("clean park", "parks", "service_request"),
        ("street light", "electricity", "complaint"),
        ("drain choke", "sewerage", "complaint")
    ]
    for t, dept, intent in words:
        res.append((t, dept, intent))
        
    return res

# ==========================================
# 6. PURE HINGLISH (Existing style but enhanced)
# ==========================================
def generate_hinglish():
    res = []
    starts = ["bhai", "sir", "kya backbakwas hai", "bmc walo", "arey", ""]
    mids = [
        ("paani nahi aa raha", "water_supply"),
        ("kachra sad raha hai", "solid_waste"),
        ("rasta kab banega", "roads"),
        ("machar ghum rahe", "health"),
        ("light gayi hai", "electricity"),
        ("baarish ka pani bhar gaya", "disaster_management"),
        ("property tax bharna hai", "revenue"),
        ("garden me kachra hai", "parks"),
        ("gutter line block hai", "sewerage")
    ]
    ends = ["kuch karo", "kab solve hoga", "pls help", "dimag kharab", ""]
    
    for s, (m, dept), e in product(starts, mids, ends):
        res.append((f"{s} {m} {e}".strip(), dept, "complaint"))
    return res

# ==========================================
# NOISE AND CHAOS INJECTOR
# ==========================================
def inject_extreme_noise(text):
    text = text if random.random() > 0.4 else text.lower()
    text = text if random.random() > 0.3 else text.upper()
    
    # 1. Spacing issues (missing spaces or double spaces)
    if random.random() > 0.8:
        text = text.replace(" ", "", random.randint(1, 3)) # remove random spaces
        
    # 2. Extreme Typos
    if random.random() > 0.7:
        typos = {"water": ["watar", "watrr", "wtaer"], 
                 "garbage": ["grbage", "garbge", "garbej"],
                 "road": ["rod", "roud", "raad"],
                 "please": ["plz", "pls", "plzzzz"],
                 "sir": ["sirr", "saar"],
                 "the": ["teh", "da"]}
        for k, v in typos.items():
            if k in text.lower():
                text = text.replace(k, random.choice(v))
                
    # 3. Repeated characters (angry typing)
    if random.random() > 0.85:
        text = text.replace("!", "!!!!").replace("?", "????")
        if "please" in text.lower():
            text = text.replace("please", "pleeeeeease")
        if "help" in text.lower():
            text = text.replace("help", "heeeeelp")
            
    # 4. Punctuation chaos
    if random.random() > 0.8:
        chars = ["... ", ",,, ", " !! ", " ?? "]
        words = text.split()
        if len(words) > 3:
            idx = random.randint(1, len(words)-1)
            words.insert(idx, random.choice(chars))
            text = " ".join(words)
            
    return text

def main():
    print("Generating Next-Gen Robust Synthetic Dataset (350k Rows)...")
    
    all_data = []
    # Heavily multiply to ensure enough base combinations for 350k records
    all_data.extend(generate_formal_english() * 1000)      
    all_data.extend(generate_broken_english() * 2000)
    all_data.extend(generate_code_mixed() * 3000)
    all_data.extend(generate_long_narratives() * 5000)
    all_data.extend(generate_micro_sentences() * 3000)
    all_data.extend(generate_hinglish() * 2000)
    
    # Inject missing intents perfectly balanced
    from generate_balanced_dataset import generate_status_check, generate_query, generate_service_request, generate_emergency
    all_data.extend(generate_status_check() * 500)
    all_data.extend(generate_query() * 500)
    all_data.extend(generate_service_request() * 500)
    all_data.extend(generate_emergency() * 500)
    
    print("Balancing dataset to exactly 70,000 per intent (Total 350,000)...")
    df_raw = pd.DataFrame(all_data, columns=["text", "department", "intent"])
    
    # Balance the dataset (70k per intent)
    balanced_data = []
    TARGET_PER_INTENT = 70000
    for intent in INTENTS:
        subset = df_raw[df_raw['intent'] == intent]
        if len(subset) >= TARGET_PER_INTENT:
            sampled = subset.sample(TARGET_PER_INTENT, random_state=42)
        else:
            sampled = subset.sample(TARGET_PER_INTENT, replace=True, random_state=42)
        balanced_data.append(sampled)
        
    df = pd.concat(balanced_data).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Applying extreme noise and linguistic chaos...")
    df["text"] = df["text"].apply(inject_extreme_noise)
    
    # Clean whitespace sequences that might have been created
    df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Drop empties if any noise ruined it
    df = df[df['text'].str.len() > 1]
    
    out_dir = Path("/home/mahesh/Desktop/mpr_latest/data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "complaints_robust.csv" 
    
    df.to_csv(out_file, index=False)
    
    print(f"\n✓ Generated {len(df)} heavily diverse and chaotic rows.")
    print("\n--- Intent Distribution ---")
    print(df["intent"].value_counts())
    print("\n--- Department Distribution ---")
    print(df["department"].value_counts())
    print(f"\n✓ Saved to {out_file}")
    
    # Show some examples
    print("\n--- Random Samples ---")
    for _, row in df.sample(10).iterrows():
        print(f"[{row['intent'].upper()}] [{row['department'].upper()}]: {row['text'][:100]}")

if __name__ == "__main__":
    main()
