import os
import random
import pandas as pd
from itertools import product
from pathlib import Path

random.seed(42)

# Department Mapping 
DEPARTMENTS = ["water_supply", "solid_waste", "roads", "health", "electricity", "disaster_management", "revenue", "parks", "sewerage"]

def generate_emergency():
    res = []
    prefixes = ["Help!", "URGENT:", "🚨", "CRITICAL", "MCD please", "Immediate action needed", "pls send team fast", "bhai", "someone come quick", ""]
    situations = [
        ("building collapsed completely malba everywhere", "disaster_management"),
        ("massive fire outbreak near transformer", "disaster_management"),
        ("huge banyan tree fell on main road", "disaster_management"),
        ("fatal accident multiple injured", "roads"),
        ("severe flooding water entering living room", "disaster_management"),
        ("live high tension wire snapped and fell", "electricity"),
        ("flyover bridge cracking and sinking", "roads"),
        ("huge rabid dog attacked kids", "health"),
        ("aag lag gayi hai 3rd floor mein", "disaster_management"),
        ("glae mein paani bhar gaya hai log doob rahe", "disaster_management"),
        ("chhat gir gayi log dab gaye", "disaster_management"),
        ("current lag raha hai nange taar se", "electricity"),
        ("sparking transformer se aag lagri", "electricity"),
        ("gas leakage smelling heavily", "disaster_management")
    ]
    locations = ["in bandra", "at mg road", "near dadar station", "ward 15", "at our residential society", "right here", "immediately", "right now!!"]
    suffixes = ["bachao", "save us!!", "send fire brigade fast", "ambulance need urjent", "do something quick", "kuch karo bhagwan ke liye", "jaldi aao pls", "danger alert!"]
    
    combos = list(product(prefixes, situations, locations, suffixes))
    for p, (sit, dept), l, s in combos:
        res.append((f"{p} {sit} {l} {s}".strip(), dept, "emergency"))
    return res

def generate_status_check():
    res = []
    starts = ["Please tell", "Check", "Status of", "What is happening with", "Update on", "I want to know", "Track", "kya hua", "mera", "please sir status of"]
    ids = ["complaint ID", "application number", "grievance ticket", "reference ID", "complaint no", "service request SR-"]
    numbers = ["45928", "XXXXX", "8902", "992A", "BMC-1929", "REQ-012"]
    pendency = ["pending since 3 weeks", "no reply from officials", "not resolved yet on portal", "abhi tak solve nahi kiya", "nobody from bmc came", "waiting for 1 month straight", "dikha raha hai open but nobody working", "stuck in processing department"]
    suffixes = ["why delay?", "kuch karo iska", "please reply @corporator", "solve this asapp", "@bmc", "frustrated citizen", "plz help", "kab hoga finally?"]
    
    combos = list(product(starts, ids, numbers, pendency, suffixes))
    for st, i, n, p, suf in combos:
        dept = random.choice(DEPARTMENTS)
        res.append((f"{st} {i} {n} is {p} {suf}".strip(), dept, "status_check"))
    return res

def generate_query():
    res = []
    starts = ["How to", "Where can I", "What is the procedure to", "Can someone tell me", "I need info on", "kaha se", "kaise kare", "info needed urgently:"]
    topics = [
        ("pay property tax online without fine", "revenue"),
        ("calculate FSI floor space index for plot", "revenue"),
        ("apply for death certificate duplicate", "health"),
        ("get a birth certificate registration done", "health"),
        ("renew my commercial trade license", "revenue"),
        ("check ward map zoning restrictions", "revenue"),
        ("file water tax arrears", "water_supply"),
        ("find garbage truck daily timing", "solid_waste"),
        ("property tax penalty due date", "revenue"),
        ("contact the local ward officer or corporator", "revenue"),
        ("which department handles potholes", "roads")
    ]
    suffixes = ["?", "please.", "plz tell bro", "any direct link?", "urgent question!!", "website link dena?", "help desk number?", "kidhar milega form?"]
    noise = ["thanks", "anyone?", "fast reply needed", ""]
    
    intros = ["Hey bmc", "Hello", "Dear Sir/Madam,", "Question:", "@PMCPune", ""]
    combos = list(product(intros, starts, topics, suffixes, noise))
    for i, st, (top, dept), suf, n in combos:
        res.append((f"{i} {st} {top} {suf} {n}".strip(), dept, "query"))
    return res

def generate_service_request():
    res = []
    intros = ["Request for", "Please arrange", "We need", "I am formally applying for", "Require", "Arrange", "Please provide", "application submitted for"]
    services = [
        ("tree pruning and heavy branch cutting", "parks"),
        ("malba and construction debris lifting from pavement", "solid_waste"),
        ("mosquito fogging in entire residential society", "health"),
        ("new garbage bins provision for our gully", "solid_waste"),
        ("drinking water tanker delivery", "water_supply"),
        ("choked sewage tank machine cleaning", "sewerage"),
        ("daily road sweeping service", "solid_waste"),
        ("dead animal carcass pickup", "health"),
        ("pest control chemical spraying", "health")
    ]
    locations = ["in my society", "near public park", "ward 10", "immediately", "at layout gate", "here", "in sector 5", "nagar", "main road"]
    politeness = ["please.", "thanks in advance.", "kindly do it.", "kripya dhyan de", "jaldi bhejo team pls", "paying heavy taxes for this", "very urgent request."]
    
    prefixes = ["Hello", "Dear Sir,", "To ward officer:", "", "@corporator"]
    combos = list(product(prefixes, intros, services, locations, politeness))
    for p, i, (serv, dept), l, pol in combos:
        res.append((f"{p} {i} {serv} {l} {pol}".strip(), dept, "service_request"))
    return res

def generate_complaints():
    res = []
    subs = ["I am extremely angry", "Worst city management", "Horrible infra", "Complaint regarding", "Reporting total failure of", "Sick of this", "kyahai yeh mcd", "Disgusting neglect"]
    issues = [
        ("huge crater potholes cracking my car suspension", "roads"),
        ("zero water supply since 4 straight days", "water_supply"),
        ("garbage dump overflowing smelling like hell", "solid_waste"),
        ("street lights totally dead pitch black", "electricity"),
        ("drainage clogged dirty smelly water covering road", "sewerage"),
        ("kachra gaadi hasn't come for a week", "solid_waste"),
        ("paani ki main pipe toot gayi hai drinking water leaks", "water_supply"),
        ("sadak pe itne gaddhe hai gaadi chalana muskil", "roads"),
        ("gutter floating on streets pure sewage", "sewerage"),
        ("illegal massive encroachment on footpaths", "revenue")
    ]
    locations = ["in kalyan east", "at mg road signal", "entire ward 4", "right outside my house door", "in entire IT sector", "all over this garbage city"]
    endings = ["do your goddamn job", "kuch toh sharam karo bmc", "bmc is totally blind", "solve it officially", "we are paying taxes for what?", "shameful reality", "wtf administration", "please fix immediately"]
    
    extra = ["", "Listen here,", "@mayor", "@bmc", "Every single day same painful issue:"]
    combos = list(product(extra, subs, issues, locations, endings))
    for e, s, (iss, dept), l, end in combos:
        res.append((f"{e} {s} {iss} {l} {end}".strip(), dept, "complaint"))
    return res

def main():
    print("Generating 100k Hyper-Realistic Hinglish dataset...")
    
    emerg = generate_emergency()
    status = generate_status_check()
    query = generate_query()
    service = generate_service_request()
    complaint = generate_complaints()
    
    def sample_exactly(lst, n):
        if len(lst) >= n:
            return random.sample(lst, n)
        else:
            return random.choices(lst, k=n)

    NUM = 20000
    final_data = sample_exactly(emerg, NUM) + sample_exactly(status, NUM) + \
                 sample_exactly(query, NUM) + sample_exactly(service, NUM) + \
                 sample_exactly(complaint, NUM)
                 
    random.shuffle(final_data)
    df = pd.DataFrame(final_data, columns=["text", "department", "intent"])
    
    # ADVANCED NOISE INJECTION (Mimicking real internet typing behavior)
    def add_extreme_realism(txt):
        txt = txt.lower() if random.random() > 0.4 else txt
        
        # Abbreviation
        txt = txt.replace("please", "plz") if random.random() > 0.2 else txt
        txt = txt.replace("you", "u") if random.random() > 0.3 else txt
        txt = txt.replace("are", "r") if random.random() > 0.7 else txt
        txt = txt.replace("because", "bcz") if random.random() > 0.8 else txt
        
        # Typographical errors (simulating fast mobile typing)
        if random.random() > 0.85:
            txt = txt.replace("water", "wter").replace("garbage", "garbge").replace("road", "rod")
            
        # Punctuation chaos
        if random.random() > 0.8:
            txt = txt + " !!!"
        if random.random() > 0.85:
            txt = txt + " ??"
            
        return txt
        
    df["text"] = df["text"].apply(add_extreme_realism)
    
    # Clean double spaces
    df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True)
    
    out_dir = Path("/home/mahesh/Desktop/mpr_latest/data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "complaints_labeled.csv"
    
    df.to_csv(out_file, index=False)
    
    print(f"✓ Generated {len(df)} heavily chaotic, real-world rows.")
    print("Class Balance:")
    print(df["intent"].value_counts())
    print("\n✓ Saved directly to", out_file)
    
if __name__ == "__main__":
    main()
