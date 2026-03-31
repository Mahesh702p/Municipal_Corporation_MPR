"""
generate_synthetic_data.py
==========================
Generates REALISTIC, NOISY labeled complaint data for the Municipal Corporation MLM.

Noise layers applied to simulate real citizen complaints:
  1. Typos & phonetic misspellings ("pothole" → "potthole", "ward" → "wadr")
  2. Missing punctuation & all-lowercase
  3. Abbreviations (MC, NMC, BMC, wp, plz, pls, thnk)
  4. Frustrated/emotional tone ("STILL NOT FIXED!!!", "this is ridiculous")
  5. Ambiguous cross-department complaints (water + drain, road + light)
  6. Incomplete sentences ("road bad. no fix.")
  7. Random code-switching mid-sentence
  8. Redundant/filler words ("like", "basically", "u see")
  9. Random capitalization & extra spaces
  10. Multiple complaints in one text

Output:
  data/processed/complaints_labeled.csv   ~15,000+ rows
  data/processed/pretrain_corpus.txt

Run: python preprocessing/generate_synthetic_data.py
"""

import csv
import os
import random
import re

random.seed(42)

OUT_DIR = "data/processed"
OUT_LABELED = os.path.join(OUT_DIR, "complaints_labeled.csv")
OUT_CORPUS = os.path.join(OUT_DIR, "pretrain_corpus.txt")

# ─────────────────────────────────────────
# NOISE HELPERS
# ─────────────────────────────────────────

TYPO_MAP = {
    "pothole": ["potthole", "pothole", "pot hole", "phole", "potholee"],
    "garbage": ["garbge", "grabage", "garbaje", "garbaage", "garbage"],
    "water": ["watr", "wter", "wateer", "wator"],
    "ward": ["wadr", "wrad", "wardd", "wrd"],
    "road": ["raod", "roaad", "roda", "rood"],
    "light": ["ligh", "ligt", "ligth", "lgiht"],
    "drain": ["drainn", "drian", "darin"],
    "property": ["proprty", "proerty", "propety"],
    "complaint": ["complain", "compliant", "compaint"],
    "municipal": ["muicipal", "municpal", "municiple"],
    "certificate": ["certifcate", "certficate", "certiifcate"],
    "illegal": ["ilegal", "illegall", "illegl"],
    "construction": ["costrution", "constuction", "constraction"],
    "collected": ["colected", "collectd", "colectd"],
    "not": ["nt", "nto", "noot"],
    "please": ["plz", "pls", "plese", "pleese"],
    "since": ["sinc", "snce", "sine"],
    "days": ["dyas", "dayz", "day"],
}

FILLER_WORDS = [
    "", "", "", "", "",  # most have no filler
    "tbh ", "u see ", "like ", "basically ", "actually ",
    "I mean ", "you know ", "bhai ", "yaar ", "sir ",
]

FRUSTRATION_SUFFIXES = [
    "", "", "", "", "", "",  # most end normally
    " This is ridiculous!",
    " STILL not resolved!!!",
    " When will this be fixed??",
    " We are very tired of this.",
    " Please take urgent action!!!",
    " Nobody is listening to us.",
    " This has been going on too long.",
    " Disgusting situation.",
    " Very disappointed with MC.",
    " Kab tak wait karein???",
    " कब होगा काम???",
]

OPENERS = [
    "", "", "", "",
    "Hello sir, ",
    "Dear MC, ",
    "To whom it may concern, ",
    "Respected officer, ",
    "Hi, ",
    "Urgent complaint: ",
    "This is to inform you that ",
    "I want to report that ",
    "We the residents of {ward} want to say that ",
    "It's been {dur} and still ",
    "Despite our previous complaints, ",
]

# ─────────────────────────────────────────
# SLOT VALUES (more realistic variety)
# ─────────────────────────────────────────

WARDS = (
    [f"ward {i}" for i in range(1, 50)] +
    [f"ward no {i}" for i in range(1, 30)] +
    [f"W-{i}" for i in range(1, 20)] +
    ["Andheri East", "Bandra West", "Koramangala", "Indiranagar",
     "Shivaji Nagar", "Pimpri", "Hadapsar", "Malviya Nagar",
     "Laxmi Nagar", "Karol Bagh", "Borivali", "Thane", "Vile Parle",
     "Kurla", "Mulund", "Powai", "Dharavi", "Govandi", "Chembur",
     "Santacruz", "Matunga", "Wadala", "Sion", "Vikhroli"]
)

DURATIONS = [
    "since 3 days", "for the past week", "since yesterday",
    "for 2 days", "since last month", "for 5 days",
    "since last night", "for 10 days", "since the monsoon",
    "3 din se", "1 hafte se", "kaafi dino se", "महीने भर से",
    "from 4 days", "past 2 weeks", "over a fortnight",
    "since monday", "from last tuesday", "since the rain",
    "since diwali", "past 6 hours", "since morning",
]

AREAS = [
    "near the school", "near the market", "at the junction",
    "outside my house", "in our colony", "near the temple",
    "at the main road", "in the lane", "near the park",
    "near station", "opposite the hospital", "at the corner",
    "at plot no 45", "near sector 7", "behind the mall",
    "in the bylane", "at the signal", "near D-block",
    "in front of my building", "at the naka",
]

AMOUNTS = ["₹5000", "₹12000", "₹8500", "₹25000", "₹3500", "some amount"]

# ─────────────────────────────────────────
# NOISE FUNCTIONS
# ─────────────────────────────────────────

def apply_typos(text: str) -> str:
    """Randomly apply typos to a few words (20% chance per word)."""
    words = text.split()
    out = []
    for w in words:
        wl = w.lower().rstrip(".,!?")
        if wl in TYPO_MAP and random.random() < 0.25:
            out.append(random.choice(TYPO_MAP[wl]))
        else:
            out.append(w)
    return " ".join(out)


def random_case(text: str) -> str:
    """Randomly lowercase, UPPERCASE parts, or leave as-is."""
    r = random.random()
    if r < 0.55:
        return text.lower()
    elif r < 0.70:
        return text  # normal
    elif r < 0.80:
        # Capitalize first word only
        return text[0].upper() + text[1:].lower() if text else text
    else:
        # Random words in uppercase (like shouting)
        words = text.split()
        return " ".join(
            w.upper() if random.random() < 0.15 else w for w in words
        )


def strip_punctuation_randomly(text: str) -> str:
    """Sometimes remove ending punctuation."""
    if random.random() < 0.4:
        text = text.rstrip(".,!?;")
    return text


def add_extra_spaces(text: str) -> str:
    """Occasionally add extra spaces (WhatsApp style)."""
    if random.random() < 0.15:
        words = text.split()
        out = []
        for w in words:
            out.append(w)
            if random.random() < 0.1:
                out.append("")  # extra space
        return " ".join(out)
    return text


def apply_abbreviations(text: str) -> str:
    """Replace some words with common abbreviations."""
    abbrevs = {
        "municipal corporation": random.choice(["MC", "BMC", "NMC", "BBMP", "mc"]),
        "please": random.choice(["plz", "pls", "please"]),
        "department": random.choice(["dept", "dept.", "deptt"]),
        "complaint": random.choice(["complaint", "complain", "cmplt"]),
        "urgent": random.choice(["urgent", "URGENT", "urgnt"]),
    }
    for full, short in abbrevs.items():
        if full in text.lower() and random.random() < 0.35:
            text = re.sub(full, short, text, flags=re.IGNORECASE, count=1)
    return text


def get_opener(ward: str, dur: str) -> str:
    """Return a random opener string."""
    opener = random.choice(OPENERS)
    try:
        return opener.format(ward=ward, dur=dur)
    except Exception:
        return opener


def noisify(text: str) -> str:
    """Apply all noise layers to a complaint."""
    text = apply_abbreviations(text)
    text = apply_typos(text)
    text = random_case(text)
    text = strip_punctuation_randomly(text)
    text = add_extra_spaces(text)
    filler = random.choice(FILLER_WORDS)
    suffix = random.choice(FRUSTRATION_SUFFIXES)
    return (filler + text + suffix).strip()


# ─────────────────────────────────────────
# TEMPLATES — much more varied & realistic
# ─────────────────────────────────────────

TEMPLATES = {
    "Water Supply": {
        "Water Shortage": {
            "en": [
                "no water in {ward} {dur}",
                "water supply stopped {ward} {dur} residents suffering badly",
                "water not coming from tap {area} {ward}",
                "no water since {dur} in {ward} when will tanker come",
                "daily water supply disrupted {ward} past {dur} pipeline blocked maybe",
                "municipal tanker not coming to {ward} {dur}",
                "tap is dry {ward} {dur} please check the pipeline",
                "water pressure very low {area} {ward} {dur} barely enough",
                "we r not getting water {ward} 4 {dur} wat is happening",
                "supply cut without notice {ward} {dur}",
                "my building {area} {ward} has no water from {dur} what to do",
            ],
            "hinglish": [
                "paani nahi aa raha {ward} mein {dur} se",
                "{ward} mein paani band hai {dur} se log taras rahe hain",
                "bhai {ward} me paani nahi chal raha kuch karo",
                "paani ki problem hai {ward} mein {dur} se nala band hogaya",
                "aaj bhi {ward} mein paani nahi aaya tanker bhejo",
                "kab aayega paani {ward} mein itna wait kar rahe hain",
                "paani ki bohot problem hai {area} mein {ward} ke andar",
                "{ward} ka paani 3 din se nahi aa rha mujhe kab milega",
                "mom ko heart problem hai isliye urgent chahiye paani {ward}",
                "nahi aa rha paani {ward} mein please koi to suno",
            ],
            "hi": [
                "{ward} में {dur} से पानी की आपूर्ति बन्द है।",
                "{ward} में पेयजल नहीं आ रहा है कृपया कार्रवाई करें",
                "{ward} क्षेत्र में जल आपूर्ति {dur} से बाधित है",
                "नल में पानी नहीं आ रहा {ward} में {dur} से",
                "{ward} में पानी न आने से बहुत परेशानी हो रही है",
                "हमारे {ward} में {dur} से पानी नहीं है जल विभाग ध्यान दे",
            ],
        },
        "Water Leakage": {
            "en": [
                "major pipe burst {area} {ward} water wasting",
                "pipe leaking {ward} {area} road flooded",
                "water leakage {area} {ward} huge wastage please fix",
                "main pipeline burst {ward} water gushing out {dur}",
                "pipe leaking outside {area} {ward} creating pothole also",
                "water pipe cracked {ward} {area} road become muddy",
                "big water leak {ward} {dur} nobody coming to fix",
            ],
            "hinglish": [
                "pipe phoot gayi {ward} {area} mein paani beh raha hai",
                "{ward} mein paani ka pipe {dur} se leak ho raha hai",
                "bhari water leakage {area} sadak bhar gayi",
                "pipe toot gayi {area} {ward} ke paas jaldi aao",
            ],
            "hi": [
                "{ward} में {area} पाइप फटा हुआ है पानी बह रहा है",
                "{ward} की सड़क पर पाइप लीकेज से पानी भर गया",
            ],
        },
        "Contaminated Water": {
            "en": [
                "water from tap {ward} yellow coloured not drinkable",
                "dirty smelly water supply {ward} {dur} health risk",
                "brown water coming {ward} children getting sick",
                "water looks muddy {ward} {area} {dur}",
                "water has weird smell and colour {ward}",
                "water filter getting choked daily {ward} water very bad quality",
            ],
            "hinglish": [
                "{ward} mein ganda paani aa raha hai peene layak nahi",
                "paani mein mitti aa rahi hai {ward} colour bhi change hai",
                "{ward} ka paani smell kar raha hai bilkul nahi peete",
            ],
            "hi": [
                "{ward} में गंदा दूषित जल आ रहा है पीने योग्य नहीं",
                "{ward} में पानी में मिट्टी आ रही है {dur} से",
            ],
        },
    },

    "Engineering": {
        "Road Pothole": {
            "en": [
                "huge potholes {area} {ward} very dangerous",
                "road full of potholes {ward} {dur} vehicles getting damaged",
                "pothole {area} {ward} bike accident happened yesterday",
                "road condition pathetic {ward} multiple potholes {dur}",
                "potholes on main road {ward} causing accidents {dur}",
                "road broken badly {ward} {area} car tyres punctured 2 times",
                "my scooter fell in pothole {ward} {area} i got hurt",
                "road dug up for construction never repaired {ward} {dur}",
                "road tar came off {dur} {ward} only stones left",
                "it rained and entire road disappeared {ward} only pits left",
                "massive crater sized pothole {area} {ward} since {dur}",
            ],
            "hinglish": [
                "{ward} mein sadak pe bade khadde hain {dur} se bahut takleef",
                "road pe gadd hai {area} {ward} bike wale girte hain",
                "{ward} ki road bilkul kharab hai kab banoge",
                "khadda hai {area} {ward} mein accident hua tha kal",
                "road repair karo {ward} mein kitne din se khadde hain yaar",
                "aaj fir gira mai {ward} ke khadde mein scooter le gaya neeche",
                "{ward} ki sadak me itne khadde hain ki raat ko dikh bhi nahi",
            ],
            "hi": [
                "{ward} में सड़क पर बड़े-बड़े गड्ढे हैं वाहन चलाना खतरनाक",
                "{ward} की सड़क की हालत खराब है {dur} से मरम्मत नहीं",
                "{area} {ward} में गड्ढे से दुर्घटना का खतरा है",
                "गड्ढे की वजह से {ward} में दो लोग गिरे अब तक नहीं भरा",
            ],
        },
        "Street Light": {
            "en": [
                "street light not working {ward} {area} {dur} very dark at night",
                "no streetlight {ward} {area} theft increasing",
                "all street lamps broken {ward} completely dark",
                "light pole broken {ward} dangerous live wire exposed",
                "street light wire hanging loose {area} {ward} shock risk",
                "entire {ward} in darkness after 9pm {dur}",
                "1 light working out of 10 in {ward} rest all dead",
            ],
            "hinglish": [
                "{ward} mein light nahi hai raat ko andhera bahut hota hai",
                "street light toot gayi {area} {ward} {dur} se",
                "{ward} ki light band hai chor ka dar lag raha hai sach mein",
                "wire latki hui hai {area} shock lag sakta hai kisi ko",
            ],
            "hi": [
                "{ward} में स्ट्रीट लाइट {dur} से बंद है रात में अंधेरा",
                "{ward} में {area} के पास खंभे का तार लटक रहा है खतरनाक",
            ],
        },
        "Drainage": {
            "en": [
                "drain blocked {ward} {area} sewage overflowing on road",
                "drain choked {ward} {dur} foul smell health hazard",
                "open manhole {area} {ward} very dangerous especially at night",
                "drainage overflow during rain {ward} flooding streets",
                "nullah blocked {ward} mosquito breeding happening",
                "gutter overflow {ward} people walking in waste water",
                "manhole cover missing {area} {ward} someone will fall",
                "sewage smell unbearable {ward} {dur} kids cant go to school",
            ],
            "hinglish": [
                "nali choke ho gayi {ward} ganda paani sadak pe aa raha",
                "{ward} {area} nali band hai {dur} se badbu aa rahi",
                "manhole open hai {area} koi gir jayega please cover lagao",
                "baarish ke baad drain overflow {ward} mein road nahi dikh raha",
                "gutter bhar gaya {ward} mein log paani mein chal rahe",
            ],
            "hi": [
                "{ward} में नाली बंद है {dur} से सड़क पर गंदा पानी",
                "{ward} में मैनहोल खुला है खतरनाक स्थिति है",
                "नाली से बदबू {ward} में {dur} से असहनीय हो गई है",
            ],
        },
    },

    "SWM": {
        "Garbage Not Collected": {
            "en": [
                "garbage not collected {ward} {dur} unbearable smell",
                "waste not picked {ward} {area} {dur} piling up on road",
                "garbage truck not coming {ward} {dur}",
                "swm worker absent from {ward} {area} {dur}",
                "dustbin overflowing {ward} not emptied {dur}",
                "garbage lying on road {ward} {dur} disease risk",
                "my area {ward} has not seen garbage van since {dur}",
                "stinking garbage pile {area} {ward} dogs spreading it everywhere",
                "we called the garbage number 3 times {ward} no response",
                "garbage collector takes money still not collecting {ward}",
                "{area} {ward} ka kuda uthao yaar {dur} se nahi utha",
            ],
            "hinglish": [
                "kachra nahi utha {ward} mein {dur} se bahut badbu",
                "garbage wala nahi aaya {ward} {area} {dur} se",
                "dustbin bhar gayi {ward} mein khali karo please",
                "{ward} mein kachra sadak pe bimari failegi",
                "safai nahi ho rahi {ward} {dur} se kachra uthao na",
                "kachra gaadi aayi hi nahi {ward} me {dur} se",
                "teen baar bulaya koi nahi aaya {ward} ka kuda uthane",
            ],
            "hi": [
                "{ward} में {dur} से कचरा नहीं उठाया गया बदबू असहनीय",
                "{ward} में कचरा गाड़ी {dur} से नहीं आई सड़क गंदी है",
                "कचरेवाले को बुलाया {ward} में पर कोई नहीं आया",
            ],
        },
        "Stray Animals": {
            "en": [
                "many stray dogs {ward} {area} biting residents",
                "stray cattle blocking road {ward} {dur}",
                "dog attack {ward} {area} child got bitten please help",
                "pack of dogs chasing people {ward} very scary",
                "cow sitting on main road {ward} traffic blocked",
                "stray dogs not sterilized {ward} population increasing",
            ],
            "hinglish": [
                "{ward} mein awaara kutte bahut hain bacchon ko kaat rahe",
                "gaaye sadak pe khadi hain {ward} traffic jam ho raha",
                "{area} {ward} mein kutte bahut aggressive hain raat ko",
                "kal kutte ne kaata ek bacche ko {ward} mein urgent karo",
            ],
            "hi": [
                "{ward} में आवारा कुत्ते लोगों को काट रहे हैं",
                "{ward} में आवारा पशु सड़क रोक रहे हैं",
            ],
        },
        "Illegal Dumping": {
            "en": [
                "people dumping garbage {area} {ward} very unhygienic",
                "illegal dump {area} {ward} flies mosquitoes everywhere",
                "construction waste dumped on road {ward} by builder",
            ],
            "hinglish": [
                "{ward} {area} mein log kachra fek rahe hain gandagi",
                "builder ne maal {area} {ward} road pe daal diya hata do",
            ],
            "hi": [
                "{ward} में {area} कचरा फेंका जा रहा है अवैध",
            ],
        },
    },

    "Public Health": {
        "Dengue/Mosquito": {
            "en": [
                "dengue cases increasing {ward} stagnant water not removed",
                "mosquito problem {ward} {area} no fogging {dur}",
                "malaria fear {ward} drain not cleaned {dur}",
                "mosquito breeding open drain {ward} health risk",
                "3 people in my building got dengue {ward} please spray",
                "waterlogging after rain {ward} mosquito problem getting worse",
            ],
            "hinglish": [
                "{ward} mein dengue ke case aa rahe hain fogging karo",
                "machhar bahut hain {ward} {dur} se spray kab karoge",
                "{area} {ward} mein paani khada hai mosquito breeding",
                "meri beti ko dengue hua {ward} mein ab bhi kuch nahi hua",
            ],
            "hi": [
                "{ward} में डेंगू के मामले बढ़ रहे हैं फॉगिंग जरूरी है",
                "{ward} में मच्छरों की समस्या {dur} से है",
            ],
        },
        "Sanitation": {
            "en": [
                "no public toilet {ward} {area} open defecation happening",
                "public toilet {ward} very dirty not cleaned {dur}",
                "toilet always locked {ward} {area} useless facility",
                "toilet broken and stinking {ward} nobody cleaning",
            ],
            "hinglish": [
                "{ward} mein public toilet nahi hai log bahar jaate hain",
                "toilet gandy hai {area} {ward} {dur} se safai karo",
                "toilet band rehta hai {ward} mein kya fayda",
            ],
            "hi": [
                "{ward} में सार्वजनिक शौचालय {dur} से साफ नहीं हुआ",
                "{ward} में शौचालय बंद रहता है लोग परेशान हैं",
            ],
        },
    },

    "Revenue/Tax": {
        "Property Tax": {
            "en": [
                "paid property tax {ward} but receipt not received",
                "tax demand notice for {ward} but payment done {dur} ago",
                "wrong property tax assessment {ward} property",
                "online property tax payment failed amount {amounts} deducted {ward}",
                "paid {amounts} property tax but still showing dues {ward}",
                "tax notice came again even after paying {ward}",
                "property tax portal down cannot pay {ward}",
            ],
            "hinglish": [
                "{ward} ka property tax bhar diya receipt nahi mili",
                "tax notice aaya hai {ward} ka lekin payment ho chuki",
                "online payment fail ho gayi {ward} mein paise kat gaye",
                "{amounts} bhar diye phir bhi due show ho raha hai {ward}",
            ],
            "hi": [
                "{ward} में संपत्ति कर भुगतान की रसीद नहीं मिली",
                "{ward} का गलत कर आकलन हुआ सुधार करें",
            ],
        },
        "Mutation": {
            "en": [
                "property mutation not done {ward} applied {dur} ago",
                "mutation request pending {ward} {dur} please process",
                "applied for mutation {dur} no update {ward}",
            ],
            "hinglish": [
                "{ward} mein mutation nahi hua {dur} se apply kiya tha",
                "mutation ka koi status nahi mila {ward} ka",
            ],
            "hi": [
                "{ward} में नामांतरण {dur} से लंबित है कब होगा",
            ],
        },
    },

    "Town Planning": {
        "Illegal Construction": {
            "en": [
                "illegal construction {ward} {area} without permission",
                "unauthorized building {ward} please stop",
                "encroachment on footpath {area} {ward} pedestrians cant walk",
                "construction without plan approval {ward}",
                "builder started work at night {ward} {area} no permission",
                "entire footpath occupied by shop {area} {ward} {dur}",
                "neighbor built extra floor without permission {ward}",
            ],
            "hinglish": [
                "{ward} {area} bina permission construction ho raha hai",
                "illegal building ban raha hai {ward} mein rok do",
                "footpath pe encroachment hai {area} {ward} mein",
                "raat ko construction chal raha tha {ward} bina permission",
                "{area} ka footpath dukan ne gheer liya {ward} mein {dur} se",
            ],
            "hi": [
                "{ward} में {area} अवैध निर्माण हो रहा है",
                "{ward} में बिना अनुमति भवन निर्माण हो रहा है",
                "पड़ोसी ने {ward} में बिना अनुमति एक मंजिल बना ली",
            ],
        },
        "Building Permission": {
            "en": [
                "building plan submitted {ward} {dur} no response",
                "commencement certificate not received {ward} despite approval",
                "oc application pending {dur} {ward} office",
                "no response to building permission application {ward} {dur}",
            ],
            "hinglish": [
                "{ward} office mein plan submit kiya {dur} reply nahi",
                "building permission ke liye {dur} wait kar raha hoon {ward}",
            ],
            "hi": [
                "{ward} में भवन निर्माण अनुमति {dur} से लंबित है",
            ],
        },
    },

    "Registration": {
        "Birth Certificate": {
            "en": [
                "birth certificate not received {ward} office {dur}",
                "registered birth {dur} but certificate not issued {ward}",
                "correction needed birth certificate {ward} wrong name spelled",
                "applied birth cert {ward} {dur} back no movement",
                "my baby born {dur} certificate still not ready {ward}",
            ],
            "hinglish": [
                "{ward} office se birth certificate nahi mila {dur} se",
                "bacche ka certificate nahi bana {ward} mein {dur} lag gaye",
                "naam galat likha hai certificate mein {ward} ka",
            ],
            "hi": [
                "{ward} से जन्म प्रमाण पत्र {dur} से नहीं मिला",
                "{ward} में नाम गलत लिखा है प्रमाण पत्र में सुधार करें",
            ],
        },
        "Death Certificate": {
            "en": [
                "death certificate not issued {ward} office {dur}",
                "death registered but certificate pending {ward}",
                "need death certificate urgently {ward} for insurance",
            ],
            "hinglish": [
                "{ward} mein death certificate nahi mila {dur} se urgent hai",
                "maa ka certificate chahiye {ward} office se {dur} ho gaye",
            ],
            "hi": [
                "{ward} से मृत्यु प्रमाण पत्र {dur} से नहीं मिला",
            ],
        },
    },

    "Licensing": {
        "Trade License": {
            "en": [
                "trade license renewal pending {dur} shop {ward}",
                "new trade license not processed {ward} {dur}",
                "fire noc required {ward} not getting it",
                "applied trade license {dur} ago {ward} still waiting",
                "my shop has no license because {ward} office not responding",
            ],
            "hinglish": [
                "{ward} mein dukan ka license renew nahi hua {dur} se",
                "trade license ke liye {dur} se wait kar raha hoon {ward}",
                "{ward} office hi nahi uthata phone trade license ke liye",
            ],
            "hi": [
                "{ward} में व्यापार लाइसेंस {dur} से लंबित है",
            ],
        },
    },

    "Fire Services": {
        "Fire Incident": {
            "en": [
                "fire broke out {area} {ward} fire brigade delayed",
                "cylinder blast {area} {ward} need immediate help",
                "fire at building {ward} {area} please send fire engine",
                "small fire {ward} {area} before it spreads please come",
            ],
            "hinglish": [
                "{ward} {area} mein aag lagi hai fire brigade bulao",
                "cylinder blast hua {area} {ward} mein emergency",
                "building mein aag {ward} jaldi aao",
            ],
            "hi": [
                "{ward} {area} में आग लग गई है दमकल बुलाएं",
                "{ward} में सिलेंडर फटा है तत्काल सहायता चाहिए",
            ],
        },
    },
}


# ─────────────────────────────────────────
# SEVERITY MAP
# ─────────────────────────────────────────
SEVERITY_MAP = {
    "Water Shortage": "HIGH", "Water Leakage": "HIGH", "Contaminated Water": "HIGH",
    "Road Pothole": "MEDIUM", "Street Light": "MEDIUM", "Drainage": "MEDIUM",
    "Garbage Not Collected": "MEDIUM", "Illegal Dumping": "LOW", "Stray Animals": "HIGH",
    "Dengue/Mosquito": "HIGH", "Sanitation": "MEDIUM",
    "Property Tax": "MEDIUM", "Mutation": "LOW",
    "Illegal Construction": "HIGH", "Building Permission": "LOW",
    "Birth Certificate": "MEDIUM", "Death Certificate": "MEDIUM",
    "Trade License": "LOW", "Fire Incident": "HIGH",
    "Road Construction": "MEDIUM",
}


def generate_record(dept, cat, lang, template_text, record_id):
    ward = random.choice(WARDS)
    dur = random.choice(DURATIONS)
    area = random.choice(AREAS)
    amounts = random.choice(AMOUNTS)
    opener = get_opener(ward, dur)

    try:
        text = template_text.format(
            ward=ward, dur=dur, area=area, amounts=amounts
        )
    except KeyError:
        text = template_text

    text = opener + text
    text = noisify(text)
    text = re.sub(r"\s+", " ", text).strip()

    severity = SEVERITY_MAP.get(cat, "MEDIUM")

    return {
        "id": record_id,
        "text": text,
        "clean_text": text,
        "department": dept,
        "category": cat,
        "severity": severity,
        "language": lang,
        "confidence": round(random.uniform(0.75, 1.0), 2),
        "source": "synthetic_noisy",
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    records = []
    record_id = 0
    REPEATS = 60  # ~15k total rows

    for dept, categories in TEMPLATES.items():
        for cat, lang_data in categories.items():
            for lang in ["en", "hinglish", "hi"]:
                if lang not in lang_data:
                    continue
                templates = lang_data[lang]
                for _ in range(REPEATS):
                    for tmpl in templates:
                        rec = generate_record(dept, cat, lang, tmpl, record_id)
                        records.append(rec)
                        record_id += 1

    random.shuffle(records)

    dept_dist = {}
    lang_dist = {}
    for r in records:
        dept_dist[r["department"]] = dept_dist.get(r["department"], 0) + 1
        lang_dist[r["language"]] = lang_dist.get(r["language"], 0) + 1

    print(f"Generated {len(records):,} noisy realistic records")
    print("Department distribution:")
    for d, c in sorted(dept_dist.items(), key=lambda x: -x[1]):
        print(f"  {d:<25} {c:>6}")
    print(f"Language: {lang_dist}")

    fieldnames = ["id", "text", "clean_text", "department", "category",
                  "severity", "language", "confidence", "source"]
    with open(OUT_LABELED, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"✓ Saved → {OUT_LABELED}")

    with open(OUT_CORPUS, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r["text"] + "\n")
    print(f"✓ Corpus → {OUT_CORPUS}")
    print("\nSample records:")
    for r in random.sample(records, 5):
        print(f"  [{r['department']:15}] {r['text'][:80]}")


if __name__ == "__main__":
    main()
