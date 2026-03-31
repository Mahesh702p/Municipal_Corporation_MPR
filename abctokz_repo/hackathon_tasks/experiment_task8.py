import unicodedata
from abctokz.normalizers.devanagari import DevanagariNormalizer
from abctokz.pretokenizers.devanagari_aware import DevanagariAwarePreTokenizer

def to_hex(text):
    return " ".join(f"U+{ord(c):04X}" for c in text)

def audit_phrase(label, phrase, norm, ptok):
    print(f"--- {label} ---")
    print(f"Raw: \"{phrase}\"")
    print(f"Hex: {to_hex(phrase)}")
    
    normalized = norm.normalize(phrase)
    print(f"\nNormalized: \"{normalized}\"")
    print(f"Hex: {to_hex(normalized)}")
    print(f"Identical? {phrase == normalized}")
    
    pretokens = ptok.pre_tokenize(normalized)
    print(f"\nPre-tokens: {pretokens}")
    print("-" * 30 + "\n")

# Initialize components
norm = DevanagariNormalizer(nfc_first=True, strip_zero_width=False)
ptok = DevanagariAwarePreTokenizer(split_on_whitespace=True, split_on_script_boundary=True)

# Phrases from Task 8
p1 = "आयो लाल, सभई चायो, झूलेलाल!"
p2 = "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"

print("--- EXPERIMENT: NORMALIZER DEEP DIVE ---\n")
audit_phrase("Sindhi Phrase", p1, norm, ptok)
audit_phrase("Marathi Phrase", p2, norm, ptok)

# Advanced Experiment: The ZWJ Risk
# In Devanagari, ZWJ (U+200D) is used to prevent full-halant form
# Example: 'क' + '्' + ZWJ (creates half-ka)
half_ka_with_zwj = "क्\u200d" 
print("--- ADVANCED: ZWJ STRIPPING RISK ---")
print(f"Input with ZWJ: \"{half_ka_with_zwj}\" ({to_hex(half_ka_with_zwj)})")

norm_strip = DevanagariNormalizer(strip_zero_width=True)
norm_keep = DevanagariNormalizer(strip_zero_width=False)

res_strip = norm_strip.normalize(half_ka_with_zwj)
res_keep = norm_keep.normalize(half_ka_with_zwj)

print(f"Normalized (Strip=True):  \"{res_strip}\" ({to_hex(res_strip)})")
print(f"Normalized (Strip=False): \"{res_keep}\" ({to_hex(res_keep)})")
