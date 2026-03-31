import sys
from pathlib import Path

# Add src to sys.path
current_dir = Path(__file__).parent.resolve()
SRC_ROOT = (current_dir.parent / "src").resolve()
sys.path.append(str(SRC_ROOT))

from abctokz import AugenblickTokenizer
from abctokz.config.schemas import (
    TokenizerConfig, 
    BPEConfig, BPETrainerConfig,
    DevanagariNormalizerConfig,
    DevanagariAwarePreTokenizerConfig
)
from abctokz.eval.metrics import fertility

# 1. Anthem Text
anthem_en = "Jana Gana Mana Adhinayaka Jaya He Bharata Bhagya Vidhata Punjab Sindhu Gujarat Maratha Dravida Utkala Banga"
anthem_hi = "जन गण मन अधिनायक जय हे भारत भाग्य विधाता पंजाब सिंधु गुजरात मराठा द्राविड उत्कल बंग"

# 2. Training Corpus (Mixed English and Hindi)
corpus_data = [
    anthem_en,
    anthem_hi,
    "India is my country and all Indians are my brothers and sisters.",
    "भारत मेरा देश है और सभी भारतीय मेरे भाई-बहन हैं।",
    "I love my country and I am proud of its rich and varied heritage.",
    "मुझे अपने देश से प्यार है और मुझे इसकी समृद्ध और विविध विरासत पर गर्व है।",
    "Unity in diversity is the hallmark of India.",
    "विविधता में एकता भारत की पहचान है।",
] * 20 # Enough for 400 vocab size

corpus_file = current_dir / "corpus_task3.txt"
corpus_file.write_text("\n".join(corpus_data), encoding="utf-8")

# 3. Train Tokenizer
VOCAB_SIZE = 400

config = TokenizerConfig(
    model=BPEConfig(vocab_size=VOCAB_SIZE),
    trainer=BPETrainerConfig(vocab_size=VOCAB_SIZE, min_frequency=1),
    normalizer=DevanagariNormalizerConfig(),
    pretokenizer=DevanagariAwarePreTokenizerConfig()
)

tokenizer = AugenblickTokenizer.from_config(config)
tokenizer.train([str(corpus_file)], config)

# 4. Encode
enc_en = tokenizer.encode(anthem_en)
enc_hi = tokenizer.encode(anthem_hi)

# 5. Word counts
word_count_en = len(anthem_en.split())
word_count_hi = len(anthem_hi.split())

# 6. Fertility (using local calculation for clarity or the library function)
fert_en = fertility([enc_en], [word_count_en])
fert_hi = fertility([enc_hi], [word_count_hi])

# 7. Output results
results = {
    "en": {
        "tokens": enc_en.tokens,
        "token_count": len(enc_en.tokens),
        "word_count": word_count_en,
        "fertility": fert_en
    },
    "hi": {
        "tokens": enc_hi.tokens,
        "token_count": len(enc_hi.tokens),
        "word_count": word_count_hi,
        "fertility": fert_hi
    }
}

import json
print("\n--- RESULTS JSON ---")
print(json.dumps(results, indent=2, ensure_ascii=False))

# Bonus: OpenAI Tiktoken comparison (if possible)
try:
    import tiktoken
    enc_tk = tiktoken.get_encoding("cl100k_base")
    tk_en_ids = enc_tk.encode(anthem_en)
    tk_hi_ids = enc_tk.encode(anthem_hi)
    
    print("\n--- TIKTOKEN BONUS ---")
    print(f"EN: {len(tk_en_ids)} tokens, Fertility: {len(tk_en_ids)/word_count_en:.4f}")
    print(f"HI: {len(tk_hi_ids)} tokens, Fertility: {len(tk_hi_ids)/word_count_hi:.4f}")
except ImportError:
    print("\n[Tiktoken not installed, skipping bonus comparison]")
