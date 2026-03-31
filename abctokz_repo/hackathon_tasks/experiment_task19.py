import sys
from pathlib import Path

# Add src to sys.path
current_dir = Path(__file__).parent.resolve()
SRC_ROOT = (current_dir.parent / "src").resolve()
sys.path.append(str(SRC_ROOT))

from abctokz import AugenblickTokenizer
from abctokz.config.schemas import (
    TokenizerConfig, 
    WordLevelConfig, WordLevelTrainerConfig,
    BPEConfig, BPETrainerConfig,
    UnigramConfig, UnigramTrainerConfig,
    DevanagariNormalizerConfig,
    DevanagariAwarePreTokenizerConfig
)

# 1. Define Corpus
corpus_data = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization algorithms affect language model efficiency.",
    "Artificial intelligence is transforming the linguistic landscape.",
    "Natural language processing requires robust subword modeling.",
    "भारत महान है",
    "नमस्ते दुनिया",
    "गणपती बप्पा मोरया पुढच्या वर्षी लवकर या",
    "AI भारत में तेजी से बढ़ रहा है",
    "हमें अपनी भाषा पर गर्व होना चाहिए",
    "Modern NLP handles multiple scripts including Devanagari.",
    "Hindi and English are often mixed in casual conversation.",
    "संविधान में भारतीय भाषाओं का महत्व बताया गया है।",
    "तंत्रज्ञान वेगाने बदलत आहे आणि आपणही बदलले पाहिजे.",
] * 50  # Repeat to give trainer enough data

corpus_file = current_dir / "corpus_task19.txt"
corpus_file.write_text("\n".join(corpus_data), encoding="utf-8")

# 2. Test Sentences
test_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization algorithms affect language model efficiency.",
    "भारत महान है",
    "गणपती बप्पा मोरया पुढच्या वर्षी लवकर या",
    "AI भारत में तेजी से बढ़ रहा है"
]

VOCAB_SIZE = 400

def train_and_eval(model_type):
    if model_type == "wordlevel":
        model_cfg = WordLevelConfig(vocab_size=VOCAB_SIZE)
        trainer_cfg = WordLevelTrainerConfig(vocab_size=VOCAB_SIZE, min_frequency=1)
    elif model_type == "bpe":
        model_cfg = BPEConfig(vocab_size=VOCAB_SIZE)
        trainer_cfg = BPETrainerConfig(vocab_size=VOCAB_SIZE, min_frequency=1)
    elif model_type == "unigram":
        model_cfg = UnigramConfig(vocab_size=VOCAB_SIZE)
        trainer_cfg = UnigramTrainerConfig(vocab_size=VOCAB_SIZE)
    
    config = TokenizerConfig(
        model=model_cfg,
        trainer=trainer_cfg,
        normalizer=DevanagariNormalizerConfig(),
        pretokenizer=DevanagariAwarePreTokenizerConfig()
    )
    
    tokenizer = AugenblickTokenizer.from_config(config)
    tokenizer.train([str(corpus_file)], config)
    
    results = []
    total_tokens = 0
    total_words = 0
    
    for sent in test_sentences:
        enc = tokenizer.encode(sent)
        results.append(enc.tokens)
        total_tokens += len(enc.tokens)
        total_words += len(sent.split())
        
    fertility = total_tokens / total_words if total_words > 0 else 0
    
    # Check UNK rate on a totally new unseen sentence
    unseen_text = "Space exploration is the next frontier for humanity."
    unk_enc = tokenizer.encode(unseen_text)
    # UNK_ID is usually 0
    unk_count = unk_enc.ids.count(0)
    unk_rate = unk_count / len(unk_enc.tokens) if len(unk_enc.tokens) > 0 else 0
    
    # Vocab analysis
    vocab = tokenizer.get_vocab()
    # Categorize tokens - simple length and prefix check
    chars = [t for t in vocab if len(t) == 1]
    subwords = [t for t in vocab if len(t) > 1 and (t.startswith("##") or t.startswith("▁"))]
    words = [t for t in vocab if len(t) > 1 and not (t.startswith("##") or t.startswith("▁"))]
    
    return {
        "tokens": results,
        "fertility": fertility,
        "unk_rate": unk_rate,
        "vocab_stats": {"chars": len(chars), "subwords": len(subwords), "words": len(words)}
    }

models = ["wordlevel", "bpe", "unigram"]
final_comparison = {}

for m in models:
    try:
        print(f"Training {m}...")
        final_comparison[m] = train_and_eval(m)
    except Exception as e:
        print(f"Error training {m}: {e}")

# 3. Output results for the report
import json
print("\n--- RESULTS JSON ---")
print(json.dumps(final_comparison, indent=2, ensure_ascii=False))
