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

# 1. Setup small English-only training data
en_corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, tokenization is fun.",
    "Python is a great programming language.",
    "NLP stands for Natural Language Processing."
]
corpus_file = current_dir / "corpus_en_only.txt"
corpus_file.write_text("\n".join(en_corpus), encoding="utf-8")

def run_unk_test(model_type):
    print(f"\n--- Testing {model_type.upper()} ---")
    
    if model_type == "wordlevel":
        model_cfg = WordLevelConfig(vocab_size=100)
        trainer_cfg = WordLevelTrainerConfig(vocab_size=100, min_frequency=1)
    elif model_type == "bpe":
        model_cfg = BPEConfig(vocab_size=100)
        trainer_cfg = BPETrainerConfig(vocab_size=100, min_frequency=1)
    elif model_type == "unigram":
        model_cfg = UnigramConfig(vocab_size=100)
        trainer_cfg = UnigramTrainerConfig(vocab_size=100)
        
    config = TokenizerConfig(
        model=model_cfg,
        trainer=trainer_cfg,
        normalizer=DevanagariNormalizerConfig(),
        pretokenizer=DevanagariAwarePreTokenizerConfig()
    )
    
    tokenizer = AugenblickTokenizer.from_config(config)
    tokenizer.train([str(corpus_file)], config)
    
    # Test cases
    scenarios = {
        "Vocabulary Gap": "antidisestablishmentarianism",
        "Script Mismatch": "भारत महान है",
        "Emoji/Symbol": "🚀🔥✨",
    }
    
    for name, text in scenarios.items():
        enc = tokenizer.encode(text)
        print(f"Scenario: {name}")
        print(f"  Input:  {text}")
        print(f"  Tokens: {enc.tokens}")
        print(f"  IDs:    {enc.ids}")

if __name__ == "__main__":
    run_unk_test("wordlevel")
    run_unk_test("bpe")
    run_unk_test("unigram")
