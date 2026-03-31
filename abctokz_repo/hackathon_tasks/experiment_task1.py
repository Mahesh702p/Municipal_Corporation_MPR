from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, unigram_multilingual, wordlevel_multilingual
import tempfile
from pathlib import Path

# Corpus WITHOUT the mantra 
corpus = [
    "hello world tokenization is interesting",
    "नमस्ते दुनिया भारत एक महान देश है",
    "मराठी भाषेत स्वागत आहे",
    "subword modeling is crucial for NLP in multiple scripts",
    "the quick brown fox jumps over the lazy dog",
    "एक दो तीन चार पांच छह सात आठ नौ दस",
    "this hackathon requires deep architectural understanding",
    "देवों के देव महादेव",
    "संस्कृत भाषा बहुत पुरानी है"
] * 100

mantra = "ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥"

with tempfile.TemporaryDirectory() as tmp:
    corpus_file = Path(tmp) / "corpus.txt"
    corpus_file.write_text("\n".join(corpus), encoding="utf-8")
    
    experiments = [
        ("WordLevel", 300, wordlevel_multilingual(vocab_size=300)),
        ("BPE", 300, bpe_multilingual(vocab_size=300)),
        ("Unigram", 300, unigram_multilingual(vocab_size=300))
    ]
    
    print(f"INPUT MANTRA: {mantra}\n")
    print(f"{'Model':<10} | {'Vocab':<6} | {'# Tokens':<8} | {'Sample Output Tokens'}")
    print("-" * 80)
    
    for model_name, vocab_size, config in experiments:
        try:
            tokenizer = Tokenizer.from_config(config)
            tokenizer.train([str(corpus_file)], config)
            enc = tokenizer.encode(mantra)
            # Show first 5 tokens
            sample_tokens = str(enc.tokens[:5])
            print(f"{model_name:<10} | {vocab_size:<6} | {len(enc.tokens):<8} | {sample_tokens}")
        except Exception as e:
            print(f"{model_name:<10} | {vocab_size:<6} | ERROR    | {str(e)}")
