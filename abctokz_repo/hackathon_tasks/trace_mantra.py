import sys
from pathlib import Path

# Add src to sys.path
current_dir = Path(__file__).parent.resolve()
SRC_ROOT = (current_dir.parent / "src").resolve()
sys.path.append(str(SRC_ROOT))

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual
import tempfile

# Create a small multilingual corpus for training
corpus = [
    "hello world tokenization is interesting",
    "नमस्ते दुनिया भारत एक महान देश है",
    "मराठी भाषेत स्वागत आहे",
    "ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥",
    "subword modeling is crucial for NLP in multiple scripts"
] * 20 

print("--- TASK 1: PIPELINE TRACE ---")

with tempfile.TemporaryDirectory() as tmp:
    corpus_file = Path(tmp) / "corpus.txt"
    corpus_file.write_text("\n".join(corpus), encoding="utf-8")
    
    # 1. Config (BPE, Vocab=300)
    config = bpe_multilingual(vocab_size=300)
    
    # 2. Train
    tokenizer = Tokenizer.from_config(config)
    tokenizer.train([str(corpus_file)], config)
    
    # The Mantra to Trace
    mantra = "ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥"
    print(f"\n[0] RAW INPUT STR: {repr(mantra)}")
    
    # Stage 1: Normalizer
    normalized = tokenizer._normalizer.normalize(mantra)
    print(f"\n[1] AFTER NORMALIZER: {repr(normalized)}")
    print(f"    Normalized changed text? {mantra != normalized}")
    
    # Stage 2: PreTokenizer
    pre_tokens = tokenizer._pretokenizer.pre_tokenize(normalized)
    print(f"\n[2] AFTER PRE-TOKENIZER (Chunks):")
    for pt in pre_tokens[:5]: # print first 5 for brevity
        print(f"    {repr(pt)}")
    print(f"    ... and {len(pre_tokens)-5} more chunks.")
    
    # Stage 3: Model Tokenization
    print(f"\n[3] AFTER BPE MODEL (Subword Pieces):")
    all_pieces = []
    for pt in pre_tokens:
        toks = tokenizer._model.tokenize(pt)
        pieces = [t for t, _ in toks]
        all_pieces.extend(pieces)
        if len(all_pieces) <= 15: # Just show the first few
            print(f"    {repr(pt):<15} -> {pieces}")
            
    # Final Encoding Output
    enc = tokenizer.encode(mantra)
    print(f"\n[*] FINAL ENCODING OBJECT:")
    print(f"    Tokens: {enc.tokens[:10]} ...")
    print(f"    IDs:    {enc.ids[:10]} ...")
    
    # Stage 4: Decoder
    decoded = tokenizer.decode(enc.ids)
    print(f"\n[4] AFTER DECODER (Reconstruction):")
    print(f"    {repr(decoded)}")
    print(f"    Perfect roundtrip? {mantra == decoded}")
