import unicodedata
from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual
import tempfile
from pathlib import Path

def run_round_trip(tok, text):
    encoding = tok.encode(text)
    decoded = tok.decode(encoding.ids)
    print(f"    IDs: {encoding.ids}")
    match = (text == decoded)
    return decoded, match

print("--- EXPERIMENT: ROUND-TRIP VERIFICATION ---\n")

# Setup a small BPE model
with tempfile.TemporaryDirectory() as tmp_root:
    root = Path(tmp_root)
    corpus_file = root / "corpus.txt"
    corpus_file.write_text("hello world tokenization is interesting नमः", encoding="utf-8")
    
    config = bpe_multilingual(vocab_size=100)
    # Ensure our tiny corpus words aren't filtered out
    config = config.model_copy(update={"trainer": config.trainer.model_copy(update={"min_frequency": 1})})
    tok = Tokenizer.from_config(config)
    tok.train([str(corpus_file)], config)

    # Test 1: Simple Success
    t1 = "tokenization"
    res1, m1 = run_round_trip(tok, t1)
    print(f"[A] SUCCESS CASE: '{t1}'")
    print(f"    Decoded: '{res1}'")
    print(f"    Match? {'PASSED' if m1 else 'FAILED'}")

    # Test 2: Space Loss Bug (Redefined: Multiple spaces)
    t2 = "hello  world" # Double space
    res2, m2 = run_round_trip(tok, t2)
    print(f"\n[B] DOUBLE SPACE LOSS: '{t2}'")
    print(f"    Decoded: '{res2}'")
    print(f"    Match? {'PASSED' if m2 else 'FAILED'}")

    # Test 4: Leading/Trailing Space
    t4 = " hello "
    res4, m4 = run_round_trip(tok, t4)
    print(f"\n[D] LEADING/TRAILING SPACE LOSS: '{t4}'")
    print(f"    Decoded: '{res4}'")
    print(f"    Match? {'PASSED' if m4 else 'FAILED'}")

    # Test 3: Unicode Normalization (NFD -> NFC)
    # नमः in NFD form
    t3 = unicodedata.normalize('NFD', "नमः")
    res3, m3 = run_round_trip(tok, t3)
    print(f"\n[C] UNICODE NFD: '{t3}' (len {len(t3)})")
    print(f"    Decoded: '{res3}' (len {len(res3)})")
    print(f"    Match? {'PASSED' if m3 else 'FAILED'}")
    
    # Check if decoded is actually NFC
    is_nfc = (res3 == unicodedata.normalize('NFC', "नमः"))
    print(f"    Is Decoded NFC? {is_nfc}")
