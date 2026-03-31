import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual

def get_file_hash(path: Path):
    """Compute SHA-256 hash of a file."""
    if not path.exists():
        return "MISSING"
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Setup corpus
corpus = [
    "Determinism is the philosophical view that all events, including moral choices.",
    "नियतिवाद वह दार्शनिक विचार है कि नैतिक विकल्पों सहित सभी घटनाएँ पहले से तय होती हैं।",
    "निश्चिततावाद हा असा तात्विक दृष्टिकोन आहे की सर्व घटना मानवी कृतींसह पूर्व-नियोजित असतात."
] * 50

print("--- EXPERIMENT: DETERMINISM VERIFICATION ---\n")

with tempfile.TemporaryDirectory() as tmp_root:
    root = Path(tmp_root)
    corpus_file = root / "corpus.txt"
    corpus_file.write_text("\n".join(corpus), encoding="utf-8")
    
    # Run 1
    dir1 = root / "model1"
    base_config = bpe_multilingual(vocab_size=200)
    # Correctly Reconstruct with seed since they are frozen
    config1 = base_config.model_copy(update={"trainer": base_config.trainer.model_copy(update={"seed": 42})})
    tok1 = Tokenizer.from_config(config1)
    tok1.train([str(corpus_file)], config1)
    tok1.save(str(dir1))
    
    hash_v1 = get_file_hash(dir1 / "vocab.json")
    hash_m1 = get_file_hash(dir1 / "merges.txt")
    
    # Run 2 (Identical Everything)
    dir2 = root / "model2"
    config2 = base_config.model_copy(update={"trainer": base_config.trainer.model_copy(update={"seed": 42})})
    tok2 = Tokenizer.from_config(config2)
    tok2.train([str(corpus_file)], config2)
    tok2.save(str(dir2))
    
    hash_v2 = get_file_hash(dir2 / "vocab.json")
    hash_m2 = get_file_hash(dir2 / "merges.txt")
    
    # Run 3 (Different Seed)
    dir3 = root / "model3"
    config3 = base_config.model_copy(update={"trainer": base_config.trainer.model_copy(update={"seed": 999})})
    tok3 = Tokenizer.from_config(config3)
    tok3.train([str(corpus_file)], config3)
    tok3.save(str(dir3))
    
    hash_v3 = get_file_hash(dir3 / "vocab.json")
    hash_m3 = get_file_hash(dir3 / "merges.txt")

    print(f"[A] TRAINING DETERMINISM:")
    print(f"    Run 1 (Seed 42) Vocab Hash: ...{hash_v1[-8:]}")
    print(f"    Run 2 (Seed 42) Vocab Hash: ...{hash_v2[-8:]}")
    print(f"    Run 3 (Seed 999) Vocab Hash: ...{hash_v3[-8:]}")
    print(f"    Match (1 vs 2)? {'PASSED' if hash_v1 == hash_v2 else 'FAILED'}")
    print(f"    Difference (1 vs 3)? {'PASSED' if hash_v1 != hash_v3 else 'FAILED'}")

    # Encoding Loop
    sentence = "Verify this claim experimentally. Don't just take the documentation's word for it."
    ids_seen = set()
    for _ in range(100):
        enc = tok1.encode(sentence)
        ids_seen.add(tuple(enc.ids))
    
    print(f"\n[B] ENCODING CONSISTENCY:")
    print(f"    Iterations: 100")
    print(f"    Unique ID sequences found: {len(ids_seen)}")
    print(f"    Consistency Result: {'PASSED' if len(ids_seen) == 1 else 'FAILED'}")

    # Benchmark check (pseudo-check)
    print(f"\n[C] SPEED VARIABILITY (Manual Observation Required)")
    import time
    start = time.time()
    for _ in range(1000): tok1.encode(sentence)
    time1 = time.time() - start
    
    start = time.time()
    for _ in range(1000): tok1.encode(sentence)
    time2 = time.time() - start
    
    print(f"    Time 1: {time1:.4f}s")
    print(f"    Time 2: {time2:.4f}s")
    print(f"    Time Match? {'YES (Rare)' if time1 == time2 else 'NO (Expected)'}")
