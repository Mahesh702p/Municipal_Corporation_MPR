"""Example: train a WordLevel tokenizer on multilingual text.

Run::

    python examples/train_wordlevel.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import wordlevel_multilingual

# ---------------------------------------------------------------------------
# Sample corpus (in a real scenario, point to actual files)
# ---------------------------------------------------------------------------

CORPUS_LINES = [
    # English
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "tokenization is important for natural language processing",
    "machine learning models need good tokenizers",
    # Hindi
    "नमस्ते दुनिया",
    "यह एक परीक्षण वाक्य है",
    "हिन्दी भाषा में टोकनाइजेशन",
    "भारत एक विशाल देश है",
    # Marathi
    "नमस्कार जग",
    "मराठी भाषेत टोकनायझेशन",
    "हे एक चाचणी वाक्य आहे",
    # Mixed
    "hello नमस्ते world दुनिया",
    "BPE tokenizer for Hindi हिन्दी",
] * 20  # repeat to get enough frequency


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Write corpus
        corpus_path = Path(tmp) / "corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        # Build config
        config = wordlevel_multilingual(vocab_size=200)

        # Create and train tokenizer
        tokenizer = Tokenizer.from_config(config)
        print("Training WordLevel tokenizer...")
        tokenizer.train([str(corpus_path)], config)
        print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")

        # Save
        artifact_dir = Path(tmp) / "wordlevel_tok"
        tokenizer.save(str(artifact_dir))
        print(f"  Saved to: {artifact_dir}")

        # Load
        loaded = Tokenizer.load(str(artifact_dir))
        print(f"  Loaded: {loaded}")

        # Encode / decode examples
        examples = [
            "hello world",
            "नमस्ते दुनिया",
            "hello नमस्ते world दुनिया",
            "मराठी भाषेत",
        ]
        for text in examples:
            enc = loaded.encode(text)
            decoded = loaded.decode(enc.ids)
            print(f"\n  Input:   {text!r}")
            print(f"  Tokens:  {enc.tokens}")
            print(f"  IDs:     {enc.ids}")
            print(f"  Decoded: {decoded!r}")


if __name__ == "__main__":
    main()
