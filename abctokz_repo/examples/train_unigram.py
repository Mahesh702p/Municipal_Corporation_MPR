"""Example: train a Unigram tokenizer on multilingual text.

Run::

    python examples/train_unigram.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import unigram_multilingual

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
    # Sindhi in Devanagari
    "सिन्धी भाषा",
    "सिन्ध प्रान्त",
    # Mixed
    "hello नमस्ते world दुनिया",
    "Devanagari script नागरी लिपि",
] * 25


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        config = unigram_multilingual(vocab_size=300)

        tokenizer = Tokenizer.from_config(config)
        print("Training Unigram tokenizer...")
        tokenizer.train([str(corpus_path)], config)
        print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")

        artifact_dir = Path(tmp) / "unigram_tok"
        tokenizer.save(str(artifact_dir))
        print(f"  Saved to: {artifact_dir}")

        loaded = Tokenizer.load(str(artifact_dir))

        examples = [
            "hello world",
            "नमस्ते दुनिया",
            "hello नमस्ते world",
            "मराठी भाषेत",
            "सिन्धी भाषा",
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
