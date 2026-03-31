"""Example: train a BPE tokenizer on multilingual text.

Run::

    python examples/train_bpe.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual

CORPUS_LINES = [
    # English
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "tokenization is important for natural language processing",
    "machine learning models need good tokenizers",
    "subword segmentation helps with rare words",
    # Hindi
    "नमस्ते दुनिया",
    "यह एक परीक्षण वाक्य है",
    "हिन्दी भाषा में टोकनाइजेशन",
    "भारत एक विशाल देश है",
    "मशीन लर्निंग मॉडल के लिए टोकनाइज़र",
    # Marathi
    "नमस्कार जग",
    "मराठी भाषेत टोकनायझेशन",
    "हे एक चाचणी वाक्य आहे",
    # Mixed
    "hello नमस्ते world दुनिया",
    "BPE tokenizer for Hindi हिन्दी",
    "Devanagari script नागरी लिपि",
] * 30


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        config = bpe_multilingual(vocab_size=500)

        tokenizer = Tokenizer.from_config(config)
        print("Training BPE tokenizer...")
        tokenizer.train([str(corpus_path)], config)
        print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")

        artifact_dir = Path(tmp) / "bpe_tok"
        tokenizer.save(str(artifact_dir))
        print(f"  Saved to: {artifact_dir}")

        loaded = Tokenizer.load(str(artifact_dir))

        examples = [
            "hello world",
            "नमस्ते दुनिया",
            "hello नमस्ते world दुनिया",
            "tokenization",
            "मराठी भाषेत",
            "subword segmentation",
        ]
        for text in examples:
            enc = loaded.encode(text)
            decoded = loaded.decode(enc.ids)
            print(f"\n  Input:   {text!r}")
            print(f"  Tokens:  {enc.tokens}")
            print(f"  IDs:     {enc.ids}")
            print(f"  Decoded: {decoded!r}")
            print(f"  Length:  {len(enc.ids)} tokens")


if __name__ == "__main__":
    main()
