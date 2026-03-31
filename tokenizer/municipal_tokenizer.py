"""
municipal_tokenizer.py
======================
MunicipalTokenizer — a wrapper around the abctokz library that:

  ✅ Fixes Gap 1: Adds <PAD> token (id=0) automatically
  ✅ Fixes Gap 2: Provides pad_sequences() built-in (TF-ready output)
  ✅ Fixes Gap 3: Hinglish normalization (common word standardization)
  ✅ Fixes Gap 4: Provides vocab_size property for Embedding layer
  ✅ Fixes Gap 5: Adds municipal domain special tokens (<DEPT>, <WARD>, etc.)

Usage:
  from tokenizer.municipal_tokenizer import MunicipalTokenizer

  # Train
  tok = MunicipalTokenizer(vocab_size=10000)
  tok.train(["data/processed/pretrain_corpus.txt"])
  tok.save("artifacts/municipal_bpe_tok")

  # Load and use
  tok = MunicipalTokenizer.load("artifacts/municipal_bpe_tok")
  ids = tok.encode("garbage not collected in ward 5")
  padded = tok.encode_batch(texts, max_len=60)  # → numpy array ready for TF
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add abctokz to path
_REPO_PATH = Path(__file__).parent.parent / "abctokz_repo" / "src"
if str(_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# Hinglish Normalization Dictionary
# Common variant spellings → canonical form so tokenizer learns one form
# ─────────────────────────────────────────────────────────────────────────────
HINGLISH_NORM = {
    # Water
    "pani": "paani", "paanee": "paani", "paany": "paani",
    "p ani": "paani", "jal": "paani",
    # Road
    "sadak": "sadak", "rok": "road", "sarak": "sadak",
    # Garbage
    "kachara": "kachra", "kachhra": "kachra", "kubda": "kachra",
    "waste": "kachra", "kooda": "kachra", "kuda": "kachra",
    # Not
    "nhi": "nahi", "nahin": "nahi", "nhin": "nahi", "ni": "nahi",
    # Electricity
    "current": "bijli", "light": "bijli", "electricity": "bijli",
    "bijlee": "bijli",
    # Municipal
    "mc": "municipal_corporation", "nagar_nigam": "municipal_corporation",
    "bmc": "municipal_corporation", "bbmp": "municipal_corporation",
    "mcgm": "municipal_corporation", "pmc": "municipal_corporation",
    # Common complaint phrases
    "nahi aata": "nahi_aa_raha", "nahi aati": "nahi_aa_rahi",
    "band hai": "band_hai", "kab aayega": "kab_aayega",
    "please": "please", "kindly": "please",
}

# Compile pattern: match whole words
_HINGLISH_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in HINGLISH_NORM.keys()) + r")\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Special tokens for municipal domain
# ─────────────────────────────────────────────────────────────────────────────
MUNICIPAL_SPECIAL_TOKENS = [
    "<PAD>",    # Padding (id=0 — always first)
    "<UNK>",    # Unknown
    "<BOS>",    # Beginning of sequence
    "<EOS>",    # End of sequence
    "<MASK>",   # For masked LM training
    "<DEPT>",   # Department placeholder
    "<WARD>",   # Ward reference
    "<ZONE>",   # Zone reference
    "<PHONE>",  # PII-masked phone
    "<EMAIL>",  # PII-masked email
    "<URL>",    # Masked URL
    "<AADHAAR>",
    "<PINCODE>",
]


class MunicipalTokenizer:
    """
    Production tokenizer for Municipal Corporation MLM project.

    Wraps abctokz with domain-specific extensions:
      - Hinglish normalization
      - PAD token at id=0
      - TF-ready encode_batch with padding/truncation
      - Municipal special tokens

    Args:
        vocab_size: Target vocabulary size (default 10000).
        model_type: "bpe" (default), "unigram", or "wordlevel".
        max_len: Default max sequence length for padding.
    """

    PAD_ID = 0
    PAD_TOKEN = "<PAD>"

    def __init__(
        self,
        vocab_size: int = 10000,
        model_type: str = "bpe",
        max_len: int = 60,
    ):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.max_len = max_len
        self._tokenizer = None  # set after train() or load()

    # ──────────────────────────────────────
    # Text preprocessing
    # ──────────────────────────────────────
    def _normalize_hinglish(self, text: str) -> str:
        """Standardize Hinglish variant spellings to canonical form."""
        def replace(match):
            word = match.group(0).lower()
            return HINGLISH_NORM.get(word, word)
        return _HINGLISH_PATTERN.sub(replace, text)

    def _preprocess(self, text: str) -> str:
        """Apply all preprocessing before tokenization."""
        text = text.strip()
        text = self._normalize_hinglish(text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ──────────────────────────────────────
    # Training
    # ──────────────────────────────────────
    def train(self, corpus_paths: list[str]):
        """Train the tokenizer on corpus files."""
        from abctokz import Tokenizer
        from abctokz.config.defaults import bpe_multilingual, unigram_multilingual, wordlevel_multilingual

        # Build config based on model type
        if self.model_type == "bpe":
            config = bpe_multilingual(vocab_size=self.vocab_size)
        elif self.model_type == "unigram":
            config = unigram_multilingual(vocab_size=self.vocab_size)
        else:
            config = wordlevel_multilingual(vocab_size=self.vocab_size)

        # Preprocess corpus into a temp file
        tmp_paths = []
        for path in corpus_paths:
            tmp_path = path + ".preprocessed.tmp"
            with open(path, encoding="utf-8") as f_in, \
                 open(tmp_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    processed = self._preprocess(line.strip())
                    if processed:
                        f_out.write(processed + "\n")
            tmp_paths.append(tmp_path)

        print(f"[MunicipalTokenizer] Training {self.model_type} tokenizer "
              f"(vocab_size={self.vocab_size}) on {len(corpus_paths)} corpus files...")

        self._tokenizer = Tokenizer.from_config(config)
        self._tokenizer.train(tmp_paths, config)

        # Clean up temp files
        for tmp in tmp_paths:
            os.remove(tmp)

        print(f"[MunicipalTokenizer] Training complete. Vocab size: {self.get_vocab_size()}")

    # ──────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────
    def encode(self, text: str) -> list[int]:
        """
        Encode a single text string → list of token IDs.
        IDs are offset by len(MUNICIPAL_SPECIAL_TOKENS) to reserve
        space for special tokens at the start of the vocabulary.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained. Call train() or load() first.")

        text = self._preprocess(text)
        enc = self._tokenizer.encode(text)
        # Offset IDs to leave room for special tokens
        offset = len(MUNICIPAL_SPECIAL_TOKENS)
        return [i + offset for i in enc.ids]

    def encode_batch(
        self,
        texts: list[str],
        max_len: Optional[int] = None,
        padding: str = "post",
        truncating: str = "post",
    ) -> np.ndarray:
        """
        Encode a batch of texts → padded numpy array (ready for TF Embedding).

        Args:
            texts: List of raw text strings.
            max_len: Max sequence length (defaults to self.max_len).
            padding: "pre" or "post" (default "post").
            truncating: "pre" or "post" (default "post").

        Returns:
            np.ndarray of shape (len(texts), max_len) with int32 dtype.
        """
        if max_len is None:
            max_len = self.max_len

        sequences = [self.encode(t) for t in texts]

        # Pad / truncate
        padded = np.zeros((len(sequences), max_len), dtype=np.int32)
        for i, seq in enumerate(sequences):
            # Truncate
            if truncating == "post":
                seq = seq[:max_len]
            else:
                seq = seq[-max_len:]

            # Pad
            seq_len = len(seq)
            if padding == "post":
                padded[i, :seq_len] = seq
            else:
                padded[i, max_len - seq_len:] = seq

        return padded  # shape: (batch, max_len) — drop directly into model.fit()

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        offset = len(MUNICIPAL_SPECIAL_TOKENS)
        adjusted = [max(0, i - offset) for i in ids if i >= offset]
        return self._tokenizer.decode(adjusted)

    # ──────────────────────────────────────
    # Vocabulary
    # ──────────────────────────────────────
    def get_vocab_size(self) -> int:
        """Total vocab size including special tokens (use for Embedding layer)."""
        if self._tokenizer is None:
            return len(MUNICIPAL_SPECIAL_TOKENS)
        return self._tokenizer.get_vocab_size() + len(MUNICIPAL_SPECIAL_TOKENS)

    def get_special_token_id(self, token: str) -> int:
        """Get ID of a special token (e.g., '<PAD>', '<MASK>')."""
        if token in MUNICIPAL_SPECIAL_TOKENS:
            return MUNICIPAL_SPECIAL_TOKENS.index(token)
        raise KeyError(f"Unknown special token: {token}")

    def get_vocab(self) -> dict[str, int]:
        """Return full vocabulary including special tokens."""
        vocab = {tok: i for i, tok in enumerate(MUNICIPAL_SPECIAL_TOKENS)}
        offset = len(MUNICIPAL_SPECIAL_TOKENS)
        if self._tokenizer:
            for token, tok_id in self._tokenizer.get_vocab().items():
                vocab[token] = tok_id + offset
        return vocab

    # ──────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────
    def save(self, path: str):
        """Save tokenizer to directory."""
        os.makedirs(path, exist_ok=True)
        if self._tokenizer is None:
            raise RuntimeError("Cannot save untrained tokenizer.")
        self._tokenizer.save(path)

        # Save our metadata
        import json
        meta = {
            "vocab_size": self.vocab_size,
            "model_type": self.model_type,
            "max_len": self.max_len,
            "special_tokens": MUNICIPAL_SPECIAL_TOKENS,
        }
        with open(os.path.join(path, "municipal_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[MunicipalTokenizer] Saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "MunicipalTokenizer":
        """Load tokenizer from directory."""
        import json
        from abctokz import Tokenizer

        meta_path = os.path.join(path, "municipal_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {"vocab_size": 10000, "model_type": "bpe", "max_len": 60}

        instance = cls(
            vocab_size=meta["vocab_size"],
            model_type=meta["model_type"],
            max_len=meta["max_len"],
        )
        instance._tokenizer = Tokenizer.load(path)
        print(f"[MunicipalTokenizer] Loaded from: {path} "
              f"(vocab_size={instance.get_vocab_size()})")
        return instance

    def __repr__(self) -> str:
        vs = self.get_vocab_size() if self._tokenizer else "untrained"
        return f"MunicipalTokenizer(model={self.model_type}, vocab_size={vs}, max_len={self.max_len})"
