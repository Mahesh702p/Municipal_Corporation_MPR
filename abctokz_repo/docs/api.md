# API Reference

## `abctokz.Tokenizer`

```python
class Tokenizer:
    def encode(self, text: str) -> Encoding: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...
    def encode_batch(self, texts: list[str]) -> list[Encoding]: ...
    def get_vocab(self) -> dict[str, int]: ...
    def get_vocab_size(self) -> int: ...
    def token_to_id(self, token: str) -> int | None: ...
    def id_to_token(self, token_id: int) -> str | None: ...
    def train(self, corpus_paths: list[str], config: TokenizerConfig) -> None: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def from_config(cls, config: TokenizerConfig) -> "Tokenizer": ...
    @classmethod
    def load(cls, path: str) -> "Tokenizer": ...
```

## `abctokz.types.Encoding`

```python
@dataclass(frozen=True)
class Encoding:
    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]
    special_tokens_mask: list[int]
    attention_mask: list[int]
    overflowing: list["Encoding"]
```

## Config Presets

```python
from abctokz.config.defaults import (
    wordlevel_multilingual,
    bpe_multilingual,
    unigram_multilingual,
    english_basic_normalizer,
    devanagari_safe_normalizer,
    multilingual_shared_normalizer,
)
```

## Normalizers

- `IdentityNormalizer()` — pass-through
- `NfkcNormalizer(strip_zero_width=True)` — NFKC + optional ZW strip
- `WhitespaceNormalizer(strip=True, collapse=True)` — whitespace cleanup
- `DevanagariNormalizer(nfc_first=True, strip_zero_width=False)` — Devanagari-safe NFC
- `SequenceNormalizer([...])` — chain multiple normalizers

## Pre-tokenizers

- `WhitespacePreTokenizer()` — split on whitespace
- `PunctuationPreTokenizer(behavior="isolated")` — split on punctuation
- `RegexPreTokenizer(pattern, invert=False)` — custom regex
- `DevanagariAwarePreTokenizer(split_on_whitespace=True, split_on_script_boundary=True)`
- `SequencePreTokenizer([...])` — chain multiple pre-tokenizers
