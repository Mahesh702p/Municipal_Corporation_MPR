### 1. Intended Design

The tokenizer architecture clearly promises a clean modular abstraction where special tokens are handled through a dedicated registry (`self._special_tokens`), and the orchestrator layer (the `AugenblickTokenizer` class) limits itself purely to pipeline coordination. Under this paradigm, tokens and out-of-vocabulary elements map strictly to definitions managed by the `SpecialToken` instances in the registry, and no extra token filtering logic should occur beyond the explicit registry parameters. 

### 2. Actual Implementation

The architecture’s promise breaks down in `src/abctokz/tokenizer.py` inside the `decode()` method. When the parameter `skip_special_tokens` is True (its default value), the code not only skips tokens found in `self._special_tokens`, but it also features a hardcoded catch-all checking rule: `or (t.startswith("<") and t.endswith(">"))`. 

Because of this specific line of code:
```python
tokens = [
    t for t in tokens
    if t and not (t in special_strs or (t.startswith("<") and t.endswith(">")))
]
```
The tokenizer strips out any token wrapped in angle brackets, completely disregarding whether it was explicitly configured in the special tokens registry. 

### 3. Demonstration

This architectural leak becomes evident when tokenizing valid language tokens that share this angle-bracket syntax, such as XML/HTML tags or emoticons like `<3>`.

```python
from abctokz import AugenblickTokenizer
from abctokz.models.wordlevel import WordLevelModel

# Setting up a model vocabulary where an HTML tag is a valid, normal token (not a special token)
model = WordLevelModel(vocab={"hello": 0, "<br>": 1, "world": 2})
tokenizer = AugenblickTokenizer(model=model)

# The tokenizer will correctly encode the sequence
encoding = tokenizer.encode("hello <br> world")
print(encoding.tokens)  # Output: ['hello', '<br>', 'world']

# But during decoding, the structural violation drops the valid token
decoded_text = tokenizer.decode(encoding.ids)
print(decoded_text)  
# Expected output: "hello <br> world"
# Actual output: "helloworld" (the "<br>" is silently removed!)
```

### 4. Severity

This represents a **severe design flaw** and a functional bug. It causes catastrophic, undetected data loss when decoding texts from crucial domains like HTML/XML parsing, casual social media messaging (emoticons), or code repositories. The pipeline silently drops legitimate tokens, fundamentally compromising the integrity of detokenized outputs while breaking the single-source-of-truth guarantee regarding special tokens. 

### 5. Minimal Fix

The fix requires reverting to the explicit `_special_tokens` registry for filtering and removing the ad-hoc pattern matching logic from the orchestrator.

```python
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to a string."""
        vocab = self._model.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        tokens = [inv_vocab.get(i, "") for i in ids]
        
        if skip_special_tokens:
            special_strs = set(self._special_tokens.keys())
            # Only skip tokens that are explicitly listed in the registry
            tokens = [
                t for t in tokens
                if t and t not in special_strs
            ]
        return self._decoder.decode(tokens)
```
