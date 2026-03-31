import sys
from pathlib import Path

# Add src to sys.path so we can import abctokz regardless of where this is run from
current_dir = Path(__file__).parent.resolve()
SRC_ROOT = (current_dir.parent / "src").resolve()
sys.path.append(str(SRC_ROOT))

from abctokz import AugenblickTokenizer
from abctokz.models.wordlevel import WordLevelModel
from abctokz.vocab.vocab import Vocabulary
from abctokz.pretokenizers.whitespace import WhitespacePreTokenizer

def evaluate_task_12():
    """
    Demonstrates Task 12: Where the Design Lies to You.
    
    The architecture claims special tokens are managed via a registry.
    The implementation hardcodes a filter for tokens matching `<*>` in decode().
    """
    print("--- Task 12: Architectural Mismatch Evaluation ---\n")

    # 1. Setup a vocabulary where an HTML-like tag is a NORMAL token, 
    # not registered in special_tokens.
    raw_vocab = {"<unk>": 0, "hello": 1, "<br>": 2, "world": 3}
    vocab = Vocabulary(raw_vocab)
    model = WordLevelModel(vocab=vocab)
    pretokenizer = WhitespacePreTokenizer()
    
    # We leave special_tokens empty. 
    # The architecture promises ONLY tokens in this dict will be skipped if skip_special_tokens=True.
    tokenizer = AugenblickTokenizer(model=model, pretokenizer=pretokenizer, special_tokens={})

    text = "hello <br> world"
    print(f"Input text: '{text}'")

    # 2. Encoding
    encoding = tokenizer.encode(text)
    print(f"Encoded tokens: {encoding.tokens}")
    print(f"Encoded IDs:    {encoding.ids}")
    
    # Verify <br> is considered a normal token (special_tokens_mask should be 0)
    print(f"Special mask:   {encoding.special_tokens_mask}")
    
    # 3. Decoding (the failure point)
    # By default skip_special_tokens is True.
    decoded = tokenizer.decode(encoding.ids, skip_special_tokens=True)
    
    print(f"\nDecoded text (skip_special_tokens=True): '{decoded}'")
    
    # 4. Assessment
    expected = "hello <br> world"
    if decoded == expected:
        print("\nRESULT: SUCCESS (No mismatch found)")
    else:
        print("\nRESULT: FAILURE (Architectural mismatch confirmed)")
        print(f"EXPECTED: '{expected}'")
        print(f"ACTUAL:   '{decoded}'")
        print("\nREASON: The code in tokenizer.py:decode() silently dropped '<br>' because it matches the pattern <*>,")
        print("even though it is NOT a registered special token. This violates the 'clean modular abstraction' promise.")

if __name__ == "__main__":
    evaluate_task_12()
