import sys
from pathlib import Path

# Add src to sys.path
current_dir = Path(__file__).parent.resolve()
SRC_ROOT = (current_dir.parent / "src").resolve()
sys.path.append(str(SRC_ROOT))

from abctokz import AugenblickTokenizer
from abctokz.models.bpe import BPEModel
from abctokz.vocab.serialization import load_vocab, load_merges
from abctokz.pretokenizers.whitespace import WhitespacePreTokenizer

def test_offset_bug():
    print("--- Testing Offset Calculation Bug ---\n")
    
    # Simple manual setup for BPE that splits "hello"
    # Assuming vocab has 'h', '##e', '##l', '##o'
    vocab = {"<unk>": 0, "h": 1, "##e": 2, "##l": 3, "##o": 4, " ": 5, "world": 6}
    # No merges needed for this manual split demonstration if we mock the model
    # Actually let's just use what's there.
    
    from abctokz.models.base import Model
    class MockBPE(Model):
        def tokenize(self, seq):
            if seq == "hello":
                return [("h", 1), ("##e", 2), ("##l", 3), ("##l", 3), ("##o", 4)]
            return [(seq, 6)]
        def get_vocab(self): return vocab
        def get_vocab_size(self): return len(vocab)
        def save(self, d): pass
        @classmethod
        def load(cls, d): return cls()

    tokenizer = AugenblickTokenizer(
        model=MockBPE(),
        pretokenizer=WhitespacePreTokenizer()
    )
    
    text = "hello world"
    encoding = tokenizer.encode(text)
    
    print(f"Text: '{text}'")
    print(f"Tokens:  {encoding.tokens}")
    print(f"Offsets: {encoding.offsets}")
    
    # The first 5 tokens correspond to 'hello'
    # Expected: (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)
    # Actual (Buggy): (0, 5), (0, 5), (0, 5), (0, 5), (0, 5)
    
    hello_offsets = encoding.offsets[:5]
    if all(o == (0, 5) for o in hello_offsets):
        print("\nRESULT: BUG CONFIRMED (All sub-tokens have the full word offset)")
    else:
        print("\nRESULT: BUG NOT FOUND or already fixed")

if __name__ == "__main__":
    test_offset_bug()
