import json
from pydantic import ValidationError

from abctokz.config.schemas import (
    TokenizerConfig,
    BPEConfig,
    BPETrainerConfig,
    UnigramTrainerConfig,
    WordLevelConfig
)
from abctokz.tokenizer import AugenblickTokenizer

print("\n--- TEST 1: Mismatched Model & Trainer ---")
try:
    # Attempting to pair a BPE model with a Unigram trainer
    config = TokenizerConfig(
        model=BPEConfig(),
        trainer=UnigramTrainerConfig()
    )
except ValidationError as e:
    print("Caught validation error:")
    print(e)


print("\n--- TEST 2: Invalid Vocab Size ---")
try:
    # Attempting to set a negative vocab_size
    model = BPEConfig(vocab_size=-5)
except ValidationError as e:
    print("Caught validation error:")
    print(e)


print("\n--- TEST 3: Extra Unknown Fields ---")
try:
    # Attempting to pass a non-existent parameter
    config = TokenizerConfig(
        model=BPEConfig(),
        foobar="baz"
    )
except ValidationError as e:
    print("Caught validation error:")
    print(e)

print("\n--- TRACE: Valid Config to Tokenizer ---")
# Build a valid config
valid_config = TokenizerConfig(
    model=BPEConfig(),
    trainer=BPETrainerConfig()
)
# Trace where the defaults came from
print("Vocab size injected by default:", valid_config.model.vocab_size)
print("Continuation prefix injected by default:", valid_config.model.continuation_prefix)

# Pass to tokenizer
tokenizer = AugenblickTokenizer.from_config(valid_config)
print("Tokenizer initialized successfully:")
print(repr(tokenizer))

