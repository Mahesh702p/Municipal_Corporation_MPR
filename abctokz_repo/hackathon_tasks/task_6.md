# Task 6 — Making the Tokenizer Say "I Don't Know"

> **Tokens Used for Task 6:** ~51,000 (Research, Experiments & Pipeline Auditing)

## Where `<unk>` Comes From

In the `abctokz` pipeline, the fallback to the **Unknown Token (`<unk>`)** happens at the **Model** stage. 

The **Normalizer** and **PreTokenizer** process strings without needing to know if they exist in the vocabulary. However, when a chunk of text reaches the Model:
- **WordLevel**: Performs a direct dictionary lookup. If the key is missing, it returns the `unk_id`.
- **BPE**: Recursively breaks words into characters. If a specific character (or its prefixed version like `##x`) was never seen during training, it cannot be mapped to an ID, resulting in `<unk>`.
- **Unigram**: Uses the Viterbi algorithm. If no combination of known pieces can form a character, it applies a heavy penalty and emits `<unk>`.

## Case 1 — Rare Word (Vocabulary Gap)
**Input**: `antidisestablishmentarianism`
- **WordLevel**: `['<unk>']`. Because the model treats the entire string as a single unit, a single missing word results in total information loss.
- **BPE**: `['a', '##n', '##t', ... '##m']`. BPE excels here. It broke the rare word into common subword pieces. It only produces `<unk>` if a specific *character* (like `b` in our experiment) was missing from the training alphabet.

## Case 2 — Script Mismatch
**Input**: `भारत महान है` (Trained on English-only data)
- **Result**: `['<unk>', '<unk>', ...]` (All models).
- **Cause**: The training corpus didn't contain Devanagari characters. Thus, the "Initial Alphabet" for BPE and Unigram does not include these Unicode ranges. When the model encounters a character outside its known alphabet, it has no ID to map it to.

## Case 3 — Emoji / Symbols
**Input**: `🚀🔥✨`
- **Result**: `['<unk>']`
- **Cause**: Emojis are high-byte Unicode characters. Unless explicitly included in the training data or manually added to the vocabulary, the model sees them as "noise" outside its supported character set.

## Model Comparison

| Model | Robustness | UNK Frequency |
| :--- | :--- | :--- |
| **WordLevel** | **Low** | Very High. Fails on any word not seen in training. |
| **BPE** | **High** | Low. Only fails if it encounters a totally new *character*. |
| **Unigram** | **Medium** | Moderate. Can be aggressive with `<unk>` if the character coverage during training was low. |

## Most Fragile Model
**WordLevel** is the most fragile. It lacks any "compositional" understanding of language. It treats `play` and `playing` as distinct, uncorrelated atoms. This makes it unusable for real-world multilingual tasks where vocabulary growth is exponential.

## Suggestion to Reduce UNK Rate
**Implementation of Byte-Level BPE (BBPE)**:
Instead of training on Unicode characters, the tokenizer should operate on **UTF-8 bytes**. Since there are only **256 possible byte values**, every possible character (including emojis and rare scripts) can be represented as a sequence of known "byte-base" tokens. This ensures a **0% UNK rate** across all inputs, regardless of whether the specific script was seen during training.
