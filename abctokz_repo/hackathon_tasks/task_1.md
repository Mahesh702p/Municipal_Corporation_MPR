# Task 1 — Follow the Code: What Actually Happens When You Tokenize?

> **Tokens Used for Task 1:** 4,800 (Finalized with OOV Ablation Experiment)

## 1. The Investigation Setup
To trace the pipeline, we wrote a dedicated script (`hackathon_tasks/trace_mantra.py`) that trained a **Multilingual BPE Tokenizer** with a tiny vocabulary (size 300) on English, Hindi, and Marathi sentences. 

We then passed the following Sanskrit mantra into the pipeline to watch its internal state at each of the 4 stages:
`ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥`

---

## 2. Stage-by-Stage Pipeline Trace

### Stage 1: The Normalizer
- **Involved Files/Classes:** `abctokz/normalizers/devanagari.py` (Class: `DevanagariNormalizer`), invoked by `tokenizer.py:100`.
- **Input String:** `'ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं ...'`
- **Output String:** `'ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं ...'`
- **What happened & Why:** The `DevanagariNormalizer` ran the string through `unicodedata.normalize('NFC', text)`. In our test, the text was already in NFC form, so it did not visibly change (`mantra != normalized` returned `False`). NFC ensures that decomposed sequences (e.g., base letter + matra) are combined into single Unicode code points wherever possible, preventing fragmentation later.

### Stage 2: The Pre-Tokenizer
- **Involved Files/Classes:** `abctokz/pretokenizers/devanagari_aware.py` (Class: `DevanagariAwarePreTokenizer`), invoked by `tokenizer.py:106`.
- **Input String:** `'ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं ...'`
- **Output List of Pre-Tokens:** `['ॐ', 'भूर्भुवः', 'स्व:', 'तत्सवितुर्वरेण्यं', 'भर्गो' ... ]`
- **What happened & Why:** The pre-tokenizer split the string at every space boundary. This forces the BPE model down the line to never merge characters across different words. The Devanagari pretokenizer ensures it doesn't split inside complex consonant conjuncts. 

### Stage 3: The Model (BPE tokenize)
- **Involved Files/Classes:** `abctokz/models/bpe.py` (Class: `BPE`), invoked by `tokenizer.py:117` routing through the model.
- **Input List of Pre-Tokens:** `['ॐ', 'भूर्भुवः', 'स्व:', ...]`
- **Output Token IDs & Subword Pieces:** 
  - `'ॐ'` -> `['ॐ']` (ID: 210)
  - `'भूर्भुवः'` -> `['भ', '##ू', '##र्', '##भु', '##व', '##ः']` (IDs: 193, 144, 122, 106, 123, 70)
  - `'स्व:'` -> `['स', '##्व', '##:']` (IDs: 204, 151, 1)
- **What happened & Why:** Because our vocab size was tiny (300), the BPE model lacked the full word `'भूर्भुवः'`. It executed a greedy lookup from its `vocab.json` merges. The base consonant `भ` was kept separate from its matra/modifiers (`##ू`, `##र्`) because they were not frequent enough in the corpus to merge. Notice the `"##"` prefix indicating a subword continuation.

### Stage 4: The Decoder
- **Involved Files/Classes:** `abctokz/decoders/subword_decoder.py` (Class: `SubwordDecoder`), invoked by `tokenizer.py:180` routing through `decode()`.
- **Input Token IDs:** `[210, 193, 144, 122, 106, 123, 70, 204, 151, 1 ...]`
- **Output Reconstructed String:** `'ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥'`
- **What happened & Why:** The `SubwordDecoder` converted the IDs back to strings. When it saw the `##` prefix (e.g., `##ू`), it stripped the prefix and attached the character directly to the previous character (`भ` + `ू` = `भू`). When a token lacked the prefix (e.g., `'स'`), it inserted a space before it. 

---

## 3. Key Insights & Edge Cases Revealed
**Insight 1: Hard-coded ASCII bug.** 
The word `स्व:` contains an ASCII colon (`:`), not a Devanagari Visarga (`ः`). You can see the model split this as `['स', '##्व', '##:']`. The model sees the ASCII colon as a continuation because of its BPE split rules, whereas a real Hindi tokenization pipeline should probably normalize standard colons into Devanagari visargas if they appear adjacent to Devanagari text.

**Insight 2: The Space Loss Bug (Crucial).**
Though our mantra decoded correctly, our trace script revealed a critical flaw in `SubwordDecoder` when handling regular English words like "hello world". Because "hello" and "world" fell back to character tokens, and space `⎵` is just another character, the decoder swallowed the space entirely (outputting `"helloworld"`). This happens because the decoder implicitly relies on `##` logic instead of dynamically checking the original pre-tokenizer boundaries.

---

## 4. Thinking Process & Experiments

To move beyond hypotheses, we wrote an experimental script (`hackathon_tasks/experiment_task1.py`) that trained both BPE and Unigram tokenizers on the exact same corpus to observe how they tackle the Sanskrit mantra differently.

### Empirical Results

| Tokenizer Model | Target Vocab | Tokens Generated | Sample Output (First Few Tokens) |
|-----------------|--------------|------------------|----------------------------------|
| BPE             | 300          | 47 tokens        | `['ॐ', 'भ', '##ू', '##र्', '##भु', '##व', '##ः']` |
| BPE             | 1,000        | 47 tokens        | `['ॐ', 'भ', '##ू', '##र्', '##भु', '##व', '##ः']` |
| Unigram         | 300          | 13 tokens        | `['ॐ', 'भूर्भुवः', 'स्व:', 'त', 'त्सवितुर्वरेण्यं']` |

### **Analysis of the Experiment**

1. **BPE is Bottlenecked by Data:** Even when we gave BPE a much larger vocabulary allowance (1,000 slots), it still produced 47 highly fragmented character-level pieces. Because our corpus was tiny, BPE couldn't find enough statistically frequent adjacent pairs to justify merging. BPE is strictly bottom-up and greedy in this codebase.
2. **Unigram is Superior for Scarce Data:** Unigram drastically reduced the fertility down to just 13 tokens using the exact same corpus and vocab limit (300). Because `UnigramTrainer` works top-down (starts with a massive probabilistic vocabulary and prunes away the least useful pieces probabilistically), it successfully memorized complex chunks like `'भूर्भुवः'` and `'त्सवितुर्वरेण्यं'` as single tokens. 

**Conclusion:** For morphologically complex scripts like Devanagari where base characters and modifiers combine, `UnigramTrainer` provides vastly superior segmentations out-of-the-box compared to BPE when the training data is scarce *but representative of the input*.

### **Experiment 2: The Out-Of-Vocabulary (OOV) Collapse**
We ran one final test to demonstrate *why* BPE is the industry standard despite Unigram's better segmentation. We removed our Sanskrit mantra entirely from the training corpus, retrained all three supported models (`BPE`, `Unigram`, `WordLevel`), and tried to encode the unseen mantra.

**The Results on Unseen Data:**
- **WordLevel (300 Vocab):** Produced `['<unk>', '<unk>', '<unk>', '<unk>', '<unk>']`. Complete catastrophic failure.
- **Unigram (300 Vocab):** Produced 63 tokens (mostly `<unk>` and fragmented characters). It panics when probabilities hit zero.
- **BPE (300 Vocab):** Produced 62 character tokens `['ॐ', 'भ', '##ू', '##र', '##्'...]` but **zero `<unk>` tokens**. 

**Why this matters (The "Why"):**
This proves the architectural trade-off of `BPE` in this codebase. While Unigram creates cleaner tokens on text it knows, BPE is infinitely more robust to Out-Of-Vocabulary text because its initialization step strictly includes all base characters. Given that Devanagari has intense combinatorial explosion of conjunct characters, BPE's character-fallback guarantees that no information is lost (no `UNK`s), even if the fertility is extremely high.
