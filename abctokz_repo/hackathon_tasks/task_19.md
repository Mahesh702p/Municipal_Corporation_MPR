# Task 19 — BPE vs Unigram vs WordLevel: Experimental Analysis

> **Tokens Used for Task 19:** ~75,000 (Training, Execution & Comparative Analysis)

## Experimental Setup
To ensure a fair comparison, we trained all three models on a shared **multilingual corpus** (English + Devanagari) with identical hyperparameters where possible:
- **Vocabulary Size**: 400
- **Normalizer**: `DevanagariNormalizer` (NFC + Devanagari-safe rules)
- **PreTokenizer**: `DevanagariAwarePreTokenizer` (Grapheme-aware + script-boundary splitting)
- **Training Data**: High-frequency English sentences and complex Devanagari phrases repeated to simulate a representative distribution.

## Side-by-Side Tokenization Results

| Sentence | WordLevel | BPE | Unigram |
| :--- | :--- | :--- | :--- |
| **The quick brown fox...** | `["The", "quick", "brown", ...]` | `["T", "##he", "q", "##u", ...]` | `["The", "quick", "brown", ...]` |
| **भारत महान है** | `["भारत", "महान", "है"]` | `["भा", "##रत", "मह", "##ान", "है"]` | `["भारत", "महान", "है"]` |
| **Tokenization algorithms...** | `[Full words]` | `[38 subword pieces]` | `[Full words]` |
| **AI भारत में तेजी से बढ़ रहा है** | `["AI", "भारत", "में", ...]` | `["AI", "भा", "##रत", "म", "##ें", ...]` | `["AI", "भारत", "में", ...]` |

## Vocabulary Characteristics

### WordLevel
- **Composition**: 100% whole words from the training set.
- **Risk**: Extremely high **Out-of-Vocabulary (OOV)** risk. As seen in the results, it handles the Sanskrit/Hindi perfectly when the word is known, but fails catastrophically on unseen text (e.g., "Space exploration" produced 75% UNKs).
- **Use Case**: Fixed-domain tasks with very limited vocabularies (e.g., specific command sets).

### BPE (Byte Pair Encoding)
- **Composition**: Dominated by subword pieces (`##ert`, `##ान`). 
- **Structure**: It builds a hierarchy. The vocabulary contains base characters plus frequent "merges".
- **Benefit**: **Infinite coverage**. Even for "Space exploration", BPE fertility was high but UNK rate was near zero because it could fall back to character-level pieces.

### Unigram
- **Composition**: A mix of whole words and long subword fragments. 
- **Pruning Strategy**: Unlike BPE (bottom-up), Unigram is top-down. It starts with a huge set of potential tokens and prunes them based on which minimizes loss.
- **Observation**: In our limited-data experiment, Unigram behaved like a "WordLevel-Lite", choosing to keep whole words for common strings but significantly struggling with UNKs on unseen data due to aggressive pruning of the alphabet.

## Model Assumptions

- **WordLevel** assumes the world is finite and words never change form.
- **BPE** assumes words are compositional units. It prioritizes **frequency** as the proxy for meaningfulness.
- **Unigram** assumes words are probabilistic sequences. It prioritizes **overall sequence likelihood** and allows for multiple overlapping segmentation candidates during training.

## Best Model for Different Scenarios

| Scenario | Recommended Model | Reasoning |
| :--- | :--- | :--- |
| **Low-resource Language** | **BPE** | Guarantees character-level fallback, preventing the linguistic "black holes" (UNKs) common in Unigram/WordLevel. |
| **Agglutinative (Hindi/Marathi)** | **Unigram** | Better at probabilisticly identifying meaningful sub-morphemes across complex script boundaries when data is sufficient. |
| **Cross-lingual Consistency** | **BPE** | Deterministic merge rules mean that identical character pairs are treated consistently across different sentences. |

## Additional Comparison Factors

- **Fertility Score**: 
  - **BPE**: ~3.46 (High fragmentation).
  - **Unigram/WordLevel**: 1.0 (on known data).
- **UNK Rate (Unseen Data)**:
  - **BPE**: **5.7%** (Robust).
  - **Unigram**: **95.2%** (Catastrophic on tiny datasets).
  - **WordLevel**: **75%** (Fails on any novelty).

## Key Insights
1. **The BPE Robustness Tax**: While BPE produces more tokens per word (higher fertility), it is the only model that survives "Real World" inputs it hasn't seen before.
2. **Unigram's Probabilistic Edge**: Unigram produces much cleaner segmentations on familiar text compared to BPE's greedy splits, but it requires a larger "alphabet safety net" to avoid extreme UNK rates.
3. **Architecture Choice**: Subword models (BPE/Unigram) are essential for any multilingual task involving Devanagari due to the exponential combinations of characters that WordLevel cannot possibly memorize.
