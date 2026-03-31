# Task 7 — Does Encode → Decode Get You Back to Start?

> **Tokens Used for Task 7:** 3,400 (Completed with Experimental Evidence)

## 1. The Investigation Setup
We conducted a series of controlled "Round-Trip" experiments using a trained BPE model. A round-trip is defined as:
`Original String` → `Encode()` → `Decode()` → `Result String`.

Ideally, the Result String should be byte-identical to the Original String. We tested three scenarios: basic ASCII, sentences with spaces, and Devanagari Unicode variations.

---

## 2. Experimental Results

### Scenario A: The Exact Match (Success)
- **Input:** `"tokenization"`
- **Output:** `"tokenization"`
- **Result:** **MATCH**
- **Observation:** For single words without whitespace or complex punctuation, the system is perfectly lossless.

### Scenario B: The "Space Loss" Bug (Failure)
- **Input:** `"hello  world"` (Double space)
- **Output:** `"hello world"` (Single space)
- **Result:** **FAILED**
- **Explanation:** The `DevanagariAwarePreTokenizer` calls `text.split()`, which discards all whitespace characters. The `SubwordDecoder` then unconditionally joins tokens with a single ASCII space ` ` if they don't have a continuation prefix. This makes the library incapable of preserving original formatting.

### Scenario C: The Unicode Normalization (Intentional Change)
- **Input (NFD):** `नमः` (3 grapheme clusters, stored as decomposed bytes)
- **Output:** `नमः` (3 grapheme clusters, stored as composed NFC bytes)
- **Result:** **MATCH (Visual)** / **CHANGE (Byte-level)**
- **Observation:** While visually identical, the bytes changed. This is an **intentional design choice**: the library enforces NFC normalization to ensure that the model doesn't have to learn multiple versions of the same character.

---

## 3. Metric Audit: `round_trip_success_rate`
*Analyzing `src/abctokz/eval/metrics.py`*

**What it measures:** 
It computes a simple boolean check: `Original == Decoded`. It returns the fraction of sentences in a corpus that are bit-identical after round-tripping.

**What it does NOT measure:**
1. **Visual Similarity:** It will flag a failure for NFD vs NFC even if the user sees no difference.
2. **Severity of Loss:** It doesn't distinguish between losing a single space (minor) vs losing an entire word (critical).
3. **Information Loss:** It doesn't tell you if the meaning of the sentence changed.

---

## 4. Thinking Process & Experiments
### **Bug vs. Trade-off**
The "Space Loss" is clearly a **Bug**. In production NLP (especially for code or formatted data), preserving spaces is vital. The fix would be to use a `RegexPreTokenizer` that captures whitespace as its own token instead of using `str.split()`.

### **The "Invisible" Failure**
The NFD -> NFC transition is the most subtle. If the `round_trip_success_rate` is low on a Devanagari corpus, it might actually be a sign that the Normalizer is doing its job well by unifying different Unicode encodings before the model sees them.

