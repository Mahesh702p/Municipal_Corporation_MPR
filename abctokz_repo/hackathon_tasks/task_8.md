# Task 8 — What Does the Normalizer Actually Do?

> **Tokens Used for Task 8:** 2,950 (Includes Advanced Unicode Hex-Audit)

## 1. The Investigation Setup
We analyzed the normalization stage using the `NfcNormalizer` and `DevanagariNormalizer`. To provide a deep technical audit, we didn't just look at the text; we inspected the **Hexadecimal Unicode codepoints** to see changes that are invisible to the naked eye.

We also conducted a "Stress Test" on **Zero Width Joiners (ZWJ)**, which are critical for the correct display of Marathi and Sindhi conjuncts.

---

## 2. Stage-by-Stage Transformation

### Phrase (i): Sindhi Folk Phrase
`"आयो लाल, सभई चायो, झूलेलाल!"`

| Stage | Output | Changes Observed |
|---|---|---|
| **Raw Input** | `"आयो लाल, सभई चायो, झूलेलाल!"` | - |
| **Normalized** | `"आयो लाल, सभई चायो, झूलेलाल!"` | **0 Changes.** Already byte-identical to NFC. |
| **Pre-Tokenized**| `['आयो', 'लाल,', 'सभई', 'चायो,', 'झूलेलाल!']` | Punctuation is **preserved and attached** to words. |

### Phrase (ii): Marathi Ganesh Chant
`"गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"`

| Stage | Output | Changes Observed |
|---|---|---|
| **Raw Input** | `"गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"` | - |
| **Normalized** | `"गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"` | **0 Changes.** No hidden Unicode decomposition found. |
| **Pre-Tokenized**| `['गणपती', 'बप्पा', 'मोरया,', 'पुढच्या', 'वर्षी', 'लवकर', 'या!']` | Commas and exclamation marks remain as word-suffixes. |

---

## 3. The NFC vs. NFKC Choice
*Why does this library use NFC for Devanagari?*

**Finding:** NFKC (Compatibility Decomposition) is far too aggressive for Indic scripts. It can decompose complex ligatures and symbols into their base parts, which might be "semantically correct" for a computer but is visually "broken" for a native speaker. NFC (Canonical Composition) is the "Goldilocks" choice: it fixes computer storage inconsistencies without messing with the cultural look of the script.

---

## 4. Thinking Process & Advanced Experiments: The "ZWJ Audit"
*Standard normalization often ignores "invisible" characters. We tested what happens if we strip Zero Width Joiners (ZWJ) from Devanagari.*

| Word Segment | Original (Visual) | Stripped (Visual) | Impact |
|---|---|---|---|
| `क्‍` (Half-Ka) | `क्‍` (U+0915 U+094D U+200D) | `क्` (U+0915 U+094D) | **CRITICAL**. The half-letter becomes a full letter with a visible halant. |

**Conclusion:** 
During this task, we proved that the `DevanagariNormalizer`'s default `strip_zero_width=False` is **vital**. If we had used standard AI "cleaning" tools that strip invisible characters, we would have corrupted the Marathi and Sindhi phrases by breaking their conjunct forms. This library is "Indic-literate" by design.

