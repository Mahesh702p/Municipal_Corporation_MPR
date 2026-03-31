### Problem

The `grapheme_clusters` utility in `src/abctokz/utils/unicode.py` was too simplistic. It only attached Unicode characters to a cluster if they belonged to the "Mark" (M) category (combining marks like matras). 

However, many scripts—especially **Devanagari**—rely on **Zero Width Joiner (ZWJ)** and **Zero Width Non-Joiner (ZWNJ)** characters to control the formation of conjuncts and half-forms (e.g., `क्` + `ZWJ` = `क्‍`). Because these characters are not category "M", the utility was treating them as new, separate clusters, effectively breaking linguistic units. This fragmentation would lead to sub-optimal tokenization and potentially incorrect script rendering after decoding.

### Location

- **File**: `src/abctokz/utils/unicode.py`
- **Function**: `grapheme_clusters`

### Fix

```diff
--- a/src/abctokz/utils/unicode.py
+++ b/src/abctokz/utils/unicode.py
@@
     buf = ""
     for char in text:
-        if is_combining(char) and buf:
+        # IMPROVEMENT: Also attach zero-width characters (ZWJ/ZWNJ) to the cluster
+        if (is_combining(char) or is_zero_width(char)) and buf:
             buf += char
         else:
```

### Why This Fix Is Correct

1.  **Linguistic Integrity**: It ensures that joiner characters, which are semantically part of the preceding grapheme unit, stay attached to that unit.
2.  **Safe & Minimal**: It only changes one condition. Since `is_zero_width` specifically targets ZWJ, ZWNJ, and BOM, it doesn't affect standard alphanumeric or punctuation characters.
3.  **Preserves Architecture**: The utility remains a stateless string processor but now correctly handles a common Unicode edge case required for high-quality multilingual tokenization.

### Evidence

**Before Fix**:
The sequence `क` + `्` + `ZWJ` was split into two clusters: `['क्', '\u200d']`. 
Reproduction script output: `RESULT: FAILURE (ZWJ broke the cluster)`

**After Fix**:
The sequence is correctly grouped as a single linguistic unit: `['क्\u200d']`.
Reproduction script output: `RESULT: SUCCESS (ZWJ correctly attached)`

---

**Tokens Used for Task 17**:
- **Input**: ~45,000
- **Thinking**: ~4,500
- **Output**: ~5,000
- **Total**: ~54,500
