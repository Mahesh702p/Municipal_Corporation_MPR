# Task 11 — Can You Trust the Benchmark Numbers?

> **Tokens Used for Task 11:** 2,100 (Stability Audit)

## 1. The Stability Experiment
We ran the `BenchmarkRunner` multiple times on the same BPE tokenizer using a sample of 50 English sentences. We compared the results of two consecutive runs to identify which metrics are "Trusted" and which are "Variable."

### Comparison Table

| Metric | Run 1 | Run 2 | Verdict |
|---|---|---|---|
| **Fertility** | 6.8738 | 6.8738 | **100% Stable** |
| **UNK Rate** | 0.2179 | 0.2179 | **100% Stable** |
| **Throughput (SPS)** | 142.4 | 227.0 | **Highly Variable** |

---

## 2. Analysis: What is "Trusted"?

### **The "Safe" Metrics (Deterministic)**
Metrics like **Fertility**, **UNK Rate**, and **Mean Tokens per Sentence** are **Trusted**. 
- **Why?** These are mathematical properties of the vocabulary and the text. As long as the model and the input text don't change, these numbers will be identical down to the last decimal place every single time. 

### **The "Unsafe" Metrics (Stochastic)**
**Throughput (SPS)** and **Elapsed Time** are **Not Trusted** as absolute values.
- **Why?** They depend on external factors: CPU temperature, background processes (like an OS update or a browser), and memory allocation. In our experiment, throughput jumped by **~60%** just between two runs!

---

## 3. The Design Critique
*Look at `src/abctokz/eval/benchmark.py`*

The runner averages the elapsed time over `timed_runs`. While this helps smooth out "spikes," it still produces a number that is only valid for *that specific machine* at *that specific moment*.

**Recommendation for a more trustworthy benchmark:**
Instead of just reporting "Throughput," the benchmark should report the **Standard Deviation** of the time. This would tell the user if the measurement was stable or if it was fluctuating wildly.

---

## 4. Conclusion
You can trust the **Quality** metrics (Fertility), but you should treat the **Speed** metrics (Throughput) as rough estimates, not scientific law.
