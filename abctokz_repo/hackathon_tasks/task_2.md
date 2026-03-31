# Task 2 — Who Does What? Mapping Module Responsibilities

> **Tokens Used for Task 2:** ~14,200 (Estimated total including codebase exploration, reading 8 module files, tool calls, and final structured generation)

## 1. The Investigation Setup
To map the responsibilities accurately, we explored the `abctokz` codebase, inspecting the internal directory structure and diving deep into the imports and classes of critical files: `tokenizer.py`, `models/base.py`, `trainers/base.py`, `eval/metrics.py`, `eval/benchmark.py`, `adapters/hf.py`, and `vocab/serialization.py`.

---

## 2. Responsibility → Module Mapping

### Training a tokenizer (learning vocabulary from text)
- **Involved Modules:** `abctokz.trainers` (`trainers/base.py`, `trainers/bpe_trainer.py`, etc.)
- **Why it exists:** The `Trainer` abstract base class acts as a factory. It completely isolates corpus reading, frequency counting, and vocabulary building logic away from runtime inference. Its only job is to consume an iterator of text and produce a trained `Model` artifact.

### Using a trained tokenizer to encode new text
- **Involved Modules:** `abctokz.tokenizer` and `abctokz.models`
- **Why it exists:** `tokenizer.py` serves as the high-level orchestrator that runs the pipeline (normalizer → pretokenizer → model → post-processor). Then, the mathematical subword logic (mapping pieces to IDs) lives in stateless `models/` like `BPEModel`. This separation allows the core subword math to remain completely ignorant of higher-level logic like lowercasing or sentence splitting.

### Saving and loading a tokenizer to/from disk
- **Involved Modules:** `abctokz.tokenizer` and `abctokz.vocab.serialization`
- **Why it exists:** `Tokenizer.save()` manages the high-level metadata like `manifest.json`. It delegates the low-level serialization of vocabularies and merge tables to `vocab/serialization.py`. This ensures standard JSON/txt formatting rules aren't hardcoded into every individual model class.

### Measuring tokenizer quality (fertility, UNK rate, etc.)
- **Involved Modules:** `abctokz.eval.metrics`
- **Why it exists:** Located in the `eval` directory, `metrics.py` defines pure mathematical functions (`fertility`, `unk_rate`). They compute objective metrics using only the standard `Encoding` data types. By keeping them out of the models, we ensure evaluation logic isn't inadvertently biased by a model's internal state.

### Comparing abctokz against external tokenizers (HuggingFace, SentencePiece)
- **Involved Modules:** `abctokz.eval.benchmark` and `abctokz.adapters`
- **Why it exists:** The `adapters` pattern (`hf.py`) wraps external libraries behind `abctokz`'s internal `Encoding` interface. The `BenchmarkRunner` orchestration then only needs to speak one language. This enables an apples-to-apples comparison without the benchmark logic needing custom code for HuggingFace's specific API.

---

## 3. Clean Module Boundary

**Module:** `abctokz.models` (e.g., `Model` base class in `models/base.py`)

**Why this boundary is satisfying:**
This is an exceptionally clean boundary because `Model` handles *only* the core algorithm for tokenizing a single, already-pre-tokenized string into subword tokens and IDs. 

It explicitly forces the model to be naive. It does not import anything from `trainers` or `pretokenizers`. It knows nothing about file I/O, white-spaces, punctuation splitting, or iterations. This strict separation satisfies the Open-Closed Principle because you could easily drop in a completely new algorithm (like WordPiece) by subclassing `Model` without ever touching the complex text iteration, special token masking, or pipeline sequence code living in `tokenizer.py`.

---

## 4. Blurry or Inconsistent Boundary

**Modules involved:** `abctokz.tokenizer.Tokenizer` and `abctokz.vocab.serialization`

**Why the boundary feels blurry:**
The `Tokenizer` class is violating the Single Responsibility Principle. While it serves as the runtime execution pipeline (orchestrating `normalizer → pretokenizer → model`), it also heavily entangles itself in input/output mechanics, file path construction, and directory management within its `.load()` and `.save()` methods. For example, `Tokenizer.load()` possesses hardcoded switch-cases (`if model_type == "bpe"`) to instantiate specific classes from disk, overlapping significantly with `vocab/serialization.py` responsibilities.

**What we would do about it (Proposed Fix):**
We should extract all top-level save/load artifact directory logic out of `Tokenizer` and move it into a dedicated `ArtifactStore` object within a new module, such as `abctokz.io.artifact`. 

The `Tokenizer` class code should solely be responsible for the execution of the state-machine pipeline in memory. The `ArtifactStore` would take ownership of hardcoded file names (like `MANIFEST_FILENAME`), schema version validation, and object instantiation. This would cleanly and permanently separate runtime text processing logic from filesystem persistence logic.
