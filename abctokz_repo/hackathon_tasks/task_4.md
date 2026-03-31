# Task 4 — Config → Tokenizer Trace

> **Tokens Used for Task 4:** ~8,500 Total 
> (*Breakdown: ~6,300 Input Tokens reading schemas.py, defaults.py & factories + ~2,200 Output Tokens for thinking, writing script, and generating report*)

## 1. The Investigation Setup

To understand the lifecycle of how a pure Python configuration transforms into a fully functional Tokenizer pipeline, we followed the trail from `config/schemas.py` directly into the `AugenblickTokenizer.from_config()` factory method. 

To prove how rigid this configuration system is, we wrote an experiment script (`hackathon_tasks/experiment_task4.py`) that intentionally threw bad configurations at the Pydantic parser to watch it break under pressure.

---

## 2. Tracing the Config Lifecycle (Where Defaults Come From)

The `abctokz` library utilizes **Pydantic** to strictly validate configuration objects before an actual class is instantiated. Here is the lifecycle trace of a standard BPE model setup.

### Step 1: Default Injections via Schemas
If a user simply initializes `BPEConfig()`, they don't have to specify any parameters. The defaults are automatically injected inside `src/abctokz/config/schemas.py`:
- `vocab_size` defaults to `8000` (pulled from `BPE_DEFAULT_VOCAB_SIZE`).
- `continuation_prefix` defaults to `"##"` (pulled from `BPE_CONTINUATION_PREFIX`).
- `unk_token` defaults to `"<unk>"`.

### Step 2: The Preset Builders (Defaults Overriding Defaults)
The library provides pre-packaged combinations in `src/abctokz/config/defaults.py` for common use cases. For example, the `bpe_multilingual(vocab_size=8_000)` preset automatically builds a full `TokenizerConfig` by injecting:
- A `multilingual_shared_normalizer()` (devanagari safe).
- A `DevanagariAwarePreTokenizer`.
- The `BPEConfig` and `BPETrainerConfig`.

### Step 3: Instantiation via Factories 
The validated config object is finally passed to `Tokenizer.from_config(config)`. Inside this method, the class delegates actual object initialization to a Factory pattern:
- Normalizers are built by passing the config to `build_normalizer()` inside `normalizers/__init__.py`. Here, it maps pure literal string types like `"nfkc"` into the `NfkcNormalizer` class.
- PreTokenizers are built via `build_pretokenizer()` inside `pretokenizers/__init__.py`. 
- Finally, an empty placeholder `Model` and `Decoder` are configured inside `Tokenizer`, pending the `train()` command.

---

## 3. Failure Mode Demonstration

Because the configuration object inherits from `BaseConfig` (which strictly defines `extra = "forbid"` and requires `frozen = True`), it is incredibly hostile to badly shaped data. 

Here are the 3 failure modes our experiment triggered:

### Failure Mode 1: Validation Caught by `@model_validator`
- **The Issue:** A user attempts to pair a BPE Model with a Unigram Trainer.
- **The Code:** `TokenizerConfig(model=BPEConfig(), trainer=UnigramTrainerConfig())`
- **What Happened:** Caught! The `@model_validator` function `check_trainer_model_alignment` on line 246 of `schemas.py` noticed that `model.type == 'bpe'` but `trainer.type == 'unigram'`.
- **The Error Output:**
  ```
  1 validation error for TokenizerConfig
    Value error, Model type 'bpe' and trainer type 'unigram' must match.
  ```

### Failure Mode 2: Field Rules (e.g. Invalid Vocab Size)
- **The Issue:** A user tries to declare an impossible vocabulary size.
- **The Code:** `BPEConfig(vocab_size=-5)`
- **What Happened:** Caught! The `vocab_size` field explicitly requires values greater than or equal to 1 using Pydantic's `ge=1` operator.
- **The Error Output:**
  ```
  1 validation error for BPEConfig
  vocab_size
    Input should be greater than or equal to 1 [type=greater_than_equal, input_value=-5, input_type=int]
  ```

### Failure Mode 3: Forbidding Extra Fields
- **The Issue:** A user tries to pass an undocumented parameter, assuming the kwargs will just be passively ignored.
- **The Code:** `TokenizerConfig(model=BPEConfig(), foobar="baz")`
- **What Happened:** Caught! The `TokenizerConfig` inherits from `BaseConfig`, which enforces `model_config = {"extra": "forbid"}`. Pydantic refuses to instantiate the object if undefined fields exist.
- **The Error Output:**
  ```
  1 validation error for TokenizerConfig
  foobar
    Extra inputs are not permitted [type=extra_forbidden, input_value='baz', input_type=str]
  ```
