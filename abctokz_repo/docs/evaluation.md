# Evaluation Guide

## Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| Fertility | tokens / reference words | ≥1.0 | Lower |
| Mean tokens/sentence | avg tokens per sentence | — | Lower |
| UNK rate | fraction of `<unk>` tokens | 0–1 | Lower |
| Round-trip success | exact encode-decode match | 0–1 | Higher |
| Seq-length ratio | tokens / chars | — | Context-dependent |
| Throughput | sentences/second | — | Higher |

## Running a Benchmark

```python
from abctokz.config.schemas import BenchmarkConfig
from abctokz.eval.benchmark import BenchmarkRunner

config = BenchmarkConfig(
    name="my_benchmark",
    corpus_paths=["data/hi.txt", "data/en.txt"],
    tokenizer_paths=["artifacts/bpe_tok", "artifacts/unigram_tok"],
    sample_size=1000,
    warmup_runs=3,
    timed_runs=10,
    output_dir="benchmarks/outputs",
)

runner = BenchmarkRunner(config)
results = runner.run()
paths = runner.save_results(results)
print(f"JSON: {paths['json']}")
print(f"Markdown: {paths['markdown']}")
```

## Intrinsic Evaluation

```python
from abctokz.eval.intrinsic import evaluate_tokenizer
from abctokz.tokenizer import Tokenizer

tokenizer = Tokenizer.load("artifacts/bpe_tok")
sentences = ["नमस्ते दुनिया", "hello world", ...]

result = evaluate_tokenizer(tokenizer, sentences, name="bpe", language="mixed")
print(result.to_dict())
```
