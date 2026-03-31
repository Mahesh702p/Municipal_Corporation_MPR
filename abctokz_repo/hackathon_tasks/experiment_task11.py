import sys
import tempfile
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent.resolve()
SRC_ROOT = (current_dir.parent / "src").resolve()
sys.path.append(str(SRC_ROOT))

from abctokz.config.schemas import BenchmarkConfig, TokenizerConfig
from abctokz.eval.benchmark import BenchmarkRunner
from abctokz.tokenizer import Tokenizer
from abctokz.config.defaults import bpe_multilingual

def run_audit():
    print("--- Task 11: Benchmark Stability Audit ---\n")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # 1. Train a dummy tokenizer
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("This is an English sentence for benchmarking purposes. " * 100)
        
        config = bpe_multilingual(vocab_size=100)
        tok = Tokenizer.from_config(config)
        tok.train([str(corpus_file)], config)
        
        tok_save_path = tmp_path / "my_tokenizer"
        tok.save(str(tok_save_path))
        
        # 2. Setup Benchmark runner
        bench_config = BenchmarkConfig(
            name="StabilityAudit",
            corpus_paths=[str(corpus_file)],
            tokenizer_paths=[str(tok_save_path)],
            sample_size=50,
            warmup_runs=2,
            timed_runs=5,
            output_dir=str(tmp_path / "results")
        )
        
        runner = BenchmarkRunner(bench_config)
        
        # Run 1
        print("Running Benchmark 1...")
        results1 = runner.run()[0]
        
        # Run 2
        print("Running Benchmark 2...")
        results2 = runner.run()[0]
        
        print("\n--- RESULTS COMPARISON ---\n")
        print(f"{'Metric':<30} | {'Run 1':<15} | {'Run 2':<15} | {'Stable?'}")
        print("-" * 75)
        
        metrics = [
            ("Fertility", results1.fertility, results2.fertility),
            ("UNK Rate", results1.unk_rate, results2.unk_rate),
            ("RT Success Rate", results1.round_trip_success_rate, results2.round_trip_success_rate),
            ("Throughput (SPS)", results1.throughput_sps, results2.throughput_sps),
            ("Mean Tokens/Sent", results1.mean_tokens_per_sentence, results2.mean_tokens_per_sentence)
        ]
        
        for name, r1, r2 in metrics:
            is_stable = "YES" if abs(r1 - r2) < 1e-9 else "NO (Variance!)"
            print(f"{name:<30} | {r1:<15.4f} | {r2:<15.4f} | {is_stable}")

if __name__ == "__main__":
    run_audit()
