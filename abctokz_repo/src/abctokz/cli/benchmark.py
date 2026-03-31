# Augenblick — abctokz
"""abctokz benchmark CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Run benchmarks comparing tokenizers.", no_args_is_help=True)
console = Console()


@app.callback(invoke_without_command=True)
def benchmark_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML benchmark config file."),
    corpus: Optional[list[Path]] = typer.Option(None, "--corpus", help="Corpus file(s)."),
    models: Optional[list[Path]] = typer.Option(None, "--model", "-m", help="Tokenizer artifact path(s)."),
    sample_size: int = typer.Option(1000, "--sample-size", help="Number of sentences to sample."),
    output_dir: str = typer.Option("benchmarks/outputs", "--output-dir", help="Output directory."),
    name: str = typer.Option("benchmark", "--name", help="Benchmark name."),
) -> None:
    """Run tokenizer benchmarks and emit JSON + Markdown reports.

    Examples::

        abctokz benchmark --config benchmarks/configs/core.yaml
        abctokz benchmark --corpus data/hi.txt --model artifacts/bpe --model artifacts/unigram
    """
    import yaml
    from abctokz.config.schemas import BenchmarkConfig
    from abctokz.eval.benchmark import BenchmarkRunner

    if config is not None:
        with open(config, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        bench_cfg = BenchmarkConfig(**raw)
    else:
        if not corpus or not models:
            console.print("[red]Error:[/red] Provide --config or both --corpus and --model flags")
            raise typer.Exit(1)
        bench_cfg = BenchmarkConfig(
            name=name,
            corpus_paths=[str(p) for p in corpus],
            tokenizer_paths=[str(p) for p in models],
            sample_size=sample_size,
            output_dir=output_dir,
        )

    console.print(f"[bold]Running benchmark:[/bold] {bench_cfg.name}")
    runner = BenchmarkRunner(bench_cfg)
    results = runner.run()
    paths = runner.save_results(results)

    console.print(f"\n[green]Benchmark complete.[/green]")
    console.print(f"  JSON report: [bold]{paths['json']}[/bold]")
    console.print(f"  Markdown report: [bold]{paths['markdown']}[/bold]")

    # Print summary table
    from abctokz.eval.reports import results_to_markdown
    console.print("\n" + results_to_markdown(results, title=bench_cfg.name))
