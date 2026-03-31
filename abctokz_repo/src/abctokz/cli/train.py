# Augenblick — abctokz
"""abctokz train CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console

app = typer.Typer(help="Train a tokenizer from a config file or inline options.", no_args_is_help=True)
console = Console()


@app.callback(invoke_without_command=True)
def train(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML training config file."),
    corpus: Optional[list[Path]] = typer.Option(None, "--corpus", help="Corpus file(s)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output artifact directory."),
    model_type: str = typer.Option("bpe", "--model", "-m", help="Model type: wordlevel|bpe|unigram."),
    vocab_size: int = typer.Option(8000, "--vocab-size", help="Target vocabulary size."),
    min_frequency: int = typer.Option(2, "--min-freq", help="Minimum token frequency."),
) -> None:
    """Train a abctokz tokenizer.

    Examples::

        abctokz train --config configs/bpe_hi_mr_sd_en.yaml
        abctokz train --corpus data/hi.txt --model bpe --vocab-size 8000 --output artifacts/bpe
    """
    from abctokz.config.schemas import (
        BPEConfig,
        BPETrainerConfig,
        TokenizerConfig,
        UnigramConfig,
        UnigramTrainerConfig,
        WordLevelConfig,
        WordLevelTrainerConfig,
    )
    from abctokz.config.defaults import multilingual_shared_normalizer, bpe_multilingual, unigram_multilingual, wordlevel_multilingual
    from abctokz.tokenizer import Tokenizer

    if config is not None:
        # Load from YAML
        with open(config, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        from abctokz.config.schemas import TrainingRunConfig
        run_cfg = TrainingRunConfig(**raw)
        corpus_paths = run_cfg.corpus
        tok_config = run_cfg.tokenizer
        output_dir = run_cfg.output_dir
    else:
        if not corpus:
            console.print("[red]Error:[/red] --corpus is required when not using --config")
            raise typer.Exit(1)
        if not output:
            console.print("[red]Error:[/red] --output is required when not using --config")
            raise typer.Exit(1)
        corpus_paths = [str(p) for p in corpus]
        output_dir = str(output)

        if model_type == "bpe":
            tok_config = bpe_multilingual(vocab_size)
        elif model_type == "unigram":
            tok_config = unigram_multilingual(vocab_size)
        elif model_type == "wordlevel":
            tok_config = wordlevel_multilingual(vocab_size)
        else:
            console.print(f"[red]Unknown model type:[/red] {model_type!r}")
            raise typer.Exit(1)

    console.print(f"[bold]Training[/bold] {tok_config.model.type} tokenizer...")
    console.print(f"  Corpus: {corpus_paths}")
    console.print(f"  Output: {output_dir}")

    tokenizer = Tokenizer.from_config(tok_config)
    tokenizer.train(corpus_paths, tok_config)
    tokenizer.save(output_dir)

    console.print(f"[green]Done![/green] Tokenizer saved to [bold]{output_dir}[/bold]")
    console.print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")
