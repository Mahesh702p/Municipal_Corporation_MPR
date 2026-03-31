# Augenblick — abctokz
"""abctokz inspect CLI command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Inspect a trained tokenizer artifact.", no_args_is_help=True)
console = Console()


@app.callback(invoke_without_command=True)
def inspect_cmd(
    model: Path = typer.Option(..., "--model", "-m", help="Path to tokenizer artifact directory."),
    top_n: int = typer.Option(20, "--top-n", help="Number of vocabulary entries to show."),
    show_vocab: bool = typer.Option(False, "--vocab", help="Print the full vocabulary."),
) -> None:
    """Inspect a saved tokenizer artifact: metadata, config, and top vocab.

    Examples::

        abctokz inspect --model artifacts/bpe_tok
        abctokz inspect --model artifacts/bpe_tok --vocab --top-n 50
    """
    from abctokz.utils.io import load_json
    from abctokz.constants import MANIFEST_FILENAME, CONFIG_FILENAME

    p = Path(model)
    if not p.exists():
        console.print(f"[red]Artifact directory not found:[/red] {p}")
        raise typer.Exit(1)

    manifest_path = p / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        info_table = Table(title="Manifest", show_header=False)
        info_table.add_column("Key", style="bold cyan")
        info_table.add_column("Value")
        for k, v in manifest.items():
            info_table.add_row(str(k), str(v))
        console.print(info_table)

    # Show vocab sample
    vocab_path = p / "vocab.json"
    pieces_path = p / "pieces.json"

    if vocab_path.exists():
        vocab = load_json(vocab_path)
        _show_vocab_table(vocab, top_n if not show_vocab else len(vocab))
    elif pieces_path.exists():
        pieces = load_json(pieces_path)
        _show_pieces_table(pieces, top_n if not show_vocab else len(pieces))

    # Show merge count for BPE
    merges_path = p / "merges.txt"
    if merges_path.exists():
        n_merges = sum(
            1 for ln in merges_path.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")
        )
        console.print(f"\n[bold]BPE merge rules:[/bold] {n_merges}")


def _show_vocab_table(vocab: dict, top_n: int) -> None:
    """Print vocab as a Rich table."""
    table = Table(title=f"Vocabulary (top {min(top_n, len(vocab))} of {len(vocab)})")
    table.add_column("Token", style="cyan")
    table.add_column("ID", justify="right", style="green")
    items = sorted(vocab.items(), key=lambda x: x[1])[:top_n]
    for tok, tid in items:
        table.add_row(repr(tok), str(tid))
    console.print(table)


def _show_pieces_table(pieces: list, top_n: int) -> None:
    """Print Unigram pieces as a Rich table."""
    table = Table(title=f"Pieces (top {min(top_n, len(pieces))} of {len(pieces)})")
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Piece", style="cyan")
    table.add_column("Score", justify="right", style="yellow")
    for i, (piece, score) in enumerate(pieces[:top_n]):
        table.add_row(str(i), repr(piece), f"{score:.4f}")
    console.print(table)
