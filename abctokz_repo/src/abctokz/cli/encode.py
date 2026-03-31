# Augenblick — abctokz
"""abctokz encode CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Encode text using a trained tokenizer.", no_args_is_help=True)
console = Console()


@app.callback(invoke_without_command=True)
def encode_cmd(
    model: Path = typer.Option(..., "--model", "-m", help="Path to tokenizer artifact directory."),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Text to encode."),
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="File with lines to encode."),
    show_offsets: bool = typer.Option(False, "--offsets", help="Show character offsets."),
    ids_only: bool = typer.Option(False, "--ids", help="Output only IDs (space-separated)."),
) -> None:
    """Encode text or file lines with a trained tokenizer.

    Examples::

        abctokz encode --model artifacts/bpe_tok --text "नमस्ते world"
        abctokz encode --model artifacts/bpe_tok --input sentences.txt --ids
    """
    from abctokz.tokenizer import Tokenizer

    tokenizer = Tokenizer.load(str(model))

    lines: list[str] = []
    if text:
        lines = [text]
    elif input_file:
        lines = [ln.strip() for ln in input_file.open(encoding="utf-8") if ln.strip()]
    else:
        console.print("[red]Error:[/red] Provide --text or --input")
        raise typer.Exit(1)

    for line in lines:
        enc = tokenizer.encode(line)
        if ids_only:
            console.print(" ".join(map(str, enc.ids)))
        else:
            table = Table(title=f"Encoding: {line[:60]}{'...' if len(line) > 60 else ''}")
            table.add_column("Pos", justify="right", style="dim")
            table.add_column("Token", style="cyan")
            table.add_column("ID", justify="right", style="green")
            if show_offsets:
                table.add_column("Offset", style="yellow")
            for i, (tok, tid) in enumerate(zip(enc.tokens, enc.ids)):
                row = [str(i), tok, str(tid)]
                if show_offsets and enc.offsets:
                    row.append(str(enc.offsets[i]))
                table.add_row(*row)
            console.print(table)
