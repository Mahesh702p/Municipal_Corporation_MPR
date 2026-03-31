# Augenblick — abctokz
"""abctokz decode CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Decode token IDs using a trained tokenizer.", no_args_is_help=True)
console = Console()


@app.callback(invoke_without_command=True)
def decode_cmd(
    model: Path = typer.Option(..., "--model", "-m", help="Path to tokenizer artifact directory."),
    ids: Optional[str] = typer.Option(None, "--ids", help="Comma-separated token IDs."),
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="File with one ID sequence per line."),
    keep_special: bool = typer.Option(False, "--keep-special", help="Keep special tokens in output."),
) -> None:
    """Decode token IDs back to text.

    Examples::

        abctokz decode --model artifacts/bpe_tok --ids "12,98,44,3"
        abctokz decode --model artifacts/bpe_tok --input ids.txt
    """
    from abctokz.tokenizer import Tokenizer

    tokenizer = Tokenizer.load(str(model))

    lines: list[str] = []
    if ids:
        lines = [ids]
    elif input_file:
        lines = [ln.strip() for ln in input_file.open(encoding="utf-8") if ln.strip()]
    else:
        console.print("[red]Error:[/red] Provide --ids or --input")
        raise typer.Exit(1)

    for line in lines:
        token_ids = [int(x.strip()) for x in line.split(",") if x.strip()]
        result = tokenizer.decode(token_ids, skip_special_tokens=not keep_special)
        console.print(result)
