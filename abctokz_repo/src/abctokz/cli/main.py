# Augenblick — abctokz
"""abctokz CLI entrypoint."""

from __future__ import annotations

import typer
from rich.console import Console

from abctokz.cli import benchmark, decode, encode, inspect, train

app = typer.Typer(
    name="abctokz",
    help="abctokz — multilingual tokenizer CLI (English + Devanagari).",
    add_completion=True,
    no_args_is_help=True,
)

console = Console()

app.add_typer(train.app, name="train")
app.add_typer(encode.app, name="encode")
app.add_typer(decode.app, name="decode")
app.add_typer(inspect.app, name="inspect")
app.add_typer(benchmark.app, name="benchmark")


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """abctokz: build, train, and evaluate multilingual tokenizers."""
    if verbose:
        from abctokz.utils.logging import configure_root_logger
        import logging
        configure_root_logger(logging.DEBUG)
