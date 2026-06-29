from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from agentic_retrieval_matrix import __version__
from agentic_retrieval_matrix.eval import (
    load_benchmark,
    load_longmemeval_subset,
    print_matrix_table,
    run_matrix,
    save_results,
)
from agentic_retrieval_matrix.types import AnswerMode, DeliveryKind, HarnessKind, RetrieverKind

app = typer.Typer(
    name="arm",
    help="Factorial benchmark for agentic search (retriever × delivery × harness).",
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"arm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Agentic Retrieval Matrix CLI."""
    pass  # pragma: no cover


def _parse_enum_list(raw: str | None, enum_cls, default):
    if not raw:
        return default
    out = []
    for part in raw.split(","):
        part = part.strip().lower()
        out.append(enum_cls(part))
    return out


@app.command("matrix")
def matrix_cmd(
    benchmark: Path = typer.Option(
        ...,
        "--benchmark",
        "-b",
        help="Path to benchmark.json (see examples/fixture)",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(Path("results"), "--output", "-o", help="Results directory"),
    work_dir: Path = typer.Option(Path(".arm_runs"), "--work-dir", help="Scratch workspace"),
    retrievers: str | None = typer.Option(
        None,
        "--retrievers",
        help="Comma-separated: grep,vector",
    ),
    deliveries: str | None = typer.Option(
        None,
        "--deliveries",
        help="Comma-separated: inline,file",
    ),
    harnesses: str | None = typer.Option(
        None,
        "--harnesses",
        help="Comma-separated harness ids (default: react)",
    ),
    answer_mode: AnswerMode = typer.Option(
        AnswerMode.BLIND,
        "--answer-mode",
        help="blind (default) or oracle (debug upper bound)",
    ),
):
    """Run the full retrieval × delivery × harness matrix."""
    questions, corpora = load_benchmark(benchmark)
    r = _parse_enum_list(retrievers, RetrieverKind, list(RetrieverKind))
    d = _parse_enum_list(deliveries, DeliveryKind, list(DeliveryKind))
    # Vector is optional — skip if extras not installed and user didn't force it
    if RetrieverKind.VECTOR in r:
        try:
            import faiss  # noqa: F401
            import sentence_transformers  # noqa: F401
        except ImportError:
            console.print("[yellow]Skipping vector: install with pip install '.[vector]'[/]")
            r = [x for x in r if x != RetrieverKind.VECTOR]

    h = _parse_enum_list(harnesses, HarnessKind, [HarnessKind.REACT])

    cells = run_matrix(
        questions,
        corpora,
        r,
        d,
        h,
        work_dir,
        answer_mode=answer_mode,
    )
    print_matrix_table(cells)
    path = save_results(cells, output)
    console.print(f"[green]Saved[/] {path}")


@app.command("single")
def single_cmd(
    benchmark: Path = typer.Option(..., "--benchmark", "-b", exists=True),
    retriever: RetrieverKind = typer.Option(RetrieverKind.GREP, "--retriever", "-r"),
    delivery: DeliveryKind = typer.Option(DeliveryKind.INLINE, "--delivery", "-d"),
    work_dir: Path = typer.Option(Path(".arm_runs"), "--work-dir"),
    answer_mode: AnswerMode = typer.Option(AnswerMode.BLIND, "--answer-mode"),
):
    """Run one configuration (fast iteration)."""
    questions, corpora = load_benchmark(benchmark)
    cells = run_matrix(
        questions,
        corpora,
        [retriever],
        [delivery],
        [HarnessKind.REACT],
        work_dir,
        answer_mode=answer_mode,
    )
    print_matrix_table(cells)
    path = save_results(cells, Path("results"))
    console.print(f"[green]Saved[/] {path}")


@app.command("longmem")
def longmem_cmd(
    data_dir: Path = typer.Option(..., "--data-dir"),
    output: Path = typer.Option(Path("results"), "--output", "-o"),
    limit: int | None = typer.Option(None, "--limit"),
    retrievers: str | None = typer.Option(None, "--retrievers"),
    deliveries: str | None = typer.Option(None, "--deliveries"),
):
    """Run matrix on a normalized LongMemEval subset directory."""
    questions, corpora = load_longmemeval_subset(data_dir, limit=limit)
    r = _parse_enum_list(retrievers, RetrieverKind, [RetrieverKind.GREP])
    d = _parse_enum_list(deliveries, DeliveryKind, list(DeliveryKind))
    cells = run_matrix(questions, corpora, r, d, [HarnessKind.REACT])
    print_matrix_table(cells)
    path = save_results(cells, output)
    console.print(f"[green]Saved[/] {path}")


if __name__ == "__main__":
    app()
