from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agentic_retrieval_matrix.eval.grader import grade_answer
from agentic_retrieval_matrix.harness import build_harness
from agentic_retrieval_matrix.types import (
    DeliveryKind,
    HarnessKind,
    MatrixCell,
    MemoryCorpus,
    Question,
    RetrieverKind,
    RunConfig,
)

console = Console()


def run_cell(
    questions: list[Question],
    corpora: dict[str, MemoryCorpus],
    retriever: RetrieverKind,
    delivery: DeliveryKind,
    harness: HarnessKind,
    work_dir: Path,
) -> MatrixCell:
    agent = build_harness(harness)
    memory_root = work_dir / "memory"
    details: list[dict] = []
    correct = 0

    for q in questions:
        corpus = corpora[q.session_id]
        config = RunConfig(
            retriever=retriever,
            delivery=delivery,
            harness=harness,
            work_dir=str(work_dir),
        )
        answer = agent.run(q, corpus, config, memory_root)
        ok = grade_answer(q, answer)
        correct += int(ok)
        details.append(
            {
                "question_id": q.id,
                "category": q.category,
                "correct": ok,
                "predicted": answer.answer,
                "gold": q.gold_answer,
                "steps": answer.steps,
                "evidence_turns": [h.turn_id for h in answer.evidence],
            }
        )

    n = len(questions)
    return MatrixCell(
        retriever=retriever,
        delivery=delivery,
        harness=harness,
        accuracy=correct / n if n else 0.0,
        n=n,
        details=details,
    )


def run_matrix(
    questions: list[Question],
    corpora: dict[str, MemoryCorpus],
    retrievers: list[RetrieverKind] | None = None,
    deliveries: list[DeliveryKind] | None = None,
    harnesses: list[HarnessKind] | None = None,
    work_dir: Path | None = None,
) -> list[MatrixCell]:
    retrievers = retrievers or list(RetrieverKind)
    deliveries = deliveries or list(DeliveryKind)
    harnesses = harnesses or [HarnessKind.REACT]
    work_dir = work_dir or Path(".arm_runs")

    cells: list[MatrixCell] = []
    for r, d, h in itertools.product(retrievers, deliveries, harnesses):
        console.print(f"[bold]Running[/] retriever={r.value} delivery={d.value} harness={h.value}")
        cell = run_cell(questions, corpora, r, d, h, work_dir)
        cells.append(cell)
        console.print(f"  accuracy={cell.accuracy:.1%} ({int(cell.accuracy * cell.n)}/{cell.n})")
    return cells


def save_results(cells: list[MatrixCell], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"matrix_{ts}.json"
    payload = {
        "generated_at": ts,
        "cells": [c.model_dump() for c in cells],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_matrix_table(cells: list[MatrixCell]) -> None:
    table = Table(title="Agentic Retrieval Matrix")
    table.add_column("Retriever")
    table.add_column("Delivery")
    table.add_column("Harness")
    table.add_column("Accuracy")
    table.add_column("N")

    for c in sorted(cells, key=lambda x: x.accuracy, reverse=True):
        table.add_row(
            c.retriever.value,
            c.delivery.value,
            c.harness.value,
            f"{c.accuracy:.1%}",
            str(c.n),
        )
    console.print(table)