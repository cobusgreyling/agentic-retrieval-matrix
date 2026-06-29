from __future__ import annotations

import json
from pathlib import Path

from agentic_retrieval_matrix.delivery import build_delivery
from agentic_retrieval_matrix.harness.answer import SnippetCandidate, parse_inline_snippets
from agentic_retrieval_matrix.retrievers import build_retriever
from agentic_retrieval_matrix.types import DeliveryKind, Hit, MemoryCorpus, Question, RunConfig


class RetrievalPipelineResult:
    __slots__ = ("candidates", "evidence", "steps", "presented", "result_hits")

    def __init__(
        self,
        candidates: list[SnippetCandidate],
        evidence: list[Hit],
        steps: int,
        presented: str,
        result_hits: list[Hit],
    ) -> None:
        self.candidates = candidates
        self.evidence = evidence
        self.steps = steps
        self.presented = presented
        self.result_hits = result_hits


def run_retrieval_pipeline(
    question: Question,
    corpus: MemoryCorpus,
    config: RunConfig,
    memory_root: Path,
) -> RetrievalPipelineResult:
    """Shared retriever → delivery → snippet extraction used by all harnesses."""
    retriever = build_retriever(config.retriever)
    delivery = build_delivery(config.delivery)
    run_dir = (
        Path(config.work_dir) / question.id / f"{config.retriever.value}_{config.delivery.value}"
    )

    retriever.index(corpus, memory_root)
    result = retriever.search(question.text, config)
    presented = delivery.present(result, run_dir)

    evidence: list[Hit] = []
    steps = 1

    if config.delivery == DeliveryKind.INLINE:
        candidates = parse_inline_snippets(presented)
        evidence = list(result.hits)
    else:
        steps += 1
        evidence = _read_file_evidence(run_dir / "retrieval_results.json")
        candidates = [
            SnippetCandidate(text=h.snippet, retrieval_score=h.score, turn_id=h.turn_id)
            for h in evidence[:1]
        ]

    return RetrievalPipelineResult(
        candidates=candidates,
        evidence=evidence,
        steps=steps,
        presented=presented,
        result_hits=list(result.hits),
    )


def _read_file_evidence(path: Path) -> list[Hit]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Hit(**h) for h in data.get("hits", [])]
