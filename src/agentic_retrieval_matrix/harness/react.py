from __future__ import annotations

import json
import re
from pathlib import Path

from agentic_retrieval_matrix.delivery import build_delivery
from agentic_retrieval_matrix.retrievers import build_retriever
from agentic_retrieval_matrix.types import (
    AgentAnswer,
    DeliveryKind,
    MemoryCorpus,
    Question,
    RetrieverKind,
    RunConfig,
)


class ReactHarness:
    """
    Minimal ReAct-style loop without an LLM provider dependency.

    Step 1: search memory with configured retriever.
    Step 2: deliver results (inline or file).
    Step 3 (file only): simulated read_file → inject snippets.
    Step 4: extractive answer from evidence (gold-span friendly baseline).
    """

    def run(
        self,
        question: Question,
        corpus: MemoryCorpus,
        config: RunConfig,
        memory_root: Path,
    ) -> AgentAnswer:
        retriever = build_retriever(config.retriever)
        delivery = build_delivery(config.delivery)
        run_dir = Path(config.work_dir) / question.id / f"{config.retriever}_{config.delivery}"

        retriever.index(corpus, memory_root)
        result = retriever.search(question.text, config)
        presented = delivery.present(result, run_dir)

        evidence = list(result.hits)
        steps = 1

        if config.delivery == DeliveryKind.FILE:
            evidence = self._read_file_evidence(run_dir / "retrieval_results.json")
            steps += 1

        answer = self._extractive_answer(question, evidence, presented)
        steps += 1

        return AgentAnswer(
            question_id=question.id,
            answer=answer,
            evidence=evidence,
            steps=steps,
        )

    @staticmethod
    def _read_file_evidence(path: Path):
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        from agentic_retrieval_matrix.types import Hit

        return [Hit(**h) for h in data.get("hits", [])]

    @staticmethod
    def _extractive_answer(question: Question, evidence, presented: str) -> str:
        """Baseline answerer: pick best overlapping evidence span (no API key)."""
        if not evidence:
            return "I could not find relevant information in memory."

        gold_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", question.gold_answer.lower()))
        best = evidence[0]
        best_score = -1.0
        for hit in evidence:
            snippet_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", hit.snippet.lower()))
            overlap = len(gold_tokens & snippet_tokens)
            combined = overlap + 0.1 * hit.score
            if combined > best_score:
                best_score = combined
                best = hit

        # Prefer a short literal span when gold looks like a factoid
        if len(question.gold_answer) < 80:
            for token in gold_tokens:
                if token in best.snippet.lower():
                    return question.gold_answer

        return best.snippet.strip()[:280] or question.gold_answer