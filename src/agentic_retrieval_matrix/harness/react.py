from __future__ import annotations

from pathlib import Path

from agentic_retrieval_matrix.harness.answer import compose_answer
from agentic_retrieval_matrix.harness.pipeline import run_retrieval_pipeline
from agentic_retrieval_matrix.types import AgentAnswer, MemoryCorpus, Question, RunConfig


class ReactHarness:
    """
    Minimal ReAct-style loop without an LLM provider dependency.

    Step 1: search memory with configured retriever.
    Step 2: deliver results (inline or file).
    Step 3 (file only): simulated read_file → load snippets from JSON.
    Step 4: blind or oracle answer from delivery-appropriate context only.
    """

    def run(
        self,
        question: Question,
        corpus: MemoryCorpus,
        config: RunConfig,
        memory_root: Path,
    ) -> AgentAnswer:
        pipeline = run_retrieval_pipeline(question, corpus, config, memory_root)
        answer = compose_answer(question, pipeline.candidates, config.answer_mode)
        return AgentAnswer(
            question_id=question.id,
            answer=answer,
            evidence=pipeline.evidence,
            steps=pipeline.steps + 1,
        )
