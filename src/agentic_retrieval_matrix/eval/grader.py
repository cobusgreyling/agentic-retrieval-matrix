from __future__ import annotations

import re

from agentic_retrieval_matrix.types import AgentAnswer, Question


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def grade_answer(question: Question, answer: AgentAnswer) -> bool:
    """
    Lightweight grader aligned with literal-span QA (LongMemEval-style).
    For production runs, swap with an LLM grader via --llm-grader.
    """
    pred = normalize(answer.answer)
    gold = normalize(question.gold_answer)
    if not gold:
        return False
    if gold in pred or pred in gold:
        return True
    pred_tokens = set(gold.split())
    if not pred_tokens:
        return False
    overlap = len(pred_tokens & set(pred.split())) / len(pred_tokens)
    return overlap >= 0.6