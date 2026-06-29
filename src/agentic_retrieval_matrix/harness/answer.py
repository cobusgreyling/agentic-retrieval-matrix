from __future__ import annotations

import re

from agentic_retrieval_matrix.types import AnswerMode, Question

_HIT_BLOCK = re.compile(
    r"--- hit \d+ \(turn=(\d+), score=([\d.]+), .*?\) ---\n(.*?)(?=\n--- hit |\Z)",
    re.DOTALL,
)

_KEYWORD_STOP = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "does",
        "did",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "have",
        "has",
        "was",
        "were",
        "user",
        "about",
        "currently",
        "their",
    }
)


def question_keywords(query: str, max_terms: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]{3,}", query.lower())
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token in _KEYWORD_STOP or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= max_terms:
            break
    return out


class SnippetCandidate:
    __slots__ = ("text", "retrieval_score", "turn_id")

    def __init__(self, text: str, retrieval_score: float = 0.0, turn_id: int = 0) -> None:
        self.text = text
        self.retrieval_score = retrieval_score
        self.turn_id = turn_id


def parse_inline_snippets(presented: str) -> list[SnippetCandidate]:
    """Extract hit snippets from inline delivery tool output."""
    out: list[SnippetCandidate] = []
    for match in _HIT_BLOCK.finditer(presented):
        text = match.group(3).strip()
        if text:
            out.append(
                SnippetCandidate(
                    text=text,
                    retrieval_score=float(match.group(2)),
                    turn_id=int(match.group(1)),
                )
            )
    return out


def blind_answer(question: Question, candidates: list[SnippetCandidate]) -> str:
    """Answer using question text only — no access to gold labels."""
    if not candidates:
        return "I could not find relevant information in memory."

    keywords = question_keywords(question.text)
    best = candidates[0]
    best_score = -1.0
    for cand in candidates:
        lower = cand.text.lower()
        kw_score = float(sum(1 for kw in keywords if kw in lower))
        combined = kw_score + 0.05 * cand.retrieval_score
        if combined > best_score or (combined == best_score and cand.turn_id > best.turn_id):
            best_score = combined
            best = cand
    return best.text.strip()[:280]


def oracle_answer(question: Question, candidates: list[SnippetCandidate]) -> str:
    """Debug-only answerer that may use gold labels (not valid for benchmarking)."""
    if not candidates:
        return "I could not find relevant information in memory."

    gold_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", question.gold_answer.lower()))
    best = candidates[0]
    best_score = -1.0
    for cand in candidates:
        snippet_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", cand.text.lower()))
        overlap = len(gold_tokens & snippet_tokens)
        if overlap > best_score:
            best_score = overlap
            best = cand

    if len(question.gold_answer) < 80:
        for token in gold_tokens:
            if token in best.text.lower():
                return question.gold_answer

    return best.text.strip()[:280]


def compose_answer(question: Question, candidates: list[SnippetCandidate], mode: AnswerMode) -> str:
    if mode == AnswerMode.ORACLE:
        return oracle_answer(question, candidates)
    return blind_answer(question, candidates)
