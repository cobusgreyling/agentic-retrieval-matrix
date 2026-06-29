from agentic_retrieval_matrix.harness.answer import (
    SnippetCandidate,
    blind_answer,
    parse_inline_snippets,
)
from agentic_retrieval_matrix.types import Question


def test_parse_inline_snippets():
    presented = (
        "[retrieval] query='test'\n"
        "backend=ripgrep\n"
        "\n--- hit 1 (turn=1, score=1.000, source=ripgrep) ---\n"
        "alpha snippet\n"
        "\n--- hit 2 (turn=2, score=0.500, source=ripgrep) ---\n"
        "beta snippet"
    )
    parsed = parse_inline_snippets(presented)
    assert [c.text for c in parsed] == ["alpha snippet", "beta snippet"]
    assert parsed[1].turn_id == 2


def test_blind_answer_picks_best_keyword_overlap():
    question = Question(
        id="q",
        text="What is the user's backup API key prefix?",
        gold_answer="sk-live-9f2",
        session_id="s",
    )
    candidates = [
        SnippetCandidate(
            "Please store my backup API key prefix sk-decoy-aaa in the draft vault.",
            turn_id=0,
        ),
        SnippetCandidate(
            "Please store my backup API key prefix sk-live-9f2 in the vault notes.",
            turn_id=2,
        ),
    ]
    answer = blind_answer(question, candidates)
    assert "sk-live-9f2" in answer
