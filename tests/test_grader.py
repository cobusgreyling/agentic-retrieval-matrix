from agentic_retrieval_matrix.eval.grader import grade_answer, normalize
from agentic_retrieval_matrix.types import AgentAnswer, Question


def test_normalize_basic():
    assert normalize("  Hello,  World!  ") == "hello world"
    assert normalize("Test-123") == "test123"
    assert normalize("") == ""


def test_grade_exact_match():
    q = Question(id="q", text="x", gold_answer="ThinkPad X1 Carbon", session_id="s")
    a = AgentAnswer(question_id="q", answer="ThinkPad X1 Carbon")
    assert grade_answer(q, a) is True


def test_grade_contains():
    q = Question(id="q", text="x", gold_answer="March 2024", session_id="s")
    a = AgentAnswer(question_id="q", answer="The renewal happened in March 2024.")
    assert grade_answer(q, a) is True


def test_grade_token_overlap():
    q = Question(id="q", text="x", gold_answer="backup API key prefix", session_id="s")
    a = AgentAnswer(question_id="q", answer="stored backup api key prefix sklive")
    assert grade_answer(q, a) is True  # multi-token overlap >= 0.6


def test_grade_fail_low_overlap():
    q = Question(id="q", text="x", gold_answer="ThinkPad X1 Carbon", session_id="s")
    a = AgentAnswer(question_id="q", answer="completely unrelated answer here")
    assert grade_answer(q, a) is False


def test_grade_empty_gold():
    q = Question(id="q", text="x", gold_answer="", session_id="s")
    a = AgentAnswer(question_id="q", answer="something")
    assert grade_answer(q, a) is False
