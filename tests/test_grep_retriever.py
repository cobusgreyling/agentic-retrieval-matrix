from pathlib import Path

from agentic_retrieval_matrix.memory_store import load_corpus_from_json
from agentic_retrieval_matrix.retrievers.grep import GrepRetriever
from agentic_retrieval_matrix.types import RunConfig

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "fixture"


def test_grep_finds_laptop_preference():
    corpus = load_corpus_from_json(FIXTURE / "session_alice.json")
    retriever = GrepRetriever()
    retriever.index(corpus, Path("/tmp/arm_test_memory"))
    result = retriever.search("What laptop model does the user prefer?", RunConfig())
    assert result.hits
    joined = " ".join(h.snippet for h in result.hits).lower()
    assert "thinkpad" in joined