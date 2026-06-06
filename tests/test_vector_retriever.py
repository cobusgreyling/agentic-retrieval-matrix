from pathlib import Path

import pytest

from agentic_retrieval_matrix.memory_store import load_corpus_from_json
from agentic_retrieval_matrix.retrievers.vector import VectorRetriever
from agentic_retrieval_matrix.types import RunConfig

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "fixture"


def test_vector_retriever_when_available():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")

    corpus = load_corpus_from_json(FIXTURE / "session_alice.json")
    retriever = VectorRetriever()
    retriever.index(corpus, Path("/tmp/arm_vec_test"))
    result = retriever.search("laptop prefer", RunConfig(top_k=3))
    assert len(result.hits) <= 3
    assert result.metadata.get("model")
    # At least one hit should relate to the known fact
    joined = " ".join(h.snippet.lower() for h in result.hits)
    assert "thinkpad" in joined or "x1" in joined or "carbon" in joined
