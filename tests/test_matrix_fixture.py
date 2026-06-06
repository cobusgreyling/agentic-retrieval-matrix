from pathlib import Path

from agentic_retrieval_matrix.eval.loader import load_benchmark
from agentic_retrieval_matrix.eval.matrix import run_matrix, save_results
from agentic_retrieval_matrix.types import DeliveryKind, HarnessKind, RetrieverKind

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "fixture"


def test_grep_full_matrix_on_fixture(tmp_path: Path):
    questions, corpora = load_benchmark(FIXTURE / "benchmark.json")
    cells = run_matrix(
        questions,
        corpora,
        [RetrieverKind.GREP],
        [DeliveryKind.INLINE, DeliveryKind.FILE],
        [HarnessKind.REACT],
        work_dir=tmp_path / "mem",
    )
    assert len(cells) == 2
    for c in cells:
        assert c.n == 3
        assert 0.0 <= c.accuracy <= 1.0
        assert c.retriever == RetrieverKind.GREP
        assert len(c.details) == 3
        # Current baseline extractive harness achieves high accuracy on this fixture
        assert c.accuracy >= 0.66

    # roundtrip save
    out = tmp_path / "results"
    path = save_results(cells, out)
    assert path.exists()
    assert "matrix_" in path.name
