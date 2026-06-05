from pathlib import Path

from agentic_retrieval_matrix.eval.loader import load_benchmark
from agentic_retrieval_matrix.eval.matrix import run_matrix
from agentic_retrieval_matrix.types import DeliveryKind, HarnessKind, RetrieverKind

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "fixture"


def test_grep_inline_beats_empty_on_fixture():
    questions, corpora = load_benchmark(FIXTURE / "benchmark.json")
    cells = run_matrix(
        questions,
        corpora,
        [RetrieverKind.GREP],
        [DeliveryKind.INLINE],
        [HarnessKind.REACT],
        work_dir=Path("/tmp/arm_matrix_test"),
    )
    assert len(cells) == 1
    assert cells[0].accuracy >= 0.66