from pathlib import Path

from agentic_retrieval_matrix.delivery.file import FileDelivery
from agentic_retrieval_matrix.delivery.inline import InlineDelivery
from agentic_retrieval_matrix.types import Hit, RetrievalResult


def test_file_delivery_writes_json():
    delivery = FileDelivery()
    result = RetrievalResult(
        query="test",
        hits=[Hit(turn_id=1, session_id="s", snippet="hello", score=1.0)],
    )
    run_dir = Path("/tmp/arm_delivery_test")
    msg = delivery.present(result, run_dir)
    assert "retrieval_results.json" in msg
    assert (run_dir / "retrieval_results.json").exists()


def test_inline_delivery_includes_snippet():
    delivery = InlineDelivery()
    result = RetrievalResult(
        query="test",
        hits=[Hit(turn_id=1, session_id="s", snippet="hello", score=1.0)],
    )
    msg = delivery.present(result, Path("/tmp/arm_inline"))
    assert "hello" in msg