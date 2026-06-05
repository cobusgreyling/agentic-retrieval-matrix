from __future__ import annotations

import json
from pathlib import Path

from agentic_retrieval_matrix.memory_store import load_corpus_from_json
from agentic_retrieval_matrix.types import MemoryCorpus, Question


def load_benchmark(path: Path) -> tuple[list[Question], dict[str, MemoryCorpus]]:
    """
    Load a benchmark bundle:
      benchmark.json  -> { "questions": [...], "corpora": {"session_id": path} }
    Each corpus path points to a corpus JSON file.
    """
    spec = json.loads(path.read_text(encoding="utf-8"))
    corpora: dict[str, MemoryCorpus] = {}
    for session_id, corpus_path in spec["corpora"].items():
        corpora[session_id] = load_corpus_from_json(path.parent / corpus_path)

    questions = [Question(**q) for q in spec["questions"]]
    return questions, corpora


def load_longmemeval_subset(
    data_dir: Path,
    limit: int | None = None,
) -> tuple[list[Question], dict[str, MemoryCorpus]]:
    """
    Adapter hook for LongMemEval exports.

    Expects `data_dir/questions.jsonl` and `data_dir/sessions/<id>.json`
    in a thin normalized format (see README). Falls back to raising with
    instructions when files are missing.
    """
    questions_path = data_dir / "questions.jsonl"
    sessions_dir = data_dir / "sessions"
    if not questions_path.exists():
        raise FileNotFoundError(
            f"LongMemEval subset not found at {data_dir}. "
            "See README: 'Connecting LongMemEval' for conversion steps."
        )

    questions: list[Question] = []
    corpora: dict[str, MemoryCorpus] = {}

    for line in questions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(
            Question(
                id=row["id"],
                text=row["question"],
                category=row.get("category", "unknown"),
                gold_answer=row["answer"],
                session_id=row["session_id"],
            )
        )
        if limit and len(questions) >= limit:
            break

    for q in questions:
        if q.session_id in corpora:
            continue
        session_path = sessions_dir / f"{q.session_id}.json"
        corpora[q.session_id] = load_corpus_from_json(session_path)

    return questions, corpora