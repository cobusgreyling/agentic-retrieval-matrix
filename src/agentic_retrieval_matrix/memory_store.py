from __future__ import annotations

import json
from pathlib import Path

from agentic_retrieval_matrix.types import MemoryCorpus, Turn


def corpus_to_memory_files(corpus: MemoryCorpus, root: Path) -> Path:
    """Materialize turns as grep-friendly text files (one file per turn)."""
    session_dir = root / corpus.memory_dir_name()
    session_dir.mkdir(parents=True, exist_ok=True)
    for turn in corpus.turns:
        path = session_dir / f"turn_{turn.turn_id:04d}.txt"
        path.write_text(
            f"role: {turn.role}\nsession: {turn.session_id}\n\n{turn.content}\n",
            encoding="utf-8",
        )
    manifest = {
        "session_id": corpus.session_id,
        "turn_count": len(corpus.turns),
        "turn_files": [f"turn_{t.turn_id:04d}.txt" for t in corpus.turns],
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return session_dir


def load_corpus_from_json(path: Path) -> MemoryCorpus:
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = [Turn(**t) for t in data["turns"]]
    return MemoryCorpus(session_id=data["session_id"], turns=turns)