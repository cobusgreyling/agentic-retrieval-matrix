from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from agentic_retrieval_matrix.memory_store import corpus_to_memory_files
from agentic_retrieval_matrix.retrievers.base import Retriever
from agentic_retrieval_matrix.types import Hit, MemoryCorpus, RetrievalResult, RunConfig


def _keywords(query: str, max_terms: int = 6) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]{3,}", query.lower())
    stop = {
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
    }
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in stop or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out or [query.strip()[:32]]


class GrepRetriever(Retriever):
    kind = "grep"

    def __init__(self) -> None:
        self._session_dir: Path | None = None
        self._corpus: MemoryCorpus | None = None

    def index(self, corpus: MemoryCorpus, memory_root: Path) -> None:
        self._corpus = corpus
        self._session_dir = corpus_to_memory_files(corpus, memory_root)

    def search(self, query: str, config: RunConfig) -> RetrievalResult:
        if self._session_dir is None or self._corpus is None:
            raise RuntimeError("GrepRetriever.index() must be called before search()")

        rg = shutil.which(config.ripgrep_bin)
        hits: list[Hit] = []
        terms = _keywords(query)

        if rg:
            pattern = "|".join(re.escape(t) for t in terms)
            proc = subprocess.run(
                [rg, "-i", "-n", "--no-heading", "-m", "3", pattern, str(self._session_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            hits = self._hits_from_rg(proc.stdout)
        else:
            hits = self._fallback_python_search(terms)

        hits.sort(key=lambda h: h.score, reverse=True)
        return RetrievalResult(
            query=query,
            hits=hits[: config.top_k],
            metadata={"terms": terms, "backend": "ripgrep" if rg else "python"},
        )

    def _hits_from_rg(self, stdout: str) -> list[Hit]:
        hits: list[Hit] = []
        for line in stdout.splitlines():
            # path:line:col:text or path:line:text
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            path_str, _line_no, snippet = parts[0], parts[1], parts[2]
            turn_id = self._turn_id_from_path(path_str)
            hits.append(
                Hit(
                    turn_id=turn_id,
                    session_id=self._corpus.session_id if self._corpus else "unknown",
                    snippet=snippet.strip()[:400],
                    score=1.0,
                    source="ripgrep",
                )
            )
        return hits

    def _fallback_python_search(self, terms: list[str]) -> list[Hit]:
        assert self._session_dir is not None and self._corpus is not None
        hits: list[Hit] = []
        for turn in self._corpus.turns:
            lower = turn.content.lower()
            score = sum(1 for t in terms if t in lower)
            if score:
                hits.append(
                    Hit(
                        turn_id=turn.turn_id,
                        session_id=turn.session_id,
                        snippet=turn.content[:400],
                        score=float(score),
                        source="python-grep",
                    )
                )
        return hits

    @staticmethod
    def _turn_id_from_path(path_str: str) -> int:
        name = Path(path_str).name
        if name.startswith("turn_") and name.endswith(".txt"):
            try:
                return int(name[5:9])
            except ValueError:
                pass
        return 0
