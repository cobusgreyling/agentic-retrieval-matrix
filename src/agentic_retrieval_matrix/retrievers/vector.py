from __future__ import annotations

from pathlib import Path

from agentic_retrieval_matrix.memory_store import corpus_to_memory_files
from agentic_retrieval_matrix.retrievers.base import Retriever
from agentic_retrieval_matrix.types import Hit, MemoryCorpus, RetrievalResult, RunConfig


class VectorRetriever(Retriever):
    """Dense retrieval over turn text. Requires optional `vector` extras."""

    kind = "vector"

    def __init__(self) -> None:
        self._corpus: MemoryCorpus | None = None
        self._texts: list[str] = []
        self._index = None
        self._model = None

    def index(self, corpus: MemoryCorpus, memory_root: Path) -> None:
        corpus_to_memory_files(corpus, memory_root)
        self._corpus = corpus
        self._texts = [t.content for t in corpus.turns]
        try:
            import faiss  # type: ignore
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Vector retriever requires: pip install 'agentic-retrieval-matrix[vector]'"
            ) from exc

        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = self._model.encode(self._texts, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(np.asarray(embeddings, dtype="float32"))

    def search(self, query: str, config: RunConfig) -> RetrievalResult:
        if self._corpus is None or self._index is None or self._model is None:
            raise RuntimeError("VectorRetriever.index() must be called before search()")

        import numpy as np

        q = self._model.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(np.asarray(q, dtype="float32"), config.top_k)
        hits: list[Hit] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist(), strict=True):
            if idx < 0:
                continue
            turn = self._corpus.turns[idx]
            hits.append(
                Hit(
                    turn_id=turn.turn_id,
                    session_id=turn.session_id,
                    snippet=turn.content[:400],
                    score=float(score),
                    source="vector",
                )
            )
        return RetrievalResult(query=query, hits=hits, metadata={"model": "all-MiniLM-L6-v2"})
