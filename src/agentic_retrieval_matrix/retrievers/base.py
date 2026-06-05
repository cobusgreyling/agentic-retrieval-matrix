from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from agentic_retrieval_matrix.types import MemoryCorpus, RetrievalResult, RunConfig


class Retriever(ABC):
    kind: str

    @abstractmethod
    def index(self, corpus: MemoryCorpus, memory_root: Path) -> None:
        """Prepare retrieval structures for a corpus."""

    @abstractmethod
    def search(self, query: str, config: RunConfig) -> RetrievalResult:
        """Return ranked evidence hits for the query."""