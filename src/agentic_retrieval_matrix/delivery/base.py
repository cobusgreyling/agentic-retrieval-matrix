from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from agentic_retrieval_matrix.types import DeliveryKind, RetrievalResult


class DeliveryChannel(ABC):
    kind: DeliveryKind

    @abstractmethod
    def present(self, result: RetrievalResult, run_dir: Path) -> str:
        """Format retrieval output for the agent harness."""