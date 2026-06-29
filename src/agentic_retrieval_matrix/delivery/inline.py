from __future__ import annotations

from pathlib import Path

from agentic_retrieval_matrix.delivery.base import DeliveryChannel
from agentic_retrieval_matrix.types import DeliveryKind, RetrievalResult


class InlineDelivery(DeliveryChannel):
    """Tool results injected directly into the agent context (paper: inline)."""

    kind = DeliveryKind.INLINE

    def present(self, result: RetrievalResult, run_dir: Path) -> str:
        if not result.hits:
            return f"[retrieval] query={result.query!r}\n(no hits)"

        backend = result.metadata.get("backend", "n/a")
        lines = [f"[retrieval] query={result.query!r}", f"backend={backend}"]
        for i, hit in enumerate(result.hits, start=1):
            header = (
                f"\n--- hit {i} (turn={hit.turn_id}, score={hit.score:.3f}, "
                f"source={hit.source}) ---\n"
            )
            lines.append(f"{header}{hit.snippet}")
        return "\n".join(lines)
