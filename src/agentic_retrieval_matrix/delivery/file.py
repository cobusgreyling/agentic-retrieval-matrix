from __future__ import annotations

import json
from pathlib import Path

from agentic_retrieval_matrix.delivery.base import DeliveryChannel
from agentic_retrieval_matrix.types import DeliveryKind, RetrievalResult


class FileDelivery(DeliveryChannel):
    """Write hits to disk; agent must read the path (paper: programmatic/file-based)."""

    kind = DeliveryKind.FILE

    def present(self, result: RetrievalResult, run_dir: Path) -> str:
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "retrieval_results.json"
        payload = {
            "query": result.query,
            "metadata": result.metadata,
            "hits": [h.model_dump() for h in result.hits],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if not result.hits:
            return (
                f"Retrieval complete. Results written to: {out_path}\n"
                "Use read_file to inspect. No hits were found."
            )
        return (
            f"Retrieval complete. Results written to: {out_path}\n"
            "You MUST call read_file on this path before answering.\n"
            f"Hit count: {len(result.hits)}"
        )
