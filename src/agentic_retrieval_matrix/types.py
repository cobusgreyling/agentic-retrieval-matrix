from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RetrieverKind(StrEnum):
    GREP = "grep"
    VECTOR = "vector"


class DeliveryKind(StrEnum):
    INLINE = "inline"
    FILE = "file"


class HarnessKind(StrEnum):
    REACT = "react"


class Turn(BaseModel):
    role: str
    content: str
    session_id: str = "default"
    turn_id: int = 0


class MemoryCorpus(BaseModel):
    """Conversation history indexed for retrieval."""

    session_id: str
    turns: list[Turn]

    def memory_dir_name(self) -> str:
        return self.session_id.replace("/", "_")


class Question(BaseModel):
    id: str
    text: str
    category: str = "unknown"
    gold_answer: str
    session_id: str


class Hit(BaseModel):
    turn_id: int
    session_id: str
    snippet: str
    score: float = 1.0
    source: str = "grep"


class RetrievalResult(BaseModel):
    query: str
    hits: list[Hit] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentAnswer(BaseModel):
    question_id: str
    answer: str
    evidence: list[Hit] = Field(default_factory=list)
    steps: int = 0


class MatrixCell(BaseModel):
    retriever: RetrieverKind
    delivery: DeliveryKind
    harness: HarnessKind
    accuracy: float
    n: int
    details: list[dict[str, Any]] = Field(default_factory=list)


class RunConfig(BaseModel):
    retriever: RetrieverKind = RetrieverKind.GREP
    delivery: DeliveryKind = DeliveryKind.INLINE
    harness: HarnessKind = HarnessKind.REACT
    max_steps: int = 3
    top_k: int = 5
    work_dir: str = ".arm_runs"
    ripgrep_bin: str = "rg"
