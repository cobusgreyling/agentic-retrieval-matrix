from agentic_retrieval_matrix.retrievers.grep import GrepRetriever
from agentic_retrieval_matrix.retrievers.vector import VectorRetriever
from agentic_retrieval_matrix.types import RetrieverKind

RETRIEVER_REGISTRY = {
    RetrieverKind.GREP: GrepRetriever,
    RetrieverKind.VECTOR: VectorRetriever,
}


def build_retriever(kind: RetrieverKind):
    return RETRIEVER_REGISTRY[kind]()