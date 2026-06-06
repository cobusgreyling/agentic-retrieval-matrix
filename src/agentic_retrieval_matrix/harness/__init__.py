from agentic_retrieval_matrix.harness.react import ReactHarness
from agentic_retrieval_matrix.types import HarnessKind

HARNESS_REGISTRY = {HarnessKind.REACT: ReactHarness}


def build_harness(kind: HarnessKind) -> ReactHarness:
    return HARNESS_REGISTRY[kind]()
