from agentic_retrieval_matrix.delivery.file import FileDelivery
from agentic_retrieval_matrix.delivery.inline import InlineDelivery
from agentic_retrieval_matrix.types import DeliveryKind

DELIVERY_REGISTRY = {
    DeliveryKind.INLINE: InlineDelivery,
    DeliveryKind.FILE: FileDelivery,
}


def build_delivery(kind: DeliveryKind) -> InlineDelivery | FileDelivery:
    return DELIVERY_REGISTRY[kind]()