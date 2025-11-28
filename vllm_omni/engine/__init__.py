"""
Engine components for vLLM-omni.
"""

from typing import Any, Optional

import msgspec
from vllm.v1.engine import EngineCoreRequest


class PromptEmbedsPayload(msgspec.Struct):
    """Serialized prompt embeddings payload for direct transfer.

    data: raw bytes of the tensor in row-major order
    shape: [seq_len, hidden_size]
    dtype: torch dtype name (e.g., "float16", "float32")
    """

    data: bytes
    shape: list[int]
    dtype: str


class AdditionalInformationEntry(msgspec.Struct):
    """One entry of additional_information.

    Two supported forms are encoded:
      - tensor: data/shape/dtype
      - list: a Python list (msgspec-serializable)
    Exactly one of (tensor_data, list_data) should be non-None.
    """

    # Tensor form
    tensor_data: Optional[bytes] = None
    tensor_shape: Optional[list[int]] = None
    tensor_dtype: Optional[str] = None

    # List form
    list_data: Optional[list[Any]] = None


class AdditionalInformationPayload(msgspec.Struct):
    """Serialized dictionary payload for additional_information.

    Keys are strings; values are encoded as AdditionalInformationEntry.
    """

    entries: dict[str, AdditionalInformationEntry]


class OmniEngineCoreRequest(EngineCoreRequest):
    """Engine core request for omni models with embeddings support.

    Extends the base EngineCoreRequest with support for prompt embeddings
    and additional information payloads, enabling direct transfer of
    pre-computed embeddings between pipeline stages.

    Attributes:
        prompt_embeds: Optional serialized prompt embeddings payload for
            direct transfer between stages
        additional_information: Optional serialized additional information
            dictionary containing tensors or lists to pass along with the request
    """

    # Optional prompt embeddings (direct-transfer version)
    prompt_embeds: Optional[PromptEmbedsPayload] = None
    # Optional additional information dictionary (serialized)
    additional_information: Optional[AdditionalInformationPayload] = None
