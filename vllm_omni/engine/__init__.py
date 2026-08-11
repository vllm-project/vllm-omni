"""
Engine components for vLLM-Omni.
"""

import warnings
from typing import Any

import msgspec


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

    Three supported forms are encoded:
      - tensor: data/shape/dtype
      - list: a Python list (msgspec-serializable)
      - scalar: a Python scalar (msgspec-serializable)
    Exactly one of (tensor_data, list_data, scalar_data) should be non-None.
    """

    # Tensor form
    tensor_data: bytes | None = None
    tensor_shape: list[int] | None = None
    tensor_dtype: str | None = None

    # List form
    list_data: list[Any] | None = None

    # Scalar form
    scalar_data: Any | None = None


class AdditionalInformationPayload(msgspec.Struct):
    """Serialized dictionary payload for additional_information.

    Keys are strings; values are encoded as AdditionalInformationEntry.
    """

    entries: dict[str, AdditionalInformationEntry]


# ---------------------------------------------------------------------------
# Deprecated compatibility aliases (remove no earlier than one release out)
# ---------------------------------------------------------------------------
# The ``OmniEngineCore*`` wire structs formerly defined here were renamed and
# relocated to ``vllm_omni.engine.stage.stage_core_types.StageLLMCore*``. These
# names were part of the documented API (docs/api/README.md), so we keep lazy
# deprecated aliases for at least one release to avoid breaking imports on
# upgrade. Resolved via PEP 562 ``__getattr__`` so the (circular) import of the
# stage subpackage happens only on access, after this package is initialized.
_DEPRECATED_ALIASES = {
    "OmniEngineCoreRequest": "StageLLMCoreRequest",
    "OmniEngineCoreOutput": "StageLLMCoreOutput",
    "OmniEngineCoreOutputs": "StageLLMCoreOutputs",
}


def __getattr__(name: str) -> Any:
    new_name = _DEPRECATED_ALIASES.get(name)
    if new_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from vllm_omni.engine.stage import stage_core_types

    warnings.warn(
        f"vllm_omni.engine.{name} is deprecated and will be removed in a future "
        f"release; use vllm_omni.engine.stage.stage_core_types.{new_name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(stage_core_types, new_name)


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_DEPRECATED_ALIASES.keys()])
