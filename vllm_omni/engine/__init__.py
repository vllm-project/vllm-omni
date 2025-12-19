"""
Engine components for vLLM-Omni.
"""

import time
from collections.abc import Mapping
from typing import Any

import msgspec
import torch
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreRequest,
    FinishReason,
    LogprobsLists,
    LogprobsTensors,
    SchedulerStats,
    UtilityOutput,
)


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
    tensor_data: bytes | None = None
    tensor_shape: list[int] | None = None
    tensor_dtype: str | None = None

    # List form
    list_data: list[Any] | None = None


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
    prompt_embeds: PromptEmbedsPayload | None = None
    # Optional additional information dictionary (serialized)
    additional_information: AdditionalInformationPayload | None = None


class OmniEngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    new_token_ids: list[int]

    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    pooling_output: dict[str, torch.Tensor] | None = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None
    # The number of tokens with prefix cache hits.
    num_cached_tokens: int = 0

    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted.
    num_nans_in_logits: int = 0

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class OmniEngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0

    # [num_reqs]
    outputs: list[OmniEngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0

    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: int | None = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: int | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()
