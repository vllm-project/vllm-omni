# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, field, fields
from typing import Any

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request

from vllm_omni.core.prefix_cache.adapter import PrefixCacheRequestEvent, PrefixCacheRequestOwner
from vllm_omni.engine import AdditionalInformationPayload


@dataclass
class OmniNewRequestData(NewRequestData):
    """New request data for omni models with embeddings support.

    Extends NewRequestData to include additional information for direct
    transfer between pipeline stages.

    Note: prompt_embeds is inherited from NewRequestData
    (torch.Tensor | None).

    Args:
        external_req_id: Optional external request ID for tracking
        additional_information: Optional serialized or materialized additional information
            dictionary containing tensors or lists
        model_intermediate_buffer: Optional runner-owned payload for
            GPUModelRunner.model_intermediate_buffer
    """

    external_req_id: str | None = None
    additional_information: AdditionalInformationPayload | dict[str, object] | None = None
    model_intermediate_buffer: dict[str, object] | None = None

    @classmethod
    def from_base(
        cls,
        data: NewRequestData,
        request: Request | None,
    ) -> "OmniNewRequestData":
        """Preserve upstream request data while attaching Omni payloads."""
        base_data = {field.name: getattr(data, field.name) for field in fields(NewRequestData)}
        return cls(
            **base_data,
            external_req_id=getattr(request, "external_req_id", None),
            additional_information=getattr(request, "additional_information", None),
            model_intermediate_buffer=getattr(request, "model_intermediate_buffer", None),
        )

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "OmniNewRequestData":
        """Create OmniNewRequestData from a Request object.

        Args:
            request: Request object to convert
            block_ids: Tuple of block ID lists for KV cache allocation
            prefill_token_ids: Optional prefill token IDs for v2 model runner

        Returns:
            OmniNewRequestData instance with data from the request
        """
        return cls(
            req_id=request.request_id,
            external_req_id=getattr(request, "external_req_id", None),
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            prompt_is_token_ids=getattr(request, "prompt_is_token_ids", None),
            prefill_token_ids=prefill_token_ids,
            additional_information=getattr(request, "additional_information", None),
            model_intermediate_buffer=getattr(request, "model_intermediate_buffer", None),
        )


@dataclass
class OmniCachedRequestData(CachedRequestData):
    """Cached request data for omni models with embeddings support.

    Args:
        prompt_token_ids: Mapping from request ID to list of prompt token IDs
    """

    prompt_token_ids: dict[str, list[int]]
    additional_information: dict[str, dict | None]


@dataclass
class OmniChunkRecvHandle:
    """Minimal identifier carried from scheduler to runner for input-receive
    registration.

    Carries routing fields and the receiving scheduler's immutable content
    owner, not the full Request. Concrete typing keeps msgspec serialization
    deterministic across IPC (default, PD-disagg, multi-node executor variants)
    and avoids the ``list[Any]`` fallback path. The owner is echoed in receive
    notifications; it is not supplied by the remote producer's payload.
    """

    request_id: str
    external_req_id: str | None = None
    payload_sender_info: dict[str, object] | None = None
    input_owner: PrefixCacheRequestOwner | None = None


@dataclass
class OmniRequestPrewarm:
    """Warm-up payload for an async-chunk placeholder, handed to the runner once.

    The orchestrator attaches it to a downstream stage's prewarm placeholder so
    the model can prepare per-request state before the first chunk arrives.
    ``payload`` is the ``ASYNC_CHUNK_PREWARM_NS`` namespace of the placeholder's
    additional_information, deserialized (e.g. ``{"ref_audio": Tensor,
    "ref_audio_sr": int}``).
    """

    request_id: str
    payload: dict[str, Any]


@dataclass
class OmniSchedulerOutput(SchedulerOutput):
    """Scheduler output with omni-specific transfer metadata."""

    finished_requests_needing_kv_transfer: dict[str, dict] = field(default_factory=dict)
    pending_input_registrations: list[OmniChunkRecvHandle] = field(default_factory=list)
    data_plane_terminal_req_ids: set[str] = field(default_factory=set)
    input_terminal_req_ids: set[str] = field(default_factory=set)
    prefix_cache_replacements: tuple[PrefixCacheRequestEvent, ...] = ()
    prefix_cache_owners: dict[str, PrefixCacheRequestOwner] = field(default_factory=dict)
    prefix_cache_terminal_owners: dict[str, PrefixCacheRequestOwner] = field(default_factory=dict)
    prefix_cache_step_sequence: int | None = None
    pending_request_prewarms: list[OmniRequestPrewarm] = field(default_factory=list)
