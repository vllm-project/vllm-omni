"""
Engine components for vLLM-Omni.
"""

import msgspec
import torch
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
)

from vllm_omni.data_entry_keys import OmniInputStruct, OmniPayloadStruct


class PromptEmbedsPayload(msgspec.Struct):
    """Serialized prompt embeddings payload for direct transfer.

    data: raw bytes of the tensor in row-major order
    shape: [seq_len, hidden_size]
    dtype: torch dtype name (e.g., "float16", "float32")
    """

    data: bytes
    shape: list[int]
    dtype: str


# Tagged-union envelope for OmniEngineCoreRequest.additional_information.
# msgspec dispatches between OmniInputStruct (request controls) and
# OmniPayloadStruct (stage outputs) via class-name tag.
AdditionalInformationField = OmniInputStruct | OmniPayloadStruct | None


class OmniEngineCoreRequest(EngineCoreRequest):
    """Engine core request for omni models with embeddings support.

    Extends the base EngineCoreRequest with support for additional
    information payloads, enabling direct transfer of pre-computed data
    between pipeline stages.

    Note: prompt_embeds is inherited from EngineCoreRequest
    (torch.Tensor | None). PromptEmbedsPayload should be decoded to
    torch.Tensor before constructing this request.

    Attributes:
        additional_information: Optional typed payload — either an
            ``OmniInputStruct`` (request-side controls) or an
            ``OmniPayloadStruct`` (stage-to-stage outputs).
    """

    additional_information: AdditionalInformationField = None

    @classmethod
    def from_request(
        cls,
        request: EngineCoreRequest,
        *,
        prompt_embeds: torch.Tensor | None = None,
        additional_information: AdditionalInformationField = None,
    ) -> "OmniEngineCoreRequest":
        """Clone an EngineCoreRequest into an OmniEngineCoreRequest with optional payload overrides."""

        if prompt_embeds is None:
            prompt_embeds = request.prompt_embeds
        if additional_information is None:
            additional_information = getattr(request, "additional_information", None)

        return cls(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            data_parallel_rank=request.data_parallel_rank,
            prompt_embeds=prompt_embeds,
            client_index=request.client_index,
            current_wave=request.current_wave,
            priority=request.priority,
            trace_headers=request.trace_headers,
            resumable=request.resumable,
            external_req_id=request.external_req_id,
            reasoning_ended=request.reasoning_ended,
            additional_information=additional_information,
        )


class OmniEngineCoreOutput(EngineCoreOutput):
    pooling_output: dict[str, torch.Tensor] | None = None
    # Finished flag for streaming input segment
    is_segment_finished: bool | None = False
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None


class OmniEngineCoreOutputs(EngineCoreOutputs):
    outputs: list[OmniEngineCoreOutput] = []
