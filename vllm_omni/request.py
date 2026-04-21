from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_omni.engine import AdditionalInformationPayload, OmniEngineCoreRequest, PromptEmbedsPayload


class OmniRequest(Request):
    """Request class for omni models, extending the base Request.

    This class extends the base vLLM Request with support for prompt
    embeddings and additional information payloads, enabling direct
    transfer of pre-computed embeddings between stages.

    Args:
        prompt_embeds: Optional serialized prompt embeddings payload.
            Used for direct transfer of embeddings between stages.
        additional_information: Optional additional information payload
            containing tensors or lists to be passed along with the request.
    """

    def __init__(
        self,
        prompt_embeds: PromptEmbedsPayload | torch.Tensor | None = None,
        # Optional external request ID for tracking
        external_req_id: str | None = None,
        additional_information: AdditionalInformationPayload | None = None,
        *args,
        **kwargs,
    ):
        prompt_embeds_tensor = self._maybe_decode_prompt_embeds(prompt_embeds)
        super().__init__(prompt_embeds=prompt_embeds_tensor, *args, **kwargs)
        # Preserve serialized prompt embeddings payload (optional)
        self.prompt_embeds_payload: PromptEmbedsPayload | None = (
            prompt_embeds if isinstance(prompt_embeds, PromptEmbedsPayload) else None
        )
        # Optional external request ID for tracking
        self.external_req_id: str | None = external_req_id
        # Serialized additional information payload (optional)
        self.additional_information: AdditionalInformationPayload | None = additional_information

    @staticmethod
    def _maybe_decode_prompt_embeds(
        prompt_embeds: PromptEmbedsPayload | torch.Tensor | None,
    ) -> torch.Tensor | None:
        if isinstance(prompt_embeds, PromptEmbedsPayload):
            dtype = getattr(np, prompt_embeds.dtype)
            arr = np.frombuffer(prompt_embeds.data, dtype=dtype)
            arr = arr.reshape(prompt_embeds.shape)
            return torch.from_numpy(arr)
        return prompt_embeds

    @classmethod
    def from_engine_core_request(
        cls,
        request: OmniEngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        """Create an OmniRequest from an OmniEngineCoreRequest.

        Args:
            request: The OmniEngineCoreRequest to convert
            block_hasher: Optional function to compute block hashes for
                prefix caching

        Returns:
            OmniRequest instance created from the engine core request
        """
        return cls(
            request_id=request.request_id,
            # Optional external request ID for tracking
            external_req_id=request.external_req_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            additional_information=request.additional_information,
            resumable=request.resumable,
            reasoning_ended=request.reasoning_ended,
        )


@dataclass
class OmniStreamingUpdate:
    """
    Override: add additional information
    Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None
    additional_information: AdditionalInformationPayload | None = None

    @classmethod
    def from_request(cls, request: "Request") -> "OmniStreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
            additional_information=request.additional_information,
        )
