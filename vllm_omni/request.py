from collections.abc import Callable
from typing import TYPE_CHECKING

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
        prompt_embeds: PromptEmbedsPayload | None = None,
        # Optional external request ID for tracking
        external_req_id: str | None = None,
        additional_information: AdditionalInformationPayload | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Serialized prompt embeddings payload (optional)
        self.prompt_embeds: PromptEmbedsPayload | None = prompt_embeds
        # Optional external request ID for tracking
        self.external_req_id: str | None = external_req_id
        # Serialized additional information payload (optional)
        self.additional_information: AdditionalInformationPayload | None = additional_information

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
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            additional_information=request.additional_information,
        )
