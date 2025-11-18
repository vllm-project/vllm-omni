from typing import TYPE_CHECKING, Callable, Optional

from vllm.v1.request import Request
from vllm.v1.structured_output.request import StructuredOutputRequest

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_omni.engine import AdditionalInformationPayload, OmniEngineCoreRequest, PromptEmbedsPayload


class OmniRequest(Request):
    def __init__(
        self,
        prompt_embeds: Optional[PromptEmbedsPayload] = None,
        additional_information: Optional[AdditionalInformationPayload] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Serialized prompt embeddings payload (optional)
        self.prompt_embeds: Optional[PromptEmbedsPayload] = prompt_embeds
        # Serialized additional information payload (optional)
        self.additional_information: Optional[AdditionalInformationPayload] = additional_information

    @classmethod
    def from_engine_core_request(
        cls,
        request: OmniEngineCoreRequest,
        block_hasher: Optional[Callable[["Request"], list["BlockHash"]]],
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=(
                StructuredOutputRequest(sampling_params=request.sampling_params) if request.sampling_params else None
            ),
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            prompt_embeds=request.prompt_embeds,
            additional_information=request.additional_information,
        )
