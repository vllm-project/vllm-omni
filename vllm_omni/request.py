from typing import Optional

from vllm.multimodal.inputs import MultiModalKwargs
from vllm.utils import is_list_of
from vllm.v1.request import Request
from vllm.v1.structured_output.request import StructuredOutputRequest

from vllm_omni.engine import (
    AdditionalInformationPayload,
    OmniEngineCoreRequest,
    PromptEmbedsPayload,
)


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
        self.additional_information: Optional[AdditionalInformationPayload] = (
            additional_information
        )

    @classmethod
    def from_engine_core_request(cls, request: OmniEngineCoreRequest) -> "Request":
        if request.mm_inputs is not None:
            assert isinstance(request.mm_inputs, list)
            assert is_list_of(
                request.mm_inputs, MultiModalKwargs
            ), "mm_inputs was not updated in EngineCore.add_request"

        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=(
                StructuredOutputRequest(sampling_params=request.sampling_params)
                if request.sampling_params
                else None
            ),
            cache_salt=request.cache_salt,
            priority=request.priority,
            prompt_embeds=request.prompt_embeds,
            additional_information=request.additional_information,
        )
