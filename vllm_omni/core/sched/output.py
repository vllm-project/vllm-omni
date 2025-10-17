from dataclasses import dataclass
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request
from typing import Optional

from vllm_omni.engine import PromptEmbedsPayload, AdditionalInformationPayload



@dataclass
class OmniNewRequestData(NewRequestData):
    # Optional serialized prompt embeddings
    prompt_embeds: Optional[PromptEmbedsPayload] = None
    # Optional serialized additional information
    additional_information: Optional[AdditionalInformationPayload] = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> NewRequestData:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_inputs=request.mm_inputs,
            mm_hashes=request.mm_hashes,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            additional_information=request.additional_information,
        )