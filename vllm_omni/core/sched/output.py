from dataclasses import dataclass, field

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request

from vllm_omni.engine import AdditionalInformationPayload, PromptEmbedsPayload


@dataclass
class OmniNewRequestData(NewRequestData):
    """New request data for omni models with embeddings support.

    Extends NewRequestData to include prompt embeddings and additional
    information for direct transfer between pipeline stages.

    Args:
        prompt_embeds: Optional serialized prompt embeddings payload
            (overrides parent's torch.Tensor type with PromptEmbedsPayload
            for cross-process serialization)
        external_req_id: Optional external request ID for tracking
        additional_information: Optional serialized additional information
            dictionary containing tensors or lists
    """

    # Optional serialized prompt embeddings (override parent type for serialization)
    prompt_embeds: PromptEmbedsPayload | None = None  # type: ignore[assignment]
    # Optional external request ID for tracking
    external_req_id: str | None = None
    # Optional serialized additional information
    additional_information: AdditionalInformationPayload | None = None

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
            prefill_token_ids=prefill_token_ids,
            additional_information=getattr(request, "additional_information", None),
        )


@dataclass
class OmniCachedRequestData(CachedRequestData):
    """Cached request data for omni models with embeddings support.

    Args:
        prompt_token_ids: Mapping from request ID to list of prompt token IDs
    """

    prompt_token_ids: dict[str, list[int]]


@dataclass
class OmniSchedulerOutput(SchedulerOutput):
    """Scheduler output with omni-specific transfer metadata."""

    finished_requests_needing_kv_transfer: dict[str, dict] = field(default_factory=dict)
