from __future__ import annotations
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.logger import init_logger

class OmniScheduler(VLLMScheduler):
    """
    OmniScheduler: Scheduler for vLLM-omni multimodal processing.

    This scheduler extends vLLM's scheduler to support multimodal and
    non-autoregressive processing with additional fields and methods
    specific to vLLM-omni.
    """

    # Ensure scheduled_new_reqs carry omni-specific payloads (e.g., additional_information)
    def schedule(self) -> SchedulerOutput:  # type: ignore[override]
        scheduler_output = super().schedule()
        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData
            # Rewrap base NewRequestData entries with OmniNewRequestData, enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_inputs=nr.mm_inputs,
                    mm_hashes=nr.mm_hashes,
                    mm_positions=nr.mm_positions,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=getattr(request, "prompt_embeds", None) if request else None,
                    additional_information=getattr(request, "additional_information", None) if request else None,
                )
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")

        return scheduler_output