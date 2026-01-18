from __future__ import annotations

from vllm.v1.core.sched.output import SchedulerOutput

from .omni_ar_base_scheduler import BaseOmniARScheduler


class OmniARScheduler(BaseOmniARScheduler):
    """
    OmniARScheduler: Standard scheduler for vLLM-Omni Auto-Regressive models.

    This scheduler extends BaseOmniARScheduler with minimal modifications,
    primarily adding Omni-specific payload enrichment to the scheduler output.
    
    For multimodal comprehension stages that need modality-aware scheduling,
    use OmniModalityAwareScheduler instead.
    
    Inherited from BaseOmniARScheduler:
    - update_from_output(): Fixes check_stop bug for multimodal models
    - _enrich_scheduler_output(): Wraps NewRequestData with OmniNewRequestData
    """

    @classmethod
    def validate_stage_config(cls, global_policy, stage_id, is_comprehension, is_dit):
        if is_dit:
            raise ValueError(
                f"Stage {stage_id} Configuration Error: {cls.__name__} is an "
                "Auto-Regressive scheduler and cannot be used for DiT/Audio generation stages. "
                "Please check your 'engine_output_type' or 'scheduler_cls' settings."
            )

    def schedule(self) -> SchedulerOutput:  # type: ignore[override]
        """
        Schedule requests and enrich output with Omni-specific payloads.
        
        This method calls the parent vLLM scheduler's schedule() and then
        enriches the output with Omni-specific data (prompt_embeds,
        additional_information) via _enrich_scheduler_output().
        """
        scheduler_output = super().schedule()
        scheduler_output = self._enrich_scheduler_output(scheduler_output)
        return scheduler_output
