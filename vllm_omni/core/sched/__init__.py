"""
Scheduling components for vLLM-Omni.

Scheduler Hierarchy:
    VLLMScheduler (vllm native)
        │
    BaseOmniScheduler
        │
        ├── BaseOmniARScheduler (shared AR logic: update_from_output, _enrich_scheduler_output)
        │       │
        │       ├── OmniARScheduler (standard AR with Omni enrichment)
        │       └── OmniModalityAwareScheduler (modality-aware scheduling)
        │
        └── OmniGenerationScheduler (one-step generation for DiT)
"""

from .omni_base_scheduler import BaseOmniScheduler
from .omni_ar_base_scheduler import BaseOmniARScheduler
from .omni_ar_scheduler import OmniARScheduler
from .omni_generation_scheduler import OmniGenerationScheduler
from .omni_modality_aware_scheduler import OmniModalityAwareScheduler
from .output import OmniNewRequestData
from typing import Tuple
from vllm.utils.import_utils import resolve_obj_by_qualname as import_class
from vllm.logger import init_logger


logger = init_logger(__name__)

__all__ = [
    "BaseOmniScheduler",
    "BaseOmniARScheduler",
    "OmniARScheduler",
    "OmniGenerationScheduler",
    "OmniModalityAwareScheduler",
    "OmniNewRequestData",
    "resolve_scheduling_config",
]


def resolve_scheduling_config(
    global_policy: str | None,
    stage_id: int,
    explicit_cls: str | None,
    is_comprehension: bool,
    is_dit: bool,
) -> Tuple[str,str]:
    """
    Called by load_stage_configs_from_yaml in vllm_omni/entrypoints/utils.py.
    Resolves scheduler conflicts in user-provided configurations.
    """
    resolved_cls,resolved_scheduling_policy="",""
    # 1) Add resolution logic here for schedulers that are enabled via global_policy.
    if global_policy == "omni_modality_aware" and is_comprehension:
        if explicit_cls and "OmniModalityAwareScheduler" not in explicit_cls:
            logger.warning(
                f"Stage {stage_id}: global policy conflicts with the explicitly "
                "assigned scheduler class and will take precedence."
            )
        resolved_cls="vllm_omni.core.sched.omni_modality_aware_scheduler.OmniModalityAwareScheduler"
        resolved_scheduling_policy="omni_modality_aware"
        return resolved_cls,resolved_scheduling_policy
    
    # 2) For schedulers that are explicitly specified by the user, this function
    #    does not need to be modified. Instead, the scheduler class should implement
    #    validate_stage_config to validate whether it can be applied to the given stage.
    #    This helps prevent runtime issues caused by assigning an incompatible scheduler.
    if explicit_cls:
        scheduler_cls = import_class(explicit_cls)
        if hasattr(scheduler_cls, "validate_stage_config"):
            scheduler_cls.validate_stage_config(global_policy,stage_id,is_comprehension,is_dit)
        resolved_cls=explicit_cls
        if "OmniModalityAwareScheduler" in explicit_cls:
            resolved_scheduling_policy="omni_modality_aware"
        else:
            resolved_scheduling_policy="fcfs" if global_policy!="priority" else "priority"
        return resolved_cls,resolved_scheduling_policy
    
    # 3) when the user didn't pass any scheduling configuration, falls back to default configuration
    if is_dit:
        resolved_cls="vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler"
        resolved_scheduling_policy="fcfs" if global_policy!="priority" else "priority"
        return resolved_cls,resolved_scheduling_policy
    
    resolved_cls="vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler"
    resolved_scheduling_policy="fcfs" if global_policy!="priority" else "priority"
    return resolved_cls,resolved_scheduling_policy

