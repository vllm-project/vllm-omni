"""vLLM-Omni integration for DiNa-LRM.

This sub-package exposes the vLLM-Omni compatible pipeline and processing
functions for DiNa-LRM (Diffusion-Native Latent Reward Model).

Files
-----
pipeline_dina_lrm.py  – vLLM-Omni pipeline (DiNaLRMPipeline + pre/post funcs)
dina_lrm_model.py     – Model code (SD3Backbone, RewardHead, encode_prompt …)
                        Self-contained; no diffusion_rm package dependency.
"""

from .dina_lrm_model import (
    SD3RewardModel,
)
from .pipeline_dina_lrm import (
    DiNaLRMPipeline,
    get_dina_lrm_post_process_func,
    get_dina_lrm_pre_process_func,
)

__all__ = [
    "DiNaLRMPipeline",
    "get_dina_lrm_pre_process_func",
    "get_dina_lrm_post_process_func",
    "SD3RewardModel",
]
