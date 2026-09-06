# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 VLA model for vllm-omni.

PaliGemma (SigLIP vision + Gemma-2B LM) + Gemma-300M action expert with AdaRMS
timestep conditioning + a flow-matching action head. Outputs a continuous action
chunk ``[horizon, action_dim]`` rather than tokens.

Differs from π0 in five places: AdaRMS timestep conditioning (``time_mlp_*``
instead of ``action_time_mlp_*``), AdaRMS norms throughout the action expert, a
200-token tokenizer budget, a discretized state carried in the prompt (no
``state_proj``), and relative-action support.
"""

from vllm_omni.diffusion.models.pi05.config import (
    Pi05Config,
    UnsupportedCheckpointCapabilityError,
    resolve_excluded_action_indices,
)
from vllm_omni.diffusion.models.pi05.modeling_pi05 import (
    GemmaVariantConfig,
    PaliGemmaWithActionExpertPi05,
    Pi05AdaRMSNorm,
    Pi05ForActionPrediction,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)
from vllm_omni.diffusion.models.pi05.pipeline_pi05 import (
    Pi05Pipeline,
    get_pi05_post_process_func,
)
from vllm_omni.diffusion.models.pi05.processor_pi05 import (
    Pi05ImageProcessor,
    Pi05RelativeActions,
    build_model_inputs,
    build_pi05_prompt,
    discretize_state,
    normalize_state,
    pil_image_to_tensor,
    prefix_token_budget,
    resize_with_pad,
    tokenize_prompt,
)

__all__ = [
    "Pi05Config",
    "UnsupportedCheckpointCapabilityError",
    "resolve_excluded_action_indices",
    "Pi05ForActionPrediction",
    "PaliGemmaWithActionExpertPi05",
    "Pi05AdaRMSNorm",
    "GemmaVariantConfig",
    "get_gemma_config",
    "create_sinusoidal_pos_embedding",
    "make_att_2d_masks",
    "prepare_attention_masks_4d",
    "Pi05ImageProcessor",
    "Pi05RelativeActions",
    "build_model_inputs",
    "build_pi05_prompt",
    "discretize_state",
    "normalize_state",
    "prefix_token_budget",
    "pil_image_to_tensor",
    "resize_with_pad",
    "tokenize_prompt",
    "Pi05Pipeline",
    "get_pi05_post_process_func",
]
