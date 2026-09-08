# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""π0 (Pi-Zero) VLA model for vllm-omni.

PaliGemma (SigLIP vision + Gemma-2B LM) + Gemma-300M action expert + a
flow-matching action head. Outputs a continuous action chunk
``[horizon, action_dim]`` rather than tokens.
"""

from vllm_omni.diffusion.models.pi0.config import Pi0Config
from vllm_omni.diffusion.models.pi0.modeling_pi0 import (
    GemmaVariantConfig,
    PaliGemmaWithActionExpert,
    Pi0ForActionPrediction,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)
from vllm_omni.diffusion.models.pi0.pipeline_pi0 import (
    Pi0Pipeline,
    get_pi0_post_process_func,
)
from vllm_omni.diffusion.models.pi0.processor_pi0 import (
    Pi0ImageProcessor,
    build_model_inputs,
    pil_image_to_tensor,
    resize_with_pad,
    tokenize_prompt,
)

__all__ = [
    "Pi0Config",
    "Pi0ForActionPrediction",
    "PaliGemmaWithActionExpert",
    "GemmaVariantConfig",
    "get_gemma_config",
    "create_sinusoidal_pos_embedding",
    "make_att_2d_masks",
    "prepare_attention_masks_4d",
    "Pi0ImageProcessor",
    "build_model_inputs",
    "pil_image_to_tensor",
    "resize_with_pad",
    "tokenize_prompt",
    "Pi0Pipeline",
    "get_pi0_post_process_func",
]
