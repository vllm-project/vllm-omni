# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CausVid Wan model package (DiT, weights, pipeline).

Shared Noisy KV engine pieces (schedule, executor, FlowEuler, KV cache) live in
``vllm_omni.experimental.ar_diffusion``.
"""

from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import (
    CAUSVID_DEFAULT_DENOISE_STEPS,
    CAUSVID_DEFAULT_SHIFT,
    HF_CAUSVID_REPO,
    HF_MODEL_ID,
    CausVidWanPipeline,
)
from vllm_omni.diffusion.models.causvid_wan.transformer import (
    StageWanTransformer,
    stage_layer_range,
)
from vllm_omni.diffusion.models.causvid_wan.weights import (
    causvid_pt_to_diffusers_state,
    iter_causvid_diffusers_weights,
    load_causvid_generator_state,
    remap_causvid_key,
)

__all__ = [
    "CAUSVID_DEFAULT_DENOISE_STEPS",
    "CAUSVID_DEFAULT_SHIFT",
    "HF_CAUSVID_REPO",
    "HF_MODEL_ID",
    "CausVidWanPipeline",
    "StageWanTransformer",
    "causvid_pt_to_diffusers_state",
    "iter_causvid_diffusers_weights",
    "load_causvid_generator_state",
    "remap_causvid_key",
    "stage_layer_range",
]
