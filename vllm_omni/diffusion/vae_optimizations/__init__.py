# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional VAE decoder optimizations."""

from vllm_omni.diffusion.vae_optimizations.gate import (
    use_pipeline_vae_fast_path,
    use_vae_fast_path,
)
from vllm_omni.diffusion.vae_optimizations.optimize import (
    clear_pipeline_vae_fast_path_caches,
    optimize_pipeline_vaes,
)

__all__ = [
    "clear_pipeline_vae_fast_path_caches",
    "optimize_pipeline_vaes",
    "use_pipeline_vae_fast_path",
    "use_vae_fast_path",
]
