# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast paths for the diffusers Wan (2.1/2.2) causal video VAE decoder.

``install_wan_vae_fastpath(vae, level=...)`` rebinds the forwards of one loaded
``AutoencoderKLWan`` instance:

* ``"lossless"`` (default): bit-exact rewrites of the decoder's data movement
  and normalization (fused Triton kernels with exact PyTorch fallbacks).
* ``"channels_last"``: additionally converts decoder convolution weights to
  channels-last memory format (faster cuDNN kernels, not bit-exact).
* ``"off"``: leave the diffusers implementation untouched.

The framework installs it from ``vllm_omni.diffusion.registry.initialize_model``
according to ``OmniDiffusionConfig.vae_fast_path``.
"""

from .decode import decode_frames
from .install import (
    REPORT_ATTR,
    VAE_FAST_PATH_LEVELS,
    WanVaeFastPathReport,
    install_wan_vae_fastpath,
    is_installed,
    uninstall_wan_vae_fastpath,
)

__all__ = [
    "REPORT_ATTR",
    "VAE_FAST_PATH_LEVELS",
    "WanVaeFastPathReport",
    "decode_frames",
    "install_wan_vae_fastpath",
    "is_installed",
    "uninstall_wan_vae_fastpath",
]
