# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent fast paths for Wan-family causal VAEs.

The diffusers Wan VAE and the implementations registered in :mod:`.families`
(Qwen-Image, Qwen-Image-2.1) share the replacement forwards below.

``install_wan_vae_fastpath(vae, level=...)`` rebinds the forwards of one loaded
``AutoencoderKLWan`` instance:

* ``"lossless"`` (default): bit-exact rewrites of the decoder's data movement
  and normalization (fused Triton kernels with exact PyTorch fallbacks).
* ``"channels_last"``: additionally converts decoder convolution weights to
  channels-last memory format (faster cuDNN kernels, not bit-exact).
* ``"off"``: leave the diffusers implementation untouched.

The framework installs it from ``vllm_omni.diffusion.registry.initialize_model``
according to ``OmniDiffusionConfig.vae_fast_path``.

``install_wan_vae_encoder_fastpath`` applies the same levels independently to
the residual, patchified encoder used by Cosmos3, controlled by
``OmniDiffusionConfig.vae_encode_fast_path``. Its report/uninstall state is
separate from the decoder's.
"""

from .decode import decode_frames
from .encode import can_encode_frames, encode_frames
from .families import WanVaeFamily, find_family, register_family
from .forwards import probe_stats, reset_probe_stats
from .install import (
    ENCODER_REPORT_ATTR,
    REPORT_ATTR,
    VAE_FAST_PATH_LEVELS,
    WanVaeFastPathReport,
    install_wan_vae_encoder_fastpath,
    install_wan_vae_fastpath,
    is_encoder_installed,
    is_installed,
    uninstall_wan_vae_encoder_fastpath,
    uninstall_wan_vae_fastpath,
)

__all__ = [
    "ENCODER_REPORT_ATTR",
    "REPORT_ATTR",
    "VAE_FAST_PATH_LEVELS",
    "WanVaeFastPathReport",
    "WanVaeFamily",
    "decode_frames",
    "find_family",
    "register_family",
    "can_encode_frames",
    "encode_frames",
    "install_wan_vae_encoder_fastpath",
    "install_wan_vae_fastpath",
    "is_installed",
    "is_encoder_installed",
    "probe_stats",
    "reset_probe_stats",
    "uninstall_wan_vae_encoder_fastpath",
    "uninstall_wan_vae_fastpath",
]
