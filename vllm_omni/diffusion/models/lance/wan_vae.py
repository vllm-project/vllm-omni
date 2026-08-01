# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility exports for Lance's Wan2.2 VAE adapter.

The checkpoint-compatible Wan2.2 VAE implementation is shared with other model
adapters from ``vllm_omni.diffusion.models.wan2_2.wan_vae``. Lance keeps the
``LanceWanVAE`` name as a compatibility alias because existing Lance pipeline
code and type annotations refer to it, but the math and checkpoint loading live
in the neutral Wan2.2 module rather than in the Lance model directory.
"""

from __future__ import annotations

from vllm_omni.diffusion.models.wan2_2.wan_vae import Wan22VAE, WanVAE_, build_wan22_vae

LanceWanVAE = Wan22VAE

__all__ = ["LanceWanVAE", "Wan22VAE", "WanVAE_", "build_wan22_vae"]
