# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

from vllm_omni.diffusion.layers.norm import RMSNormVAE


def patch_wan_rms_norm():
    """Patch diffusers Wan RMSNorm implementation with RMSNormVAE."""

    for module_name, module in sys.modules.items():
        if hasattr(module, "WanRMS_norm"):
            setattr(module, "WanRMS_norm", RMSNormVAE)
