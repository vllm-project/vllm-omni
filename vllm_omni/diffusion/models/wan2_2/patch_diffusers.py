# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

from vllm_omni.diffusion.layers.norm import RMSNormVAE


def patch_wan_rms_norm():
    """Patch diffusers Wan RMSNorm implementation with RMSNormVAE."""

    # Probe `__dict__` directly instead of `hasattr`: the latter triggers
    # custom `__getattr__` hooks (e.g. transformers' image_processing alias
    # modules), which both emit spurious deprecation warnings and can lazily
    # import submodules, mutating sys.modules mid-iteration.
    for module in list(sys.modules.values()):
        module_dict = getattr(module, "__dict__", None)
        if module_dict is not None and "WanRMS_norm" in module_dict:
            module_dict["WanRMS_norm"] = RMSNormVAE
