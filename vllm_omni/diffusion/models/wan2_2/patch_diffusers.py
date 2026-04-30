# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

from vllm_omni.diffusion.layers.norm import RMSNormVAE


def patch_wan_rms_norm():
    """Patch diffusers Wan RMSNorm implementation with RMSNormVAE."""

    # NOTE: iterate over a snapshot of sys.modules. `hasattr` can trigger lazy
    # submodule imports (e.g. transformers' `_LazyModule.__getattr__`), which
    # mutate sys.modules during iteration and raise
    # `RuntimeError: dictionary changed size during iteration`.
    for module_name, module in list(sys.modules.items()):
        if hasattr(module, "WanRMS_norm"):
            setattr(module, "WanRMS_norm", RMSNormVAE)
