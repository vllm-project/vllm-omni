# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Torch Tweak backend selection for diffusion pipelines on Intel GPUs.

Torch Tweak measures XPU graphs, Torch Inductor, and eager on the recorded input shapes of each
served module and keeps the fastest one per shape. It replaces torch.compile for the stage,
and ``enforce_eager`` turns both off. Enable it with
``additional_config={"torch_tweak": {"workdir": ...}}``:

* the first run records the inputs to the work directory;
* the next run tunes them on the first forward and writes the choice;
* later runs load the choice.

``mode`` and ``backends`` are passed to ``torch_tweak.torch.ServingConfig``.
``modules`` lists the pipeline attributes to serve, the DiT modules by default.
A shape that was not recorded runs the module compiled with dynamic shapes.
"""

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader.config import offload_enabled
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

DEFAULT_DIT_MODULES = ("transformer", "transformer_2")


def serve_with_torch_tweak(pipeline: nn.Module, od_config: OmniDiffusionConfig, device: torch.device) -> bool:
    """Wraps the configured modules of pipeline in place. Returns False when Torch Tweak is not enabled."""
    settings = dict(od_config.additional_config.get("torch_tweak") or {})
    if not settings:
        return False
    _check_supported(od_config)

    import torch_tweak.torch as tt

    attr_names = settings.pop("modules", None) or getattr(pipeline, "_dit_modules", None) or DEFAULT_DIT_MODULES
    config = tt.ServingConfig(**settings)
    served = []
    for attr_name in attr_names:
        module = getattr(pipeline, attr_name, None)
        if module is None:
            continue
        tt.ServedModule(module, attr_name, config, device=device)
        served.append(attr_name)
    logger.info("Torch Tweak serves %s from %s.", served, config.workdir)
    return True


def _check_supported(od_config: OmniDiffusionConfig) -> None:
    if not current_omni_platform.is_xpu():
        raise ValueError(f"Torch Tweak requires an Intel XPU, not {current_omni_platform.device_type}")
    # Each rank profiles a different number of steps, so collectives inside the module would deadlock.
    if od_config.parallel_config.world_size > 1:
        raise ValueError("Torch Tweak requires a single-GPU diffusion stage")
    # Offloading moves weights between calls. A replayed XPU graph reads the addresses it captured.
    if offload_enabled(od_config):
        raise ValueError("Torch Tweak does not support CPU offloading")
    # A recorded XPU graph replays every block that a cache backend would skip.
    if od_config.cache_backend not in (None, "none"):
        raise ValueError("Torch Tweak requires cache_backend='none'")
