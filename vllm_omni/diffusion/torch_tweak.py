# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Torch Tweak backend selection for diffusion pipelines on Intel GPUs.

Torch Tweak measures XPU graphs, Torch Inductor, and eager on the recorded input shapes of each
served module and keeps the fastest one per shape. It replaces torch.compile for the stage,
and ``enforce_eager`` turns both off.
Enable it with ``additional_config={"torch_tweak": {"workdir": ...}}``:

* the first run records the inputs to the work directory;
* the next run tunes them on the first forward and writes the choice;
* later runs load the choice.

``mode`` and ``backends`` are passed to ``torch_tweak.torch.ServingConfig``.
``modules`` lists the pipeline attributes to serve, the DiT modules by default.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

DEFAULT_DIT_MODULES = ("transformer", "transformer_2")


def serve_with_torch_tweak(pipeline: Any, od_config: Any, device: torch.device) -> bool:
    """Wraps the configured modules of pipeline in place. Returns False when Torch Tweak is not enabled."""
    settings = dict(od_config.additional_config.get("torch_tweak") or {})
    if not settings:
        return False
    if not current_omni_platform.is_xpu():
        logger.warning("Torch Tweak runs only on Intel XPU; ignoring it on %s.", current_omni_platform.device_type)
        return False

    import torch_tweak.torch as tt

    attr_names = settings.pop("modules", None) or getattr(pipeline, "_dit_modules", None) or DEFAULT_DIT_MODULES
    config = tt.ServingConfig(**settings)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    served = []
    for attr_name in attr_names:
        module = getattr(pipeline, attr_name, None)
        if module is None:
            continue
        tt.serve(module, f"{od_config.model}.{attr_name}.rank{rank}", config, device=device)
        served.append(attr_name)
    logger.info("Torch Tweak serves %s from %s.", served, config.workdir)
    return True
