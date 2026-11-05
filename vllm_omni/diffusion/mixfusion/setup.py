# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion startup wiring: config validation and driver mounting.

Shared components (engine, model loader) call these once; they hold no
MixFusion logic of their own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks.mixfusion import apply_mixfusion
from vllm_omni.diffusion.mixfusion.chunked import resolve_mixfusion_adapter
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.utils.tf_utils import find_module_with_attr

logger = init_logger(__name__)

_DEFAULT_DIT_ATTRS = ("transformer", "transformer_2", "dit", "unet")


def validate_mixfusion_config(od_config: Any, *, supported: Callable[[Any], bool]) -> bool:
    """Check server-level MixFusion settings; return whether step execution is required.

    ``supported`` resolves the pipeline capability; it is only called when
    MixFusion is enabled, so other configs never pay for the class lookup.

    Server-level MixFusion on a pipeline that cannot denoise mixed resolutions
    is a configuration error: the scheduler would relax height/width out of the
    batch key and the runner would crash on the resulting shape mismatch.
    Scheduler-paged diffusion KV runs request-level only, while MixFusion
    needs step execution, so the two cannot be combined.
    """
    if not getattr(od_config, "enable_mixfusion", False):
        return False
    if not supported(od_config):
        raise ValueError(
            f"{getattr(od_config, 'model_class_name', None)!r} does not support MixFusion. "
            "Disable enable_mixfusion, or choose a pipeline with supports_mixfusion=True."
        )
    from vllm_omni.diffusion.diffusion_kv.config import is_scheduler_paged_kv_mode

    if is_scheduler_paged_kv_mode(getattr(od_config, "diffusion_kv_mode", None)):
        raise ValueError(
            "enable_mixfusion=True requires step execution, which diffusion_kv_mode='paged_scheduler' "
            "does not support; disable one of them."
        )
    return True


def _dit_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """The pipeline's DiT components, preferring its declared ``_dit_modules``."""
    transformer_attrs = getattr(model, "_dit_modules", None) or _DEFAULT_DIT_ATTRS
    transformers = []
    search_root = model
    for attr in transformer_attrs:
        if not hasattr(search_root, attr):
            # Some pipelines nest the module that owns the transformer.
            module = find_module_with_attr(search_root, attr)
            if module is None:
                continue
            search_root = module
        transformer = getattr(search_root, attr)
        if transformer is not None:
            transformers.append((attr, transformer))
    return transformers


def mount_mixfusion(model: nn.Module, od_config: Any, *, supported: bool) -> None:
    """Apply ``_mixfusion_plan`` hooks and mount the driver on a capable pipeline.

    The MixFusion counterpart of ``_apply_sequence_parallel_if_enabled``. The
    hooks stay inert unless the driver arms a ``MixFusionStep``, so a model
    with the hooks installed still runs its serial forward unchanged when
    MixFusion is not requested.

    The driver is the entry point for ragged batches even without a plan: its
    serial fallback has to run because the dense path cannot accept list
    latents. A request may opt in on any ``supports_mixfusion`` pipeline (the
    server flag only sets the default), so the driver is mounted for every
    capable pipeline.
    """
    if not (getattr(od_config, "enable_mixfusion", False) or supported):
        return
    transformers = _dit_modules(model)
    if not transformers:
        return

    # One step packs one DiT: mount the first DiT whose plan resolves to an
    # adapter (the framework's chunked one for a declarative plan). Without
    # one, the driver still serves ragged batches serially. Admission rejects
    # SP per step, so the hooks stay inert there.
    for attr, transformer in transformers:
        plan = getattr(transformer, "_mixfusion_plan", None)
        adapter = resolve_mixfusion_adapter(transformer, plan)
        if plan is not None and adapter is not None:
            logger.info("Applying MixFusion hooks to %s (%s): %s", type(transformer).__name__, attr, ", ".join(plan))
            apply_mixfusion(transformer, plan, adapter)
            model.mixfusion_driver = MixFusionDriver(model, transformer, adapter)
            return
    logger.warning("No transformer declares a packable _mixfusion_plan; MixFusion requests run serially.")
    model.mixfusion_driver = MixFusionDriver(model, transformers[0][1], None)
