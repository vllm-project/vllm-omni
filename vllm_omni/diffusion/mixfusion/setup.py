# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion startup wiring: config validation and driver mounting.

Shared components (engine, model loader) call these once; they hold no
MixFusion logic of their own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks.mixfusion import apply_mixfusion, plan_foreign_hooks
from vllm_omni.diffusion.mixfusion.chunked import resolve_mixfusion_adapter
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.mixfusion.scheduling import MixFusionCapability
from vllm_omni.diffusion.utils.tf_utils import find_module_with_attr

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.request import OmniDiffusionRequest

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

    The engine discovers the effective capability after all model wrappers
    are installed. Pipelines without a plan keep ordinary dense scheduling.
    """
    if not (getattr(od_config, "enable_mixfusion", False) or supported):
        return
    transformers = _dit_modules(model)
    if not transformers:
        return

    for attr, transformer in transformers:
        plan = getattr(transformer, "_mixfusion_plan", None)
        adapter = resolve_mixfusion_adapter(transformer, plan)
        if plan is not None and adapter is not None:
            logger.info("Applying MixFusion hooks to %s (%s): %s", type(transformer).__name__, attr, ", ".join(plan))
            apply_mixfusion(transformer, plan, adapter)
            model.mixfusion_driver = MixFusionDriver(model, transformer, adapter)
            return
    logger.info("No packable _mixfusion_plan; keeping dense request admission.")


def packing_unavailable_reason(pipeline: nn.Module) -> str | None:
    """Inspect the loaded runtime without depending on varlen support."""
    driver = getattr(pipeline, "mixfusion_driver", None)
    if driver is None:
        return "no packable MixFusion plan"
    transformer = driver.transformer
    if not getattr(transformer, "_mixfusion_hooks_applied", False):
        raise RuntimeError("Declared MixFusion hooks were not installed")
    if getattr(pipeline, "attention_kwargs", None):
        return "attention_kwargs is set"
    parallel = getattr(pipeline, "parallel_config", None)
    if parallel is not None and parallel.sequence_parallel_size > 1:
        return "sequence parallelism is enabled"
    if parallel is not None and parallel.cfg_parallel_size > 1:
        return "CFG parallel is enabled"
    foreign = plan_foreign_hooks(transformer, transformer._mixfusion_plan)
    if foreign:
        return f"foreign hooks on the packed path ({', '.join(foreign)})"
    return None


def inspect_mixfusion_capability(pipeline: nn.Module) -> MixFusionCapability:
    reason = packing_unavailable_reason(pipeline)
    if reason is not None:
        return MixFusionCapability(reason=reason)
    prepare = getattr(pipeline, "prepare_mixfusion_request", None)
    configure = getattr(pipeline, "mixfusion_request_config", None)
    if prepare is None or configure is None:
        return MixFusionCapability(reason="model declares no host-side MixFusion request preparation")
    config = configure()
    if config is None:
        return MixFusionCapability(reason="model configuration does not support packed requests")
    return MixFusionCapability(request_config=config, reason="")


def initialize_mixfusion_admission(engine: DiffusionEngine) -> None:
    """Discover initialized rank capabilities once, before accepting requests."""
    from vllm_omni.diffusion.registry import resolve_diffusion_pipeline_cls, supports_mixfusion

    engine.mixfusion_capability = MixFusionCapability()
    engine.mixfusion_prepare_request = None
    if engine.execution_mode != "step_batch" or not supports_mixfusion(engine.od_config):
        return
    capabilities = engine.executor.collective_rpc(
        "get_mixfusion_capabilities", unique_reply_rank=0, exec_all_ranks=True
    )
    if not capabilities or any(item != capabilities[0] for item in capabilities[1:]):
        raise RuntimeError("MixFusion capabilities disagree across Worker ranks")
    engine.mixfusion_capability = capabilities[0]
    if capabilities[0].request_config is None:
        logger.info("MixFusion uses dense admission: %s", capabilities[0].reason)
        return
    pipeline_cls = resolve_diffusion_pipeline_cls(engine.od_config)
    prepare = getattr(pipeline_cls, "prepare_mixfusion_request", None)
    if not callable(prepare):
        raise RuntimeError("Loaded MixFusion model has no host-side request preparer")
    engine.mixfusion_prepare_request = prepare


def prepare_mixfusion_admission(engine: DiffusionEngine, request: OmniDiffusionRequest) -> None:
    """Attach trusted host metadata; unsupported requests retain dense keys."""
    from vllm_omni.diffusion.mixfusion.request import (
        DEFAULT_MAX_CHUNKS,
        DEFAULT_MIN_CHUNK_TOKENS,
        mixfusion_requested,
    )
    from vllm_omni.diffusion.mixfusion.scheduling import plan_batch

    request.mixfusion_info = None
    prepare = getattr(engine, "mixfusion_prepare_request", None)
    if prepare is None or not request.use_step_execution:
        return
    if not mixfusion_requested(request.sampling_params, engine.od_config, supported=True):
        return
    extras = request.sampling_params.extra_args or {}
    info = prepare(
        request,
        engine.mixfusion_capability.request_config,
        int(
            extras.get(
                "mixfusion_min_chunk_tokens",
                getattr(engine.od_config, "mixfusion_min_chunk_tokens", DEFAULT_MIN_CHUNK_TOKENS),
            )
        ),
        int(extras.get("mixfusion_max_chunks", getattr(engine.od_config, "mixfusion_max_chunks", DEFAULT_MAX_CHUNKS))),
    )
    if info is not None and plan_batch([(request.request_id, info)]) is not None:
        request.mixfusion_info = info
