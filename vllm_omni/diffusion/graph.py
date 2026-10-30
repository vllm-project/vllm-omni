# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-graph replay of diffusion denoise steps.

A denoise loop calls the transformer once per step with the same tensor shapes;
only tensor contents change. Recording one step as a device graph and replaying
it for the following steps removes the per-kernel launch path, which pays when
a step is launch-bound (many small kernels per unit of device work) and its
shapes repeat across steps and requests.

Capture and replay are done by the platform's graphed-forward class
(``current_omni_platform.get_graphed_forward_cls()``: ``GraphedForward`` on
CUDA, ``XPUGraphedForward`` on XPU). It keys graphs by input shapes and
non-tensor values, keeps a small LRU, and falls back to eager on any capture
failure, agreed across the DiT ranks. This module decides whether a pipeline may
use one at all (:func:`build_denoise_graph_runner`) and lets state owned outside
the pipeline drop the graphs that froze it (:func:`invalidate_denoise_graphs`).

Safety contract for the step function: a replay re-runs the recorded kernels on
the captured buffers, so everything the step reads from outside its arguments
(module attributes, forward context) must keep the same storage while a graph is
resident, and must not change per step. :func:`write_through` helps with the
former; the refusals below cover the configurations that break the latter.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.cuda_graph import GraphedForward
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# A new shape (image size, prompt length) costs one capture; two resident
# graphs cover CFG-free and CFG requests without holding many graph pools.
DENOISE_GRAPH_MAX_ENTRIES = 2

# Every live runner in this process, so state owned outside the pipeline (LoRA
# adapters) can drop the graphs that froze it.
_RUNNERS: weakref.WeakSet[GraphedForward] = weakref.WeakSet()


def write_through(dst: torch.Tensor | None, src: torch.Tensor) -> torch.Tensor:
    """Write ``src`` into ``dst``'s storage when they match, otherwise return ``src``.

    For state that a graph body reads off a module rather than off its arguments:
    rebinding such a tensor would leave the replay pointing at a freed
    allocation, so the buffer has to keep the same storage once its shape has
    settled. The caller must invalidate its graphs whenever this returns ``src``,
    because that is the case where the storage did move.
    """
    if dst is not None and dst.shape == src.shape and dst.dtype == src.dtype and dst.device == src.device:
        dst.copy_(src)
        return dst
    return src


def is_capturing(device: torch.device) -> bool:
    """Whether a device graph is recording on ``device``'s current stream.

    For host-side checks that read a device tensor (``.item()``, ``bool()``):
    such a read synchronizes, which a capture rejects.
    """
    return device.type == current_omni_platform.device_type and current_omni_platform.is_graph_capturing()


class _StepForward(nn.Module):
    """Adapts a step function to the keyword-only module forward the graph class wraps."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        super().__init__()
        self._fn = fn

    def forward(self, **kwargs: Any) -> Any:
        return self._fn(**kwargs)


def invalidate_denoise_graphs(reason: str) -> None:
    """Drop every captured denoise graph in this process; later steps recapture.

    For state a graph froze that lives outside the step arguments and outside
    the pipeline, such as which LoRA adapter is active: its Python branches ran
    once at capture time, so a replay keeps the capture-time adapter.
    """
    for runner in list(_RUNNERS):
        if runner.num_graphs:
            logger.debug("%s: dropping %d captured denoise graph(s) (%s)", runner.name, runner.num_graphs, reason)
            runner.clear()


def _step_dependent_attention(od_config: Any) -> str | None:
    """Why attention would change from one denoise step to the next, if it would.

    Attention layers read the step index or timestep off the forward context, not
    off the step's arguments, so a replay would keep the capture step's choice.
    """
    kv_cache_dtype = getattr(od_config, "diffusion_kv_cache_dtype", None)
    if kv_cache_dtype not in (None, "auto", "float") and getattr(
        od_config, "diffusion_kv_cache_skip_step_indices", None
    ):
        return "diffusion_kv_cache_skip_steps makes KV quantization depend on the step index"
    attention_config = getattr(od_config, "diffusion_attention_config", None)
    if attention_config is not None:
        specs = [attention_config.default, *attention_config.per_role.values()]
        for spec in specs:
            skip_softmax = getattr(spec, "skip_softmax", None)
            if skip_softmax is not None and skip_softmax.disabled_until_timestep > 0:
                return "TRTLLM_ATTN skip_softmax.disabled_until_timestep makes attention depend on the timestep"
    return None


def _refusal(od_config: Any) -> str | None:
    # Lazy: the offloader package imports diffusion.data, which imports the LoRA
    # manager, which imports this module.
    from vllm_omni.diffusion.offloader.config import offload_enabled

    if getattr(od_config, "enforce_eager", False):
        return "enforce_eager=True"
    kv_mode = DiffusionKVCacheMode(getattr(od_config, "diffusion_kv_mode", None) or DiffusionKVCacheMode.DENSE_LEGACY)
    if kv_mode is not DiffusionKVCacheMode.DENSE_LEGACY:
        # The attention layers take the paged KV adapter (block tables, slot
        # mapping) from the forward context, not from the step's arguments, so
        # a replay would keep reading the capture-time request's KV layout.
        return f"diffusion_kv_mode={kv_mode.value!r} reads per-request paged KV state outside the step arguments"
    if (step_dependent := _step_dependent_attention(od_config)) is not None:
        return step_dependent
    cache_backend = getattr(od_config, "cache_backend", None) or "none"
    if cache_backend != "none":
        return f"cache_backend={cache_backend!r} decides per step whether the transformer runs at all"
    if offload_enabled(od_config):
        # Every strategy (model-level, layerwise, distributed layerwise) and the
        # diffusion_offload_config API: hooks move or all-gather the weights.
        return "CPU offload moves the weights that a captured graph reads"
    if getattr(getattr(od_config, "parallel_config", None), "pipeline_parallel_size", 1) > 1:
        return "pipeline parallelism hands intermediate tensors between stages"
    return None


def _capture_agreement_group() -> torch.distributed.ProcessGroup | None:
    """The DiT ranks' group, so a capture failure on one rank turns graphs off on all."""
    from vllm_omni.diffusion.distributed.parallel_state import get_world_group

    try:
        world = get_world_group()
    except AssertionError:
        return None
    return world.device_group if world.world_size > 1 else None


def build_denoise_graph_runner(
    od_config: Any,
    fn: Callable[..., Any],
    name: str,
    *,
    refusal: str | None = None,
) -> GraphedForward | None:
    """Return a graphed step runner if ``od_config`` opts in and nothing makes a step unsafe to replay.

    ``fn`` takes keyword arguments only (tensors, dicts/lists/tuples of them,
    and primitives) and returns a tensor or a container of tensors. ``refusal``
    lets the pipeline add its own reason to stay eager. The runner is called as
    ``runner(**kwargs)`` and returns fresh tensors on every call.
    """
    from vllm_omni.diffusion.distributed.utils import get_local_device

    requested = bool(getattr(od_config, "diffusion_use_graph", False))
    # Stated either way: a silent None is indistinguishable from "engaged and did not pay".
    logger.info("%s: diffusion_use_graph=%s", name, requested)
    if not requested:
        return None
    refusal = refusal or _refusal(od_config)
    if refusal is not None:
        logger.warning("%s: diffusion_use_graph ignored because %s", name, refusal)
        return None
    device = get_local_device()
    graph_cls = current_omni_platform.get_graphed_forward_cls()
    runner = graph_cls(
        _StepForward(fn),
        device=device,
        max_entries=DENOISE_GRAPH_MAX_ENTRIES,
        warmup_iters=1,
        group=_capture_agreement_group(),
        name=name,
    )
    if not runner.enabled:
        logger.warning(
            "%s: diffusion_use_graph ignored because %s cannot capture device graphs on %s",
            name,
            graph_cls.__name__,
            device,
        )
        return None
    _RUNNERS.add(runner)
    return runner


def warn_if_denoise_graph_unsupported(od_config: Any, pipeline: Any) -> None:
    """Say so when ``diffusion_use_graph`` is set on a pipeline that never builds a runner.

    Pipelines opt in by assigning ``self._denoise_graph_runner`` (``None`` when refused, which
    :func:`build_denoise_graph_runner` already reports); anything else would ignore the flag silently.
    """
    if getattr(od_config, "diffusion_use_graph", False) and not hasattr(pipeline, "_denoise_graph_runner"):
        logger.warning(
            "diffusion_use_graph=True is ignored: %s does not support denoise graphs; running eager.",
            type(pipeline).__name__,
        )
