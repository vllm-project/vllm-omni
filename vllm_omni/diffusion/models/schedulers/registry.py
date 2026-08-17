# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler registry and construction seam for diffusion pipelines.

This module lets external packages (e.g. RL frameworks such as verl-omni)
inject their own scheduler classes into vllm-omni's diffusion pipelines
without editing any denoise loop. A scheduler class can be registered under
a name via :func:`register_scheduler` or advertised through the
``vllm_omni.schedulers`` entry-point group, and is then selected engine-wide
via ``OmniDiffusionConfig.scheduler`` (registry name or dotted class path)
with constructor overrides from ``OmniDiffusionConfig.scheduler_kwargs``.

Injected scheduler classes must satisfy the same contract the stock
pipelines already rely on (diffusers-style):

* ``step(noise_pred, t, latents, return_dict=False, generator=None)`` —
  called only through ``CFGParallelMixin.scheduler_step[_maybe_with_cfg]``;
  with ``return_dict=False`` it must return a tuple whose first element is
  the stepped latents. ``generator`` is only passed when not ``None``.
* ``set_timesteps(num_inference_steps, device=...)`` and expose
  ``.timesteps`` afterwards.
* ``set_begin_index(...)`` for pipeline-parallel timestep slicing.
* a ``.config`` attribute (diffusers ``ConfigMixin``-style) carrying at
  least ``num_train_timesteps``.
* deepcopy-safe: step-wise execution deep-copies the scheduler per request.

Injected classes are constructed with
``cls.from_pretrained(od_config.model, subfolder="scheduler",
local_files_only=od_config.local_files_only, **scheduler_kwargs)``,
matching how the stock schedulers are loaded. When no scheduler is
configured, :func:`build_pipeline_scheduler` falls through to the calling
pipeline's own ``default_builder`` so the default path stays bit-identical.

This module must stay importable without torch/diffusers and must not
import ``OmniDiffusionConfig`` at runtime (duck typing only) to avoid
import cycles.
"""

import importlib
import importlib.metadata
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

SCHEDULER_ENTRY_POINT_GROUP = "vllm_omni.schedulers"

_SCHEDULER_REGISTRY: dict[str, type] = {}
_entry_points_loaded = False


def register_scheduler(name: str, cls: type | None = None):
    """Register a scheduler class under ``name``.

    Usable as a direct call (``register_scheduler("flow_match_sde", MyScheduler)``)
    or as a decorator (``@register_scheduler("flow_match_sde")``).
    """

    def _do(c: type) -> type:
        _SCHEDULER_REGISTRY[name] = c
        return c

    return _do(cls) if cls is not None else _do


def _load_entry_points() -> None:
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    for ep in importlib.metadata.entry_points(group=SCHEDULER_ENTRY_POINT_GROUP):
        _SCHEDULER_REGISTRY.setdefault(ep.name, ep.load())
    _entry_points_loaded = True


def resolve_scheduler_cls(ref: str | type | None) -> type | None:
    """Resolve a scheduler reference to a class.

    ``None`` and class objects pass through unchanged. Strings resolve
    against the registry (including lazily loaded ``vllm_omni.schedulers``
    entry points) first, then fall back to a dotted import path. A bare
    unknown name raises ``KeyError`` listing the registered names.
    """
    if ref is None or isinstance(ref, type):
        return ref
    _load_entry_points()
    if ref in _SCHEDULER_REGISTRY:
        return _SCHEDULER_REGISTRY[ref]
    module, _, attr = ref.rpartition(".")  # dotted-path fallback
    if not module:
        raise KeyError(f"Unknown scheduler '{ref}'. Registered: {sorted(_SCHEDULER_REGISTRY)}")
    return getattr(importlib.import_module(module), attr)


def build_pipeline_scheduler(
    od_config: "OmniDiffusionConfig",
    scheduler_cls: str | type | None = None,
    scheduler_kwargs: dict[str, Any] | None = None,
    default_builder: Callable[[], Any] | None = None,
):
    """Single construction seam replacing hardcoded scheduler ``from_pretrained`` sites.

    Resolution order: explicit ``scheduler_cls`` arg > ``od_config.scheduler``
    > pipeline default. When no scheduler is configured, ``default_builder``
    is invoked and its return value is passed through untouched, preserving
    the pipeline's existing construction bit-for-bit.
    """
    cls = resolve_scheduler_cls(scheduler_cls) or resolve_scheduler_cls(getattr(od_config, "scheduler", None))
    if cls is None:
        assert default_builder is not None
        return default_builder()
    kwargs = dict(getattr(od_config, "scheduler_kwargs", None) or {})
    kwargs.update(scheduler_kwargs or {})
    return cls.from_pretrained(
        od_config.model,
        subfolder="scheduler",
        local_files_only=getattr(od_config, "local_files_only", False),
        **kwargs,
    )
