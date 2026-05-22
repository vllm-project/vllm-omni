# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project and the xDiT authors.
#
# This module is adapted from xDiT (https://github.com/xdit-project/xdit)

"""
PipeFusion scheduler mixin for diffusion models.

Provides patch-wise cache management for schedulers in PipeFusion mode:
- Splitting scheduler state (model outputs, cached samples) into per-patch versions
- Swapping per-patch state in/out during each scheduler step
- Gating step-level state advancement to only the last patch
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, ClassVar

import torch

from vllm_omni.diffusion.distributed.pipefusion.pipefusion_runtime import (
    get_pipefusion_runtime,
    is_pipefusion_initialized,
)


class PipeFusionSchedulerMixin:
    """
    Mixin for schedulers that participate in PipeFusion patch-wise execution.

    In async PipeFusion mode, the scheduler processes multiple spatial patches
    per timestep. Each patch needs its own copy of certain scheduler state
    (e.g., cached model outputs, previous samples) while sharing others
    (e.g., step index, timestep list).

    Subclasses declare per-patch cached attributes via `_pipefusion_patch_cache_spec`
    and use the provided helpers to manage state during `step()`.
    """

    # Override in subclass: list of (attr_name, cache_type)
    #   cache_type "list" = list[Tensor | None], each element split independently
    #   cache_type "tensor" = single Tensor | None, split directly
    _pipefusion_patch_cache_spec: ClassVar[list[tuple[str, str]]] = []

    # Shared scheduler state attributes that should only advance/change on the last patch
    _pipefusion_shared_state_attrs: ClassVar[list[str]] = [
        "_step_index",
        "lower_order_nums",
        "this_order",
    ]

    # Shared list/tensor attributes that should only be updated on the first patch (patch_idx == 0)
    _pipefusion_first_patch_only_attrs: ClassVar[list[str]] = ["timestep_list"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if is_pipefusion_initialized():
            init = cls.__dict__.get("__init__")
            if callable(init):

                @wraps(init)
                def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
                    init(self, *args, **kwargs)
                    self._init_patch_caches()

                cls.__init__ = wrapped_init

            step = cls.__dict__.get("step")
            if callable(step):

                @wraps(step)
                def wrapped_step(self, *args: Any, **kwargs: Any) -> Any:
                    if not get_pipefusion_runtime().patch_mode:
                        return step(self, *args, **kwargs)
                    return self._pipefusion_scheduler_step(step, *args, **kwargs)

                cls.step = wrapped_step

    def _pipefusion_scheduler_step(self, step: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a scheduler step with PipeFusion patch-state bookkeeping."""
        patch_idx, is_last_patch = self._step_begin()

        # Snapshot shared state to restore if not last patch
        shared_snapshots = {}
        for attr in getattr(self, "_pipefusion_shared_state_attrs", []):
            if hasattr(self, attr):
                val = getattr(self, attr)
                shared_snapshots[attr] = val.clone() if isinstance(val, torch.Tensor) else val

        # Snapshot first-patch-only state to restore if patch_idx > 0
        first_patch_snapshots = {}
        if patch_idx > 0:
            for attr in getattr(self, "_pipefusion_first_patch_only_attrs", []):
                if hasattr(self, attr):
                    val = getattr(self, attr)
                    if isinstance(val, list):
                        first_patch_snapshots[attr] = list(val)
                    elif isinstance(val, torch.Tensor):
                        first_patch_snapshots[attr] = val.clone()
                    else:
                        first_patch_snapshots[attr] = val

        # Run original step
        res = step(self, *args, **kwargs)

        # Save updated "tensor" caches back to patch storage
        if self._pf_patch_caches is not None:
            for attr_name, cache_type in self._pipefusion_patch_cache_spec:
                if cache_type == "tensor":
                    self._update_value(attr_name, getattr(self, attr_name))

        # Restore shared state if not the last patch
        if not is_last_patch:
            for attr, val in shared_snapshots.items():
                setattr(self, attr, val)

        # Restore first-patch-only state if patch_idx > 0
        if patch_idx > 0:
            for attr, val in first_patch_snapshots.items():
                setattr(self, attr, val)

        return res

    def _init_patch_caches(self) -> None:
        """Initialize per-patch cache storage. Call during __init__."""
        self._pf_patch_caches: dict[str, list[Any]] | None = None

    def split_caches_for_patches(self, patch_sizes: list[int], dim: int = -2) -> None:
        """
        Split cached scheduler state into per-patch versions for async pipeline.

        Called when transitioning from sync to async pipeline mode.
        Each attribute listed in `_pipefusion_patch_cache_spec` is split
        along the specified dimension.

        Args:
            patch_sizes: Size of each patch along the split dimension.
            dim: Dimension along which to split (default -2 for height).
        """
        num_patches = get_pipefusion_runtime().num_pipeline_patch
        self._pf_patch_caches = {}

        for attr_name, cache_type in self._pipefusion_patch_cache_spec:
            value = getattr(self, attr_name)

            if cache_type == "list":
                # list[Tensor | None] — split each non-None element
                per_patch: list[list[torch.Tensor | None]] = []
                for patch_idx in range(num_patches):
                    patch_list: list[torch.Tensor | None] = []
                    for item in value:
                        if item is not None:
                            splits = item.split(patch_sizes, dim=dim)
                            patch_list.append(splits[patch_idx])
                        else:
                            patch_list.append(None)
                    per_patch.append(patch_list)
                self._pf_patch_caches[attr_name] = per_patch

            elif cache_type == "tensor":
                # single Tensor | None — split directly
                if value is not None:
                    splits = value.split(patch_sizes, dim=dim)
                    self._pf_patch_caches[attr_name] = list(splits)
                else:
                    self._pf_patch_caches[attr_name] = [None] * num_patches

    def clear_patch_caches(self) -> None:
        """Clear per-patch caches when exiting async pipeline mode."""
        self._pf_patch_caches = None

    def _step_begin(self) -> tuple[int, bool]:
        """
        Begin a scheduler step: get patch context and load per-patch state.

        Combines patch context lookup and state loading into a single call.
        In patch mode, swaps cached attributes to the current patch's versions.

        Returns:
            (patch_idx, is_last_patch) tuple:
            - patch_idx: Current patch index (0 if not in patch mode).
            - is_last_patch: Whether this is the last patch in the sequence.
        """
        runtime_state = get_pipefusion_runtime()

        patch_idx, is_last_patch = 0, True
        if runtime_state.patch_mode and self._pf_patch_caches is not None:
            patch_idx = runtime_state.pipeline_patch_idx
            is_last_patch = patch_idx == runtime_state.num_pipeline_patch - 1

            for attr_name, _ in self._pipefusion_patch_cache_spec:
                if attr_name in self._pf_patch_caches:
                    setattr(self, attr_name, self._pf_patch_caches[attr_name][patch_idx])

        return patch_idx, is_last_patch

    def _update_value(self, attr_name: str, value: Any) -> None:
        """
        Save a value back to the per-patch cache.

        Needed for "tensor" caches where assignment rebinds the attribute
        rather than mutating it in-place. "list" caches that are mutated
        in-place (e.g., shifting elements) are updated automatically since
        `_step_begin` sets self.attr to the actual list object.

        Args:
            attr_name: Name of the cached attribute.
            value: The value to save.
        """
        runtime_state = get_pipefusion_runtime()

        if self._pf_patch_caches is not None and attr_name in self._pf_patch_caches:
            patch_mode = runtime_state.patch_mode
            patch_idx = runtime_state.pipeline_patch_idx if patch_mode else 0
            self._pf_patch_caches[attr_name][patch_idx] = value
