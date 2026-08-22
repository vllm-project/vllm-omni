# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVTX ranges for Falcon Perception, arranged so they cannot break a graph.

``torch.cuda.nvtx.range_push`` is an untraceable CUDA call: if TorchDynamo sees
one it graph-breaks, which is worse than having no annotation at all — a break
inside AnyUp costs more than the range measures. So the placement rule here is
absolute, not best-effort:

    **Level 1 ranges only ever wrap call sites that live outside every
    compiled region.**

Concretely, the annotated sites are the runner-facing entry points
(``preprocess`` / ``postprocess`` / ``embed_multimodal`` / ``make_omni_output``)
and the stage-1 request body. Stage 1's only compiled callee is
``itok_upsampler.forward``; wrapping the *call* to it is outside that boundary,
so the compiled function is entered with no NVTX op in its trace.

Nothing is annotated inside ``FalconPerceptionThinker.forward``. That is the
entry point vLLM compiles when ``enforce_eager: false``, and a range there would
be traced. Per-layer detail inside the backbone is already available for free
from upstream's ``VLLM_NVTX_SCOPES_FOR_PROFILING=1``, which registers its hooks
*after* the first Dynamo trace for exactly this reason
(``vllm/v1/worker/gpu_model_runner.py``: ``_register_layerwise_nvtx_hooks``).

Level 2 adds ranges *inside* AnyUp via forward hooks. Those are installed only
when AnyUp is not compiled, and ``install_anyup_hooks`` refuses to install
otherwise — a hook on a submodule of a compiled module either gets traced (a
break) or is silently skipped, and neither is a measurement worth trusting.

Environment
-----------
``FALCON_PERCEPTION_NVTX``
    ``0`` (default) off — every helper here becomes a no-op with no CUDA call.
    ``1`` coarse ranges at the safe call sites above.
    ``2`` additionally per-submodule ranges inside AnyUp (requires compile off).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = ["fine_enabled", "install_anyup_hooks", "mark", "nvtx_enabled", "nvtx_range"]


def _level() -> int:
    try:
        return int(os.environ.get("FALCON_PERCEPTION_NVTX", "0"))
    except ValueError:
        return 0


def _anyup_compiled() -> bool:
    return os.environ.get("FALCON_PERCEPTION_COMPILE_ANYUP", "0") not in ("", "0")


# Resolved once at import. The toggles are read from the environment at process
# start and never change mid-run, and re-reading os.environ inside a per-decode-
# step helper would itself cost more than the range.
_LEVEL = _level()
_ENABLED = _LEVEL >= 1
# Level 2 is dropped, not honoured-and-broken, when AnyUp is compiled.
_FINE = _LEVEL >= 2 and not _anyup_compiled()


class _NullRange:
    """Zero-cost stand-in so call sites need no ``if`` around them."""

    __slots__ = ()

    def __enter__(self) -> _NullRange:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


_NULL = _NullRange()


def nvtx_enabled() -> bool:
    return _ENABLED


def fine_enabled() -> bool:
    return _FINE


if _ENABLED:

    def nvtx_range(name: str):
        """Context manager pushing an NVTX range. Never call inside a graph."""
        return torch.cuda.nvtx.range(name)

    def mark(name: str) -> None:
        """Instantaneous NVTX marker — used for cache hit/miss, not timing."""
        torch.cuda.nvtx.mark(name)

else:

    def nvtx_range(name: str) -> _NullRange:  # noqa: ARG001
        return _NULL

    def mark(name: str) -> None:  # noqa: ARG001
        return


def install_anyup_hooks(anyup: nn.Module, *, compiled: bool) -> int:
    """Wrap AnyUp's submodules in NVTX ranges via forward hooks.

    Returns the number of hooks installed (0 when level < 2 or when AnyUp is
    compiled). Hooks — rather than inline ranges — because they can be attached
    after the fact and removed without touching the module's own code, and
    because that is the shape upstream vLLM already uses for layerwise scopes.

    Refuses to install on a compiled module: the ranges would either be traced
    into the graph (a break) or never fire, and a profile that silently measures
    neither is worse than one that measures nothing.
    """
    if not _FINE or compiled:
        return 0

    names = (
        "image_encoder",
        "key_encoder",
        "query_encoder",
        "key_features_encoder",
        "cross_decode",
        "aggregation",
    )

    installed = 0
    for name in names:
        sub = getattr(anyup, name, None)
        if sub is None or getattr(sub, "_fp_nvtx_hooked", False):
            continue

        def _pre(_mod: nn.Module, _args: tuple, _name: str = name) -> None:
            torch.cuda.nvtx.range_push(f"fp/anyup/{_name}")

        def _post(_mod: nn.Module, _args: tuple, _out: object) -> None:
            torch.cuda.nvtx.range_pop()

        sub.register_forward_pre_hook(_pre)
        sub.register_forward_hook(_post)
        sub._fp_nvtx_hooked = True
        installed += 1

    return installed
