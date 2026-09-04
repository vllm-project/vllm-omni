# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replay the PersonaPlex per-frame model steps from CUDA graphs.

The lockstep tick issues ~100 tiny kernels (7B temporal step, then a 16-step
depformer over 6 layers). On an H100 80GB that is launch-bound, not
compute-bound: the per-frame step measures 77.5 ms of the 80 ms budget eager
vs 30.2 ms graphed, so with serving overhead on top the eager server falls
behind realtime, a deficit full duplex never recovers. Shapes are static
every tick, so the steps capture cleanly into CUDA graphs and replay with
one launch.

Opt-in via ``PersonaPlexConfig.cuda_graphs``; capture failure falls back to
eager.

Capture discipline (matching torch's own machinery and the production
implementations in vLLM / TensorRT-LLM / TGI / moshi):

* No warmup executions here. The runtime drives throwaway frames through the
  real serving path at load time, before any caller connects, so JIT/autotune
  has fired and the recorded example call has steady-state shapes. It then
  wipes every streaming buffer in place, which preserves the addresses these
  graphs are welded to.
* Capture itself only records; the recorded kernels do not execute.
* All graphs capture into the platform-global memory pool (the same pool
  vLLM's own runners use), in the order they replay.
* Replay validates arity, tensor shapes, and non-tensor argument equality
  against capture time; a mismatch logs once and runs that call eagerly
  rather than silently replaying with stale baked-in values. Replay inside an
  active outer stream capture also falls back to eager, so a wrapping graph
  (e.g. vLLM FULL graph mode) is never corrupted.
"""

from __future__ import annotations

import logging

import torch
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)


class _GraphedCallable:
    def __init__(self, fn, sample_args: tuple, sample_kwargs: dict):
        self._fn = fn
        self._warned = False
        self._static_args = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in sample_args)
        self._static_kwargs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in sample_kwargs.items()}
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=current_platform.get_global_graph_pool()):
            self._static_out = fn(*self._static_args, **self._static_kwargs)
        torch.accelerator.synchronize()

    def _matches(self, args, kwargs) -> bool:
        if len(args) != len(self._static_args) or set(kwargs) != set(self._static_kwargs):
            return False
        for dst, src in zip(self._static_args, args):
            if isinstance(dst, torch.Tensor) != isinstance(src, torch.Tensor):
                return False
            if isinstance(dst, torch.Tensor):
                if dst.shape != src.shape or dst.dtype != src.dtype:
                    return False
            elif dst != src:
                return False
        for k, dst in self._static_kwargs.items():
            src = kwargs.get(k)
            if isinstance(dst, torch.Tensor) != isinstance(src, torch.Tensor):
                return False
            if isinstance(dst, torch.Tensor):
                if dst.shape != src.shape or dst.dtype != src.dtype:
                    return False
            elif dst != src:
                return False
        return True

    def __call__(self, *args, **kwargs):
        # Replaying inside an active stream capture would corrupt the outer
        # graph; run eagerly so the caller can complete its capture.
        if torch.cuda.is_current_stream_capturing():
            return self._fn(*args, **kwargs)
        if not self._matches(args, kwargs):
            if not self._warned:
                self._warned = True
                logger.warning("[cudagraph] replay args diverged from capture; running eagerly")
            return self._fn(*args, **kwargs)
        for dst, src in zip(self._static_args, args):
            if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
                dst.copy_(src, non_blocking=True)
        for k, dst in self._static_kwargs.items():
            src = kwargs.get(k)
            if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
                dst.copy_(src, non_blocking=True)
        self._graph.replay()
        out = self._static_out
        # The clone keeps outputs safe to hold across ticks; production servers
        # return the static buffer and require consumption before next replay.
        if isinstance(out, torch.Tensor):
            return out.clone()
        if isinstance(out, tuple):
            return tuple(o.clone() if isinstance(o, torch.Tensor) else o for o in out)
        return out


def graph_capture(fn, sample_args=(), sample_kwargs=None, label="step"):
    """Return a graphed version of fn, or fn itself if capture is not possible."""
    sample_kwargs = sample_kwargs or {}
    try:
        with torch.inference_mode(False), torch.no_grad():
            g = _GraphedCallable(fn, sample_args, sample_kwargs)
        logger.info("[cudagraph] captured %s", label)
        return g
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cudagraph] capture failed for %s (%s); staying eager", label, exc)
        return fn
