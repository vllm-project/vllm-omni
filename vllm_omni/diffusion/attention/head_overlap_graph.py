# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPUGraph capture for the head-split Ulysses ping-pong region.

The head-split overlap trades launch count for overlap: it turns one all-to-all
into ``groups`` of them so group N's attention can run against group N+1's
collective. That is the right shape for a host-bound step, but the extra
launches are exactly what a host-bound step cannot afford -- measured on A3, the
unsplit path at 45 s/step became 66-79 s/step with a 7-way split.

Capture is what makes the split pay: the same multi-stream skeleton replays as
one launch instead of ~4*groups, while the streams still overlap on the device.

Two properties of the region force the design here:

* **The graph is only valid for one sparse/dense phase.** The RainFusion backend
  picks its kernel from the denoise step index (``start_step``/``end_step``), so
  a graph captured in the dense phase would silently replay the dense kernel
  after the switch. The cache key therefore carries a phase tag the backend
  supplies, and a phase change simply captures a second graph.
* **Failure must be survivable.** Capturing an unfamiliar region can fail for
  reasons that only show up on device. A failed capture disables the cache and
  falls back to the eager path for good, instead of taking the process down
  mid-request the way a hard capture assertion would.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


_PROCESS_POOL = None


def _process_graph_pool():
    """One pool for every layer's graph.

    Each captured region must keep its own static inputs and intermediates
    alive, so the pool only bounds the *arena*; a per-layer pool would reserve
    a fresh arena per layer on top of that.
    """
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        _PROCESS_POOL = torch.npu.graph_pool_handle()
    return _PROCESS_POOL


def _supported() -> bool:
    npu = getattr(torch, "npu", None)
    return npu is not None and all(
        hasattr(npu, name) for name in ("NPUGraph", "graph", "graph_pool_handle", "synchronize")
    )


def _stream_is_capturing() -> bool:
    npu = getattr(torch, "npu", None)
    probe = getattr(npu, "is_current_stream_capturing", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except (RuntimeError, TypeError):
        return False


# A captured region retains its static inputs, its output and every intermediate
# allocated inside it, so one graph per attention layer is unaffordable: on A3
# the region's Q/K/V are full-packed-sequence sized (348 MiB each for 25472
# tokens), which came to ~2 GiB retained per layer and OOM'd on layer 12 of 50.
# The region runs after the projections, so it holds no weights, reads no
# metadata tensor and depends on nothing but its inputs and the dense/sparse
# phase -- which is what lets every layer share one graph (verified bit-exact on
# device: 8 ranks, 7 groups, 4 distinct data sets, all 0.000e+00).
_SHARED_CACHE: HeadOverlapGraphCache | None = None

# Deliberately pessimistic. A first cut at 4x/2 GiB let a capture through on a
# card with ~6 GiB device-free, and it still exhausted the device mid-capture:
# the failure cost the whole request, because an OOM inside a capture poisons
# the stream rather than falling back cleanly. Underestimating here is far more
# expensive than declining to capture, so the bar is set where a card that is
# merely "not empty" declines rather than tries.
_RETAINED_BYTES_PER_INPUT_BYTE = 8
_CAPTURE_SAFETY_BYTES = 6 * 2**30


def get_shared_cache(name: str = "head-overlap", max_graphs: int = 4) -> HeadOverlapGraphCache:
    """Process-wide cache. Layers with the same key replay the same graph."""
    global _SHARED_CACHE
    if _SHARED_CACHE is None:
        _SHARED_CACHE = HeadOverlapGraphCache(name=name, max_graphs=max_graphs)
    return _SHARED_CACHE


class HeadOverlapGraphCache:
    """Capture ``compute(q, k, v) -> out`` per key and replay it.

    Keys are ``(shape, dtype, groups, phase)``. Sharing one instance across
    layers is the point of the class: see ``get_shared_cache``.
    """

    def __init__(self, name: str, max_graphs: int = 4) -> None:
        self.name = name
        self.max_graphs = max(0, int(max_graphs))
        self._graphs: dict[tuple, tuple[Any, list[torch.Tensor], torch.Tensor]] = {}
        self._failed = False
        self._budget_exhausted = False
        self._pool = None
        self._replays = 0

    @property
    def captures(self) -> int:
        return len(self._graphs)

    def run(
        self,
        key: tuple,
        inputs: Sequence[torch.Tensor],
        compute: Callable[[Sequence[torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Return the region output, or None when the caller must run eagerly.

        The first call for a key runs ``compute`` eagerly (which also primes the
        lazy kernels and allocator state the capture needs) and captures the
        graph behind it. Later calls copy into the captured static inputs and
        replay.
        """
        if self._failed or self.max_graphs == 0 or not _supported() or _stream_is_capturing():
            return None

        entry = self._graphs.get(key)
        if entry is not None:
            graph, static_inputs, static_output = entry
            with torch.inference_mode():
                for static, current in zip(static_inputs, inputs, strict=True):
                    static.copy_(current)
                graph.replay()
                self._replays += 1
                if self._replays == 1:
                    logger.info("%s: first NPUGraph replay", self.name)
                return static_output.clone()

        if len(self._graphs) >= self.max_graphs:
            # Shapes are stable per request, so reaching the cap means something
            # unexpected is varying; stop growing rather than thrash.
            logger.warning_once(
                "%s: reached the %d-graph cap; new shapes take the eager path.", self.name, self.max_graphs
            )
            return None

        if not self._budget_allows(inputs):
            return None

        return self._capture(key, inputs, compute)

    def _budget_allows(self, inputs: Sequence[torch.Tensor]) -> bool:
        """Skip capture when the graph would not fit, instead of OOM-ing on it.

        A capture is an all-or-nothing memory commitment, and failing it does not
        merely fall back: on this stack the OOM surfaces as a stream error that
        takes the request and the worker with it. So the budget is checked before
        the first allocation, and the estimate is intentionally pessimistic --
        declining to capture costs only the overlap's launch saving, while
        guessing wrong costs the request.
        """
        if self._budget_exhausted:
            return False
        try:
            free, _total = torch.npu.mem_get_info()
        except Exception:  # noqa: BLE001 - no budget information, no capture
            return False
        needed = _RETAINED_BYTES_PER_INPUT_BYTE * sum(value.numel() * value.element_size() for value in inputs)
        required = needed + _CAPTURE_SAFETY_BYTES
        if required > free:
            self._budget_exhausted = True
            logger.warning(
                "%s: skipping NPUGraph capture — an estimated %.1f GiB is needed but only %.1f GiB is free; "
                "the head-split region keeps its eager path.",
                self.name,
                required / 2**30,
                free / 2**30,
            )
            return False
        return True

    def _capture(self, key: tuple, inputs: Sequence[torch.Tensor], compute) -> torch.Tensor | None:
        try:
            static_inputs = [value.detach().clone() for value in inputs]
            torch.npu.synchronize()
            # Prime lazy kernels and allocator state before the capture; a
            # first-touch allocation inside the capture invalidates it.
            warmup = compute([value.detach() for value in static_inputs])
            torch.npu.synchronize()

            graph = torch.npu.NPUGraph()
            pool = _process_graph_pool()
            with torch.inference_mode(), torch.npu.graph(graph, pool=pool):
                static_output = compute([value.detach() for value in static_inputs])
            torch.npu.synchronize()
        except Exception as exc:  # noqa: BLE001 - the fallback is the point
            self._failed = True
            logger.warning(
                "%s: NPUGraph capture failed (%s: %s); the head-split region keeps its eager path "
                "for the rest of this process.",
                self.name,
                type(exc).__name__,
                exc,
            )
            return None

        self._graphs[key] = (graph, static_inputs, static_output)
        logger.info("%s: captured NPUGraph %d/%d for key=%s", self.name, len(self._graphs), self.max_graphs, key)
        return warmup
