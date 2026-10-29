# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""XPU device graphs for keyword-only module forwards.

``GraphedForward`` (``models/taomate_h3/cuda_graph.py``) stays the CUDA
implementation. This subclass keeps its call keying, static buffers, replay,
LRU and rank agreement, and replaces only the capture:

- warm-up forwards run on a ``torch.xpu`` side stream;
- capture goes through ``current_omni_platform.capture_graph``, which gives
  every graph a private pool: on torch 2.14+xpu, capturing into a shared pool
  after another graph in it was freed hits an XPUCachingAllocator assert;
- there is no first-capture probe, since XPU has no sync debug mode. A capture
  that fails still turns graphs off on every rank of ``group``.
"""

from __future__ import annotations

import traceback
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.utils.torch_utils import supports_xpu_graph

from vllm_omni.diffusion.models.taomate_h3.cuda_graph import (
    GraphedForward,
    _adaln_cache_bypassed,
    _Entry,
    clone_static,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class XPUGraphedForward(GraphedForward):
    """``GraphedForward`` that captures XPU graphs."""

    def __init__(self, module: nn.Module, *, enabled: bool = True, **kwargs: Any) -> None:
        super().__init__(module, enabled=enabled, **kwargs)
        # The CUDA base turns itself off on any other device type.
        self.enabled = bool(enabled) and self.device.type == "xpu" and supports_xpu_graph()
        self._probe_pending = False

    def disable(self, reason: str) -> None:
        if self.enabled:
            logger.warning("%s XPU graphs disabled: %s", self.name, reason)
        self.enabled = False
        self.clear()

    def _capture(
        self,
        key: Any,
        kwargs: dict[str, Any],
        keep_alive: tuple[Any, ...] = (),
        variant: Any = None,
        *,
        pin: bool = False,
    ) -> _Entry | None:
        failure: str | None = None
        static: dict[str, Any] = {}
        graph: Any = None
        outputs: Any = None
        try:
            static = clone_static(kwargs)
            with _adaln_cache_bypassed(self.module):
                side = torch.xpu.Stream(device=self.device)
                side.wait_stream(torch.xpu.current_stream(self.device))
                # The first capture warms lazy kernel/library state for the
                # whole process; later shapes need a single warm forward.
                warmups = self.warmup_iters if self.captures == 0 else min(self.warmup_iters, 1)
                with torch.xpu.stream(side):
                    for _ in range(warmups):
                        self.module(**static)
                torch.xpu.current_stream(self.device).wait_stream(side)
                captured: list[Any] = []
                graph = current_omni_platform.capture_graph(lambda: captured.append(self.module(**static)))
                if graph is None:
                    raise RuntimeError(f"{type(current_omni_platform).__name__} cannot capture XPU graphs")
                outputs = captured[0]
        except Exception as exc:  # noqa: BLE001 - any capture failure means eager from now on
            failure = f"{type(exc).__name__}: {exc}"
            logger.warning("%s XPU graph capture failed:\n%s", self.name, traceback.format_exc())
        ok = self._agree(failure is None)
        if not ok:
            del graph, outputs, static
            self.disable(failure or "a peer rank failed to capture")
            return None
        if len(self._entries) >= self.max_entries:
            victim = next((k for k, e in self._entries.items() if not e.pinned), None)
            if victim is None:
                if not self._pinned_overflow_warned:
                    self._pinned_overflow_warned = True
                    logger.warning(
                        "%s: all %d resident XPU graphs are pinned; growing past max_entries",
                        self.name,
                        len(self._entries),
                    )
            else:
                del self._entries[victim]
                self.evictions += 1
        entry = _Entry(graph=graph, static_kwargs=static, outputs=outputs, keep_alive=tuple(keep_alive), pinned=pin)
        self._entries[key] = entry
        self.captures += 1
        logger.info("%s XPU graph captured for %s (%d resident)", self.name, variant, len(self._entries))
        return entry
