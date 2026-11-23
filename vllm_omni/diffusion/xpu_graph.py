# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""XPU device graphs for keyword-only module forwards.

``GraphedForward`` (``cuda_graph.py``) stays the CUDA implementation. This
subclass keeps its call keying, static buffers, replay, LRU and rank agreement,
and overrides only the device hooks:

- warm-up forwards run on a ``torch.xpu`` side stream;
- capture goes through ``current_omni_platform.capture_graph``, which gives
  every graph a private pool: on torch 2.14+xpu, capturing into a shared pool
  after another graph in it was freed hits an XPUCachingAllocator assert;
- there is no first-capture probe, since XPU has no sync debug mode. A capture
  that fails still turns graphs off on every rank of ``group``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from vllm.utils.torch_utils import supports_xpu_graph

from vllm_omni.diffusion.cuda_graph import GraphedForward
from vllm_omni.platforms import current_omni_platform


class XPUGraphedForward(GraphedForward):
    """``GraphedForward`` that captures XPU graphs."""

    graph_kind = "XPU"
    _sync_probe = False

    def _graphs_supported(self) -> bool:
        return self.device.type == "xpu" and supports_xpu_graph()

    @contextmanager
    def _warmup_stream(self) -> Iterator[None]:
        side = torch.xpu.Stream(device=self.device)
        side.wait_stream(torch.xpu.current_stream(self.device))
        with torch.xpu.stream(side):
            yield
        torch.xpu.current_stream(self.device).wait_stream(side)

    def _capture_graph(self, body: Callable[[], Any]) -> tuple[Any, Any]:
        captured: list[Any] = []
        graph = current_omni_platform.capture_graph(lambda: captured.append(body()))
        if graph is None:
            raise RuntimeError(f"{type(current_omni_platform).__name__} cannot capture XPU graphs")
        return graph, captured[0]
