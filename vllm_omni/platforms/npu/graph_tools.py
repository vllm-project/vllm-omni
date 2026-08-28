# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable exact-signature NPUGraph capture and replay helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class _ReplayGraph(Protocol):
    def replay(self) -> None: ...


@dataclass
class CapturedDeviceGraph:
    graph: _ReplayGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_outputs: tuple[torch.Tensor, ...]

    @property
    def tensor_workspace_bytes(self) -> int:
        """Host-computable size of graph-owned static tensor storage."""
        return sum(value.numel() * value.element_size() for value in (*self.static_inputs, *self.static_outputs))

    def replay(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            for static, current in zip(self.static_inputs, inputs, strict=True):
                static.copy_(current)
            self.graph.replay()
            # Persistent graph outputs are overwritten by the next replay.
            # Clone before they become request-owned streaming cache entries.
            return tuple(output.detach().clone() for output in self.static_outputs)


@dataclass(frozen=True)
class GraphRunInfo:
    """Selection metadata for one exact-graph invocation."""

    mode: str
    reason: str | None
    workspace_bytes: int


def _tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), str(value.device)


def _graph_workspace_bytes(graph: object) -> int:
    return int(getattr(graph, "tensor_workspace_bytes", 0))


class NPUExactGraphRunner:
    """Capture and replay tensor-only functions for exact NPU signatures."""

    def __init__(
        self,
        *,
        max_graphs: int = 32,
        component_name: str = "device graph",
        disable_config_hint: str = "disable graph capture",
    ) -> None:
        self.max_graphs = max(0, int(max_graphs))
        self.component_name = component_name
        self.disable_config_hint = disable_config_hint
        self._enabled = self.max_graphs > 0
        self._graphs: dict[tuple[object, ...], CapturedDeviceGraph] = {}
        self._failed_keys: set[tuple[object, ...]] = set()
        self._hits = 0
        self._misses = 0
        self._fallbacks = 0
        self._graph_pool: object | None = None

    @staticmethod
    def is_supported() -> bool:
        npu = getattr(torch, "npu", None)
        return npu is not None and all(
            hasattr(npu, name)
            for name in (
                "NPUGraph",
                "graph",
                "is_current_stream_capturing",
                "synchronize",
            )
        )

    @staticmethod
    def _stream_is_capturing() -> bool:
        npu = getattr(torch, "npu", None)
        is_capturing = getattr(npu, "is_current_stream_capturing", None)
        if not callable(is_capturing):
            return False
        try:
            return bool(is_capturing())
        except (RuntimeError, TypeError):
            return False

    def _eligible(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        return (
            self._enabled
            and bool(inputs)
            and all(value.device.type == "npu" for value in inputs)
            and self.is_supported()
            and not self._stream_is_capturing()
        )

    def _ineligible_reason(self, inputs: tuple[torch.Tensor, ...]) -> str | None:
        if not self._enabled:
            return "disabled"
        if not inputs:
            return "empty_inputs"
        if not all(value.device.type == "npu" for value in inputs):
            return "non_npu"
        if not self.is_supported():
            return "unsupported_api"
        if self._stream_is_capturing():
            return "nested_capture"
        return None

    @property
    def stats(self) -> dict[str, int]:
        return {
            "captures": len(self._graphs),
            "failed": len(self._failed_keys),
            "hits": self._hits,
        }

    @property
    def telemetry(self) -> dict[str, int]:
        """Extended Host-only counters without changing the legacy stats API."""
        return {
            **self.stats,
            "misses": self._misses,
            "fallbacks": self._fallbacks,
            "workspace_bytes": sum(_graph_workspace_bytes(graph) for graph in self._graphs.values()),
        }

    def capture(
        self,
        inputs: tuple[torch.Tensor, ...],
        compute: Callable[..., tuple[torch.Tensor, ...]],
    ) -> CapturedDeviceGraph:
        npu = torch.npu
        static_inputs = tuple(value.detach().clone() for value in inputs)
        npu.synchronize()
        graph = npu.NPUGraph()
        if self._graph_pool is None:
            from vllm.platforms import current_platform

            self._graph_pool = current_platform.get_global_graph_pool()
        with torch.inference_mode(), npu.graph(graph, pool=self._graph_pool):
            static_outputs = compute(*static_inputs)
        npu.synchronize()
        return CapturedDeviceGraph(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
        )

    def run(
        self,
        operation: str,
        inputs: tuple[torch.Tensor, ...],
        constants: tuple[object, ...],
        compute: Callable[..., tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        outputs, _ = self.run_with_info(operation, inputs, constants, compute)
        return outputs

    def run_with_info(
        self,
        operation: str,
        inputs: tuple[torch.Tensor, ...],
        constants: tuple[object, ...],
        compute: Callable[..., tuple[torch.Tensor, ...]],
        *,
        fallback_compute: Callable[..., tuple[torch.Tensor, ...]] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], GraphRunInfo]:
        """Run an exact graph and expose capture/replay/fallback selection.

        ``fallback_compute`` is used only when graph execution is ineligible or
        the bounded cache is full. Capture failures remain fatal because the
        torch-npu allocator/capture state cannot be assumed reusable.
        """
        if self._failed_keys:
            raise RuntimeError(
                f"{self.component_name} cannot continue after a failed NPUGraph capture; "
                f"restart the stage process and {self.disable_config_hint} before retrying."
            )
        if not self._eligible(inputs):
            self._fallbacks += 1
            fallback = compute if fallback_compute is None else fallback_compute
            return fallback(*inputs), GraphRunInfo(
                mode="fallback",
                reason=self._ineligible_reason(inputs) or "ineligible",
                workspace_bytes=0,
            )

        key = (
            operation,
            constants,
            tuple(_tensor_signature(value) for value in inputs),
        )
        graph = self._graphs.get(key)
        if graph is not None:
            self._hits += 1
            if self._hits == 1:
                logger.info("%s started NPUGraph replay", self.component_name)
            return graph.replay(inputs), GraphRunInfo(
                mode="replay",
                reason=None,
                workspace_bytes=_graph_workspace_bytes(graph),
            )

        self._misses += 1
        if len(self._graphs) >= self.max_graphs:
            self._fallbacks += 1
            logger.warning_once(
                "%s reached the %d-entry NPUGraph limit; new tensor shapes will use eager execution.",
                self.component_name,
                self.max_graphs,
            )
            fallback = compute if fallback_compute is None else fallback_compute
            return fallback(*inputs), GraphRunInfo(
                mode="fallback",
                reason="graph_capacity",
                workspace_bytes=0,
            )

        # Prime lazy kernels and allocator state before capture. The current
        # call returns eager outputs; the next exact signature replays the graph.
        eager_outputs = compute(*inputs)
        try:
            self._graphs[key] = self.capture(inputs, compute)
        except Exception as exc:
            self._failed_keys.add(key)
            self._enabled = False
            logger.exception(
                "%s failed to capture NPUGraph for %s; the torch-npu "
                "allocator/RNG capture state may be invalid and this stage "
                "process must be restarted.",
                self.component_name,
                operation,
            )
            raise RuntimeError(
                f"{self.component_name} NPUGraph capture failed for {operation}; "
                f"restart the stage process. To run eagerly, {self.disable_config_hint}."
            ) from exc
        logger.info(
            "%s captured NPUGraph %d/%d for %s",
            self.component_name,
            len(self._graphs),
            self.max_graphs,
            operation,
        )
        captured = self._graphs[key]
        return eager_outputs, GraphRunInfo(
            mode="capture",
            reason="signature_miss",
            workspace_bytes=_graph_workspace_bytes(captured),
        )
