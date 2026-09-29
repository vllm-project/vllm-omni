# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Runner-owned lifecycle for model-declared model-local CUDA Graph Components."""

from __future__ import annotations

import logging
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import torch
from tqdm import tqdm
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import VllmConfig
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.interfaces.model_local_cudagraph import (
    ModelLocalCaptureMode,
    ModelLocalCUDAGraphComponent,
    ModelLocalCUDAGraphDescriptor,
    ModelLocalGraphHandle,
    ModelLocalRuntimeResolution,
    SupportsModelLocalCUDAGraph,
)

logger = logging.getLogger(__name__)

MAX_LOG_ITEMS = 100
LOG_EVERY_CALLS = 100


@dataclass
class ModelLocalCUDAGraphEntry:
    """Worker-owned resources for one captured Component Descriptor."""

    descriptor: ModelLocalCUDAGraphDescriptor
    graph: torch.cuda.CUDAGraph
    buffers: object
    captured_output: object


_COMPONENT_POLICY_KEYS = frozenset({"max_extra_graphs"})


def clone_tensor_tree(value: object) -> object:
    """Clone tensor leaves after model-specific graph output processing."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        cloned = [clone_tensor_tree(item) for item in value]
        if hasattr(value, "_fields"):
            constructor = type(value)
            return constructor(*cast(tuple[Any, ...], tuple(cloned)))
        return tuple(cloned)
    if isinstance(value, list):
        return [clone_tensor_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tensor_tree(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ModelLocalComponentRuntimeConfig:
    max_extra_graphs: int = 0


@dataclass
class ManagedComponent:
    component: ModelLocalCUDAGraphComponent
    entries: OrderedDict[ModelLocalCUDAGraphDescriptor, ModelLocalCUDAGraphEntry]
    capture_mode: ModelLocalCaptureMode
    max_graphs: int | None = None


class _NoOpRecorder:
    __slots__ = ()

    def record_graph_hit(self, resolution: ModelLocalRuntimeResolution) -> None:
        del resolution

    def record_fallback(self, resolution: ModelLocalRuntimeResolution) -> None:
        del resolution

    def record_replay_error(self, resolution: ModelLocalRuntimeResolution) -> None:
        del resolution


class _ComponentRecorder:
    __slots__ = ("_sink", "_component_id")

    def __init__(self, sink: ModelLocalGraphStatsSink, component_id: str) -> None:
        self._sink = sink
        self._component_id = component_id

    def record_graph_hit(self, resolution: ModelLocalRuntimeResolution) -> None:
        self._sink.record("hit", self._component_id, resolution)

    def record_fallback(self, resolution: ModelLocalRuntimeResolution) -> None:
        self._sink.record("fallback", self._component_id, resolution)

    def record_replay_error(self, resolution: ModelLocalRuntimeResolution) -> None:
        self._sink.record("replay_error", self._component_id, resolution)


class ModelLocalGraphStatsSink:
    """Manager-read, Recorder-write runtime counters."""

    def __init__(self, *, enabled: bool, max_log_items: int = MAX_LOG_ITEMS) -> None:
        if max_log_items <= 0:
            raise ValueError("max_log_items must be positive")
        self.enabled = enabled
        self.max_log_items = max_log_items
        self._calls: Counter[str] = Counter()
        self._outcomes: Counter[tuple[str, str]] = Counter()
        # Counts graph specializations actually selected for replay.
        self._descriptors: OrderedDict[tuple[str, object], int] = OrderedDict()
        # Counts runtime keys observed before Descriptor bucket selection.
        self._runtime_keys: OrderedDict[tuple[str, object], int] = OrderedDict()

    def recorder_for(self, component_id: str) -> _ComponentRecorder | _NoOpRecorder:
        if not self.enabled:
            return _NoOpRecorder()
        return _ComponentRecorder(self, component_id)

    def record(
        self,
        outcome: str,
        component_id: str,
        resolution: ModelLocalRuntimeResolution,
    ) -> None:
        self._calls[component_id] += 1
        self._outcomes[(component_id, outcome)] += 1
        if resolution.descriptor is not None:
            self._record_detail(self._descriptors, (component_id, resolution.descriptor.variant))
        key = (component_id, resolution.runtime_key.variant)
        self._record_detail(self._runtime_keys, key)
        total_calls = sum(self._calls.values())
        if total_calls % LOG_EVERY_CALLS == 0:
            logger.info("Model-local CUDA Graph runtime stats after %d calls: %s", total_calls, self.snapshot())

    def _record_detail(self, items: OrderedDict[tuple[str, object], int], key: tuple[str, object]) -> None:
        if key not in items and len(items) == self.max_log_items:
            items.popitem(last=False)
        items[key] = items.get(key, 0) + 1
        items.move_to_end(key)

    def snapshot(self) -> dict[str, object]:
        return {
            "calls": dict(self._calls),
            "outcomes": dict(self._outcomes),
            "descriptors": dict(self._descriptors),
            "runtime_keys": dict(self._runtime_keys),
        }


class ModelLocalCUDAGraphManager:
    """Consumes resolved Components and owns capture/bind/restore lifecycle."""

    def __init__(self, *, vllm_config: VllmConfig, device: torch.device) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.components: tuple[ModelLocalCUDAGraphComponent, ...] = ()
        self.managed_components: dict[str, ManagedComponent] = {}
        self._component_configs: dict[str, ModelLocalComponentRuntimeConfig] = {}
        self._runtime_capture_stream: torch.cuda.Stream | None = None
        self._prepared = False
        self._capture_attempted = False

        raw_config = getattr(vllm_config.model_config, "model_local_cudagraph", None)
        if raw_config is None:
            raw_config = {}
        if not isinstance(raw_config, Mapping):
            raise TypeError("model_local_cudagraph must be a mapping")
        self.config = dict(raw_config)
        log_stats = bool(getattr(getattr(vllm_config, "observability_config", None), "cudagraph_metrics", False))
        self.stats_sink = ModelLocalGraphStatsSink(enabled=log_stats)

    def _synchronized_free_memory(self) -> int:
        if self.device.type == "cpu":
            return 0
        torch.accelerator.synchronize()
        return int(torch.accelerator.get_memory_info()[0])

    def _validate_nonnegative(self, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"{value!r} must be a non-negative integer")
        return value

    def prepare(self, model: SupportsModelLocalCUDAGraph) -> None:
        if self._prepared:
            raise RuntimeError("ModelLocalCUDAGraphManager.prepare() called more than once")

        components = tuple(model.get_model_local_cudagraph_components())
        component_by_id: dict[str, ModelLocalCUDAGraphComponent] = {}
        for component in components:
            if not isinstance(component, ModelLocalCUDAGraphComponent):
                raise TypeError(
                    "get_model_local_cudagraph_components() must return ModelLocalCUDAGraphComponent objects"
                )
            if not component.component_id:
                raise ValueError("Model-local CUDA Graph component_id must not be empty")
            if component.component_id in component_by_id:
                raise ValueError(f"Duplicate model-local CUDA Graph component_id: {component.component_id}")
            try:
                if len(set(component.descriptors)) != len(component.descriptors):
                    raise ValueError(f"Duplicate Descriptor in Component {component.component_id}")
            except TypeError as exc:
                raise TypeError(f"Descriptors for Component {component.component_id} must be hashable") from exc
            component_by_id[component.component_id] = component
            if not component.descriptors:
                logger.info(
                    "Model-local CUDA Graph Component %s is known but has no startup "
                    "Descriptors for the resolved model configuration",
                    component.component_id,
                )

        unknown_component_ids = set(self.config) - set(component_by_id)
        if unknown_component_ids:
            names = ", ".join(sorted(str(name) for name in unknown_component_ids))
            raise ValueError(f"Unknown model-local CUDA Graph component override(s): {names}")

        for component_id, raw_component in self.config.items():
            component = component_by_id[component_id]
            if raw_component is None:
                raw_component = {}
            if not isinstance(raw_component, Mapping):
                raise TypeError(f"model_local_cudagraph.{component_id} must be a mapping")
            unknown_keys = set(raw_component) - _COMPONENT_POLICY_KEYS - set(component.supported_config_keys)
            if unknown_keys:
                names = ", ".join(sorted(unknown_keys))
                raise ValueError(f"Unknown config key(s) for model-local Component {component_id}: {names}")
            mode = component.capture_mode
            if mode is ModelLocalCaptureMode.PRECAPTURE and "max_extra_graphs" in raw_component:
                raise ValueError(f"max_extra_graphs requires a lazy capture mode for Component {component_id}")
            if mode is ModelLocalCaptureMode.PURE_LAZY and not component.descriptors:
                raise ValueError(f"Pure lazy Component {component_id} needs descriptors for memory profiling")
            max_extra = self._validate_nonnegative(raw_component.get("max_extra_graphs", 0))
            if mode.allows_lazy_capture and max_extra == 0:
                logger.warning(
                    "Model-local CUDA Graph Component %s has unbounded lazy capture; memory profiling cannot reserve "
                    "for every future runtime Descriptor",
                    component_id,
                )
            self._component_configs[component_id] = ModelLocalComponentRuntimeConfig(
                max_extra_graphs=max_extra,
            )

        self.components = components
        self._prepared = True
        logger.info(
            "Prepared runner-owned model-local CUDA Graph Components: %s",
            [component.component_id for component in components],
        )

    def capture_entry(
        self,
        component: ModelLocalCUDAGraphComponent,
        descriptor: ModelLocalCUDAGraphDescriptor,
        *,
        graph_pool: object | None = None,
    ) -> ModelLocalCUDAGraphEntry | None:
        if not component.validate_descriptor(descriptor):
            return None
        routine = component.routine
        buffers = routine.allocate_buffers(descriptor, self.device)
        num_warmups = max(
            1,
            int(getattr(self.vllm_config.compilation_config, "cudagraph_num_of_warmups", 0)),
        )
        for _ in range(num_warmups):
            with routine.capture_context(descriptor, buffers):
                routine.forward_for_capture(buffers)
        capture_stream = torch.cuda.current_stream(self.device)
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with routine.capture_context(descriptor, buffers):
                with (
                    torch.inference_mode(),
                    torch.cuda.graph(
                        graph,
                        pool=(current_platform.get_global_graph_pool() if graph_pool is None else graph_pool),
                        stream=capture_stream,
                    ),
                ):
                    captured_output = routine.forward_for_capture(buffers)
        except BaseException:
            graph.reset()
            raise
        return ModelLocalCUDAGraphEntry(
            descriptor=descriptor,
            graph=graph,
            buffers=buffers,
            captured_output=captured_output,
        )

    def profile_memory(self) -> int:
        """Estimate startup graph memory with throwaway captures."""
        if not self._prepared:
            raise RuntimeError("ModelLocalCUDAGraphManager must be prepared before profiling")

        profiling_pool = current_platform.graph_pool_handle()
        captured: list[ModelLocalCUDAGraphEntry] = []
        estimate = 0
        try:
            for component in self.components:
                component_config = self._component_configs.get(component.component_id)
                if component_config is None or not component.descriptors:
                    continue
                samples: list[int] = []
                for descriptor in component.capture_descriptors[:2]:
                    free_before = self._synchronized_free_memory()
                    entry = self.capture_entry(component, descriptor, graph_pool=profiling_pool)
                    if entry is None:
                        continue
                    free_after = self._synchronized_free_memory()
                    captured.append(entry)
                    samples.append(max(0, free_before - free_after))
                if not samples:
                    continue
                first_capture = samples[0]
                per_graph = max(samples[1] if len(samples) > 1 else 0, 1 << 20)
                if component.capture_mode is ModelLocalCaptureMode.PURE_LAZY:
                    extra_graphs = max(0, component_config.max_extra_graphs - 1)
                else:
                    lazy_graphs = component_config.max_extra_graphs if component.capture_mode.allows_lazy_capture else 0
                    extra_graphs = len(component.capture_descriptors) - 1 + lazy_graphs
                estimate += first_capture + per_graph * extra_graphs
                logger.debug(
                    "Estimated model-local Component %s CUDA graph memory: "
                    "%.2f MiB first-capture + %d x %.2f MiB per-graph",
                    component.component_id,
                    first_capture / (1 << 20),
                    extra_graphs,
                    per_graph / (1 << 20),
                )
        finally:
            for entry in captured:
                self._destroy_entry(entry)
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
        logger.info("Estimated runner-owned vocoder CUDA graph memory: %.2f MiB", estimate / (1 << 20))
        return estimate

    def _capture_and_register(
        self,
        managed: ManagedComponent,
        descriptor: ModelLocalCUDAGraphDescriptor,
    ) -> ModelLocalCUDAGraphEntry | None:
        existing = managed.entries.get(descriptor)
        if existing is not None:
            return existing

        entry = self.capture_entry(managed.component, descriptor)
        if entry is not None:
            managed.entries[descriptor] = entry
        return entry

    @staticmethod
    def _destroy_entry(entry: ModelLocalCUDAGraphEntry) -> None:
        entry.graph.reset()
        entry.buffers = None
        entry.captured_output = None

    def _available_descriptors(
        self,
        managed: ManagedComponent,
    ) -> frozenset[ModelLocalCUDAGraphDescriptor]:
        return frozenset(managed.entries.keys())

    @contextmanager
    def _runtime_capture_scope(self):
        if self._runtime_capture_stream is None:
            self._runtime_capture_stream = torch.cuda.Stream(device=self.device)
        caller = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(caller)
        self._runtime_capture_stream.wait_event(ready)
        set_cudagraph_capturing_enabled(True)
        try:
            with torch.cuda.stream(self._runtime_capture_stream):
                yield
                complete = torch.cuda.Event()
                complete.record(self._runtime_capture_stream)
            caller.wait_event(complete)
        finally:
            set_cudagraph_capturing_enabled(False)

    def _runtime_capture_and_register(
        self,
        managed: ManagedComponent,
        descriptor: ModelLocalCUDAGraphDescriptor,
    ) -> ModelLocalCUDAGraphEntry | None:
        if not managed.capture_mode.allows_lazy_capture:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        existing = managed.entries.get(descriptor)
        if existing is not None:
            return existing
        if managed.max_graphs is not None and len(managed.entries) >= managed.max_graphs:
            return None
        with self._runtime_capture_scope():
            entry = self._capture_and_register(managed, descriptor)
        if entry is not None:
            logger.info(
                "Lazy-captured model-local CUDA Graph Component %s Descriptor %r",
                managed.component.component_id,
                descriptor,
            )
        return entry

    def _make_runtime_miss_handler(
        self,
        managed: ManagedComponent,
    ) -> Callable[[ModelLocalRuntimeResolution], ModelLocalCUDAGraphEntry | None]:
        if not managed.capture_mode.allows_lazy_capture:
            return lambda resolution: None

        def on_runtime_miss(resolution: ModelLocalRuntimeResolution) -> ModelLocalCUDAGraphEntry | None:
            descriptor = resolution.descriptor
            if descriptor is None:
                descriptor = managed.component.routine.make_lazy_descriptor(resolution.runtime_key)
            if descriptor is None:
                return None
            return self._runtime_capture_and_register(managed, descriptor)

        return on_runtime_miss

    def _build_runtime_callable(
        self,
        managed: ManagedComponent,
        recorder: _ComponentRecorder | _NoOpRecorder,
    ) -> Callable[..., Any]:
        component = managed.component
        entries = MappingProxyType(managed.entries)
        routine = component.routine
        on_runtime_miss = self._make_runtime_miss_handler(managed)
        clone_output = component.clone_output

        def runtime_callable(*args: Any, **kwargs: Any) -> Any:
            routine.validate_runtime_inputs(args, kwargs)
            resolution = routine.resolve_runtime(args, kwargs, self._available_descriptors(managed))
            entry = entries.get(resolution.descriptor) if resolution.descriptor is not None else None
            if entry is None:
                entry = on_runtime_miss(resolution)
            if entry is None:
                recorder.record_fallback(resolution)
                return routine.eager_call(*args, **kwargs)

            descriptor = entry.descriptor
            graph_resolution = (
                resolution
                if resolution.descriptor == descriptor
                else ModelLocalRuntimeResolution(
                    runtime_key=resolution.runtime_key,
                    descriptor=descriptor,
                )
            )
            try:
                routine.copy_runtime_inputs(args, kwargs, entry.buffers)
                entry.graph.replay()
                output = routine.output_after_replay(args, kwargs, entry.buffers, entry.captured_output)
                if clone_output:
                    output = clone_tensor_tree(output)
            except Exception:
                recorder.record_replay_error(graph_resolution)
                raise
            recorder.record_graph_hit(graph_resolution)
            return output

        return runtime_callable

    def capture_and_bind(self) -> int:
        # Capture every Component before binding any Handle. During capture,
        # calls through other Components must stay eager; after binding, a
        # unified graph's eager_call fallback can invoke segmented graph
        # Components and interleave them with uncaptured eager operations.
        if not self._prepared:
            raise RuntimeError("ModelLocalCUDAGraphManager must be prepared before capture")
        if self._capture_attempted:
            raise RuntimeError("Model-local CUDA Graph capture has already been attempted")
        self._capture_attempted = True

        capture_start = time.perf_counter()
        free_before = self._synchronized_free_memory()
        prepared: dict[str, ManagedComponent] = {}
        captured: dict[str, ManagedComponent] = {}
        active: dict[str, ManagedComponent] = {}
        selected = [component for component in self.components if component.component_id in self._component_configs]
        for component in selected:
            if component._bound_handle is not None:
                raise RuntimeError(f"Component already bound before capture: {component.component_id}")
        try:
            # Prepare every Component before capturing any of them.
            for component in selected:
                prepared[component.component_id] = ManagedComponent(
                    component=component,
                    entries=OrderedDict(),
                    capture_mode=component.capture_mode,
                )

            for component_id, managed in prepared.items():
                component = managed.component
                component_config = self._component_configs[component_id]
                capture_descriptors = (
                    component.capture_descriptors
                    if component.capture_mode is not ModelLocalCaptureMode.PURE_LAZY
                    else ()
                )
                progress = (
                    tqdm(
                        total=len(capture_descriptors),
                        desc=f"Capture {component.component_id}",
                        unit="graph",
                        leave=True,
                    )
                    if capture_descriptors
                    else None
                )
                try:
                    for descriptor in capture_descriptors:
                        try:
                            self._capture_and_register(managed, descriptor)
                        finally:
                            assert progress is not None
                            progress.update(1)
                finally:
                    if progress is not None:
                        progress.close()
                logger.info(
                    "Model-local CUDA Graph Component %s captured %d/%d startup Descriptors",
                    component.component_id,
                    len(managed.entries),
                    len(capture_descriptors),
                )
                # Zero means no runtime graph-count limit, not zero lazy slots.
                managed.max_graphs = (
                    None
                    if managed.capture_mode.allows_lazy_capture and component_config.max_extra_graphs == 0
                    else len(managed.entries) + component_config.max_extra_graphs
                )
                if not managed.entries and not managed.capture_mode.allows_lazy_capture:
                    continue
                captured[component_id] = managed

            for component_id, managed in captured.items():
                component = managed.component
                runtime_callable = self._build_runtime_callable(
                    managed,
                    self.stats_sink.recorder_for(component_id),
                )
                component._bind_handle(
                    ModelLocalGraphHandle(
                        runtime_callable,
                        lambda managed=managed: self._available_descriptors(managed),
                    )
                )
                active[component_id] = managed
        except BaseException:
            for managed in prepared.values():
                managed.component._restore_eager()
                for entry in managed.entries.values():
                    self._destroy_entry(entry)
                managed.entries.clear()
            raise

        self.managed_components = active
        free_after = self._synchronized_free_memory()
        captured_memory = max(0, free_before - free_after)
        logger.info(
            "Model-local CUDA Graph capture finished in %.2fs, bound=%s, memory=%.2f MiB",
            time.perf_counter() - capture_start,
            list(active),
            captured_memory / (1 << 20),
        )
        return captured_memory

    def clear(self) -> None:
        for managed in self.managed_components.values():
            managed.component._restore_eager()
            for entry in managed.entries.values():
                self._destroy_entry(entry)
            managed.entries.clear()
        self.managed_components.clear()
        self._runtime_capture_stream = None
        if self.stats_sink.enabled:
            logger.info("Model-local CUDA Graph runtime stats: %s", self.stats_sink.snapshot())
