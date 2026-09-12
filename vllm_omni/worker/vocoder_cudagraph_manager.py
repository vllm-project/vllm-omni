# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Runner-owned lifecycle for model-declared vocoder CUDA Graph Components."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import torch
from tqdm import tqdm
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import VllmConfig
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.interfaces.vocoder_cudagraph import (
    SupportsVocoderCUDAGraph,
    VocoderCUDAGraphComponent,
    VocoderCUDAGraphDescriptor,
    VocoderGraphHandle,
    VocoderRuntimeResolution,
)

logger = logging.getLogger(__name__)


@dataclass
class VocoderCUDAGraphEntry:
    """Worker-owned resources for one captured Component Descriptor."""

    descriptor: VocoderCUDAGraphDescriptor
    graph: torch.cuda.CUDAGraph
    buffers: object
    captured_output: object
    replay_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


_FRAMEWORK_CONFIG_KEYS = frozenset(
    {
        "log_stats",
        "components",
    }
)
_COMPONENT_POLICY_KEYS = frozenset(
    {
        "enabled",
        "enable_lazy_capture",
        "max_extra_graphs",
    }
)


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
class VocoderComponentRuntimeConfig:
    enabled: bool = True
    enable_lazy_capture: bool = False
    max_extra_graphs: int = 0


@dataclass
class ManagedComponent:
    component: VocoderCUDAGraphComponent
    entries: OrderedDict[VocoderCUDAGraphDescriptor, VocoderCUDAGraphEntry]
    enable_lazy_capture: bool
    failed_descriptors: set[VocoderCUDAGraphDescriptor] = field(default_factory=set)
    max_graphs: int | None = None


class _NoOpRecorder:
    __slots__ = ()

    def record_graph_hit(self, resolution: VocoderRuntimeResolution) -> None:
        del resolution

    def record_fallback(self, resolution: VocoderRuntimeResolution) -> None:
        del resolution

    def record_replay_error(self, resolution: VocoderRuntimeResolution) -> None:
        del resolution


class _ComponentRecorder:
    __slots__ = ("_sink", "_component_id")

    def __init__(self, sink: VocoderGraphStatsSink, component_id: str) -> None:
        self._sink = sink
        self._component_id = component_id

    def record_graph_hit(self, resolution: VocoderRuntimeResolution) -> None:
        self._sink.record("hit", self._component_id, resolution)

    def record_fallback(self, resolution: VocoderRuntimeResolution) -> None:
        self._sink.record("fallback", self._component_id, resolution)

    def record_replay_error(self, resolution: VocoderRuntimeResolution) -> None:
        self._sink.record("replay_error", self._component_id, resolution)


class VocoderGraphStatsSink:
    """Manager-read, Recorder-write runtime counters."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self._calls: Counter[str] = Counter()
        self._outcomes: Counter[tuple[str, str]] = Counter()
        # Counts graph specializations actually selected for replay.
        self._descriptors: Counter[tuple[str, object]] = Counter()
        # Counts runtime keys observed before Descriptor bucket selection.
        self._runtime_keys: Counter[tuple[str, object]] = Counter()

    def recorder_for(self, component_id: str) -> _ComponentRecorder | _NoOpRecorder:
        if not self.enabled:
            return _NoOpRecorder()
        return _ComponentRecorder(self, component_id)

    def record(
        self,
        outcome: str,
        component_id: str,
        resolution: VocoderRuntimeResolution,
    ) -> None:
        with self._lock:
            self._calls[component_id] += 1
            self._outcomes[(component_id, outcome)] += 1
            if resolution.descriptor is not None:
                self._descriptors[(component_id, resolution.descriptor.variant)] += 1
            key = (component_id, resolution.runtime_key.variant)
            self._runtime_keys[key] += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "calls": dict(self._calls),
                "outcomes": dict(self._outcomes),
                "descriptors": dict(self._descriptors),
                "runtime_keys": dict(self._runtime_keys),
            }


class VocoderCUDAGraphManager:
    """Consumes resolved Components and owns capture/bind/restore lifecycle."""

    def __init__(self, *, vllm_config: VllmConfig, device: torch.device) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.components: tuple[VocoderCUDAGraphComponent, ...] = ()
        self.managed_components: dict[str, ManagedComponent] = {}
        self._component_configs: dict[str, VocoderComponentRuntimeConfig] = {}
        self._capture_lock = threading.RLock()
        self._runtime_capture_stream: torch.cuda.Stream | None = None
        self._prepared = False
        self._capture_finished = False

        raw_config = getattr(vllm_config.model_config, "vocoder_cudagraph_config", None)
        if raw_config is None:
            raw_config = {}
        if not isinstance(raw_config, Mapping):
            raise TypeError("vocoder_cudagraph must be a mapping")
        self.config = dict(raw_config)
        log_stats = self.config.get("log_stats", False)
        self.stats_sink = VocoderGraphStatsSink(enabled=log_stats)

    @staticmethod
    def _model_shared_config_keys(model: object) -> frozenset[str]:
        declaration: object = getattr(model, "vocoder_cudagraph_shared_config_keys", frozenset())
        if callable(declaration):
            declaration = declaration()
        return frozenset(cast(Sequence[str], declaration))

    def _synchronized_free_memory(self) -> int:
        if self.device.type == "cpu":
            return 0
        torch.accelerator.synchronize()
        return int(torch.accelerator.get_memory_info()[0])

    def prepare(self, model: SupportsVocoderCUDAGraph) -> None:
        if self._prepared:
            raise RuntimeError("VocoderCUDAGraphManager.prepare() called more than once")

        components = tuple(model.get_vocoder_cudagraph_components())
        component_by_id: dict[str, VocoderCUDAGraphComponent] = {}
        for component in components:
            if not isinstance(component, VocoderCUDAGraphComponent):
                raise TypeError("get_vocoder_cudagraph_components() must return VocoderCUDAGraphComponent objects")
            if not component.component_id:
                raise ValueError("Vocoder CUDA Graph component_id must not be empty")
            if component.component_id in component_by_id:
                raise ValueError(f"Duplicate vocoder CUDA Graph component_id: {component.component_id}")
            try:
                if len(set(component.descriptors)) != len(component.descriptors):
                    raise ValueError(f"Duplicate Descriptor in Component {component.component_id}")
            except TypeError as exc:
                raise TypeError(f"Descriptors for Component {component.component_id} must be hashable") from exc
            component_by_id[component.component_id] = component
            if not component.descriptors:
                logger.info(
                    "Vocoder CUDA Graph Component %s is known but has no startup "
                    "Descriptors for the resolved model configuration",
                    component.component_id,
                )

        unknown_shared = set(self.config) - _FRAMEWORK_CONFIG_KEYS - self._model_shared_config_keys(model)
        if unknown_shared:
            names = ", ".join(sorted(unknown_shared))
            raise ValueError(f"Unknown vocoder_cudagraph config key(s): {names}")

        raw_component_configs = self.config.get("components", {})
        if not isinstance(raw_component_configs, Mapping):
            raise TypeError("vocoder_cudagraph.components must be a mapping")
        unknown_component_ids = set(raw_component_configs) - set(component_by_id)
        if unknown_component_ids:
            names = ", ".join(sorted(str(name) for name in unknown_component_ids))
            raise ValueError(f"Unknown vocoder CUDA Graph component override(s): {names}")

        for component_id, component in component_by_id.items():
            raw_component = raw_component_configs.get(component_id, {})
            if not isinstance(raw_component, Mapping):
                raise TypeError(f"vocoder_cudagraph.components.{component_id} must be a mapping")
            unknown_keys = set(raw_component) - _COMPONENT_POLICY_KEYS - set(component.supported_config_keys)
            if unknown_keys:
                names = ", ".join(sorted(unknown_keys))
                raise ValueError(f"Unknown config key(s) for vocoder Component {component_id}: {names}")
            enabled = raw_component.get("enabled", True)
            lazy = raw_component.get("enable_lazy_capture", False)
            max_extra = raw_component.get("max_extra_graphs", 0)
            self._component_configs[component_id] = VocoderComponentRuntimeConfig(
                enabled=enabled,
                enable_lazy_capture=lazy,
                max_extra_graphs=max_extra,
            )

        self.components = components
        self._prepared = True
        logger.info(
            "Prepared runner-owned vocoder CUDA Graph Components: %s",
            [component.component_id for component in components],
        )

    def capture_entry(
        self,
        component: VocoderCUDAGraphComponent,
        descriptor: VocoderCUDAGraphDescriptor,
    ) -> VocoderCUDAGraphEntry | None:
        routine = component.routine
        buffers: object | None = None
        graph = None
        try:
            buffers = routine.allocate_buffers(descriptor, self.device)
            num_warmups = max(
                1,
                int(getattr(self.vllm_config.compilation_config, "cudagraph_num_of_warmups", 0)),
            )
            for _ in range(num_warmups):
                # Prepare the reusable static buffers before every warmup.
                routine.prepare_for_capture(buffers)
                try:
                    routine.forward_for_capture(buffers)
                finally:
                    routine.after_capture(buffers)
            routine.prepare_for_capture(buffers)
            torch.cuda.current_stream(self.device).synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with (
                    torch.inference_mode(),
                    torch.cuda.graph(
                        graph,
                        pool=current_platform.get_global_graph_pool(),
                    ),
                ):
                    captured_output = routine.forward_for_capture(buffers)
            finally:
                routine.after_capture(buffers)
            return VocoderCUDAGraphEntry(
                descriptor=descriptor,
                graph=graph,
                buffers=buffers,
                captured_output=captured_output,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            if graph is not None:
                graph.reset()
            logger.warning(
                "Failed to capture vocoder CUDA Graph Component %s Descriptor %r; this Descriptor will remain eager",
                component.component_id,
                descriptor,
                exc_info=True,
            )
            return None

    def profile_memory(self) -> int:
        """Estimate startup graph memory with throwaway captures."""
        if not self._prepared:
            raise RuntimeError("VocoderCUDAGraphManager must be prepared before profiling")

        captured: list[VocoderCUDAGraphEntry] = []
        estimate = 0
        try:
            for component in self.components:
                component_config = self._component_configs[component.component_id]
                if not component_config.enabled or not component.descriptors:
                    continue
                samples: list[int] = []
                for descriptor in component.descriptors[:2]:
                    free_before = self._synchronized_free_memory()
                    entry = self.capture_entry(component, descriptor)
                    if entry is None:
                        continue
                    free_after = self._synchronized_free_memory()
                    captured.append(entry)
                    samples.append(max(0, free_before - free_after))
                if not samples:
                    continue
                first_capture = samples[0]
                per_graph = max(samples[1] if len(samples) > 1 else 0, 1 << 20)
                lazy_graphs = component_config.max_extra_graphs if component_config.enable_lazy_capture else 0
                extra_graphs = len(component.descriptors) - 1 + lazy_graphs
                estimate += first_capture + per_graph * extra_graphs
                logger.debug(
                    "Estimated vocoder Component %s CUDA graph memory: "
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
        descriptor: VocoderCUDAGraphDescriptor,
    ) -> VocoderCUDAGraphEntry | None:
        existing = managed.entries.get(descriptor)
        if existing is not None:
            managed.entries.move_to_end(descriptor)
            return existing
        if descriptor in managed.failed_descriptors:
            return None

        entry = self.capture_entry(managed.component, descriptor)
        if entry is None:
            managed.failed_descriptors.add(descriptor)
            return None

        managed.entries[descriptor] = entry
        managed.entries.move_to_end(descriptor)
        if managed.max_graphs is not None:
            self._evict_lru_if_needed(managed)
        return entry

    def _evict_lru_if_needed(self, managed: ManagedComponent) -> None:
        assert managed.max_graphs is not None
        while len(managed.entries) > managed.max_graphs:
            self._evict_oldest(managed)

    @staticmethod
    def _destroy_entry(entry: VocoderCUDAGraphEntry) -> None:
        entry.graph.reset()
        entry.buffers = None
        entry.captured_output = None

    def _evict_oldest(self, managed: ManagedComponent) -> None:
        _, entry = managed.entries.popitem(last=False)
        self._destroy_entry(entry)

    def _touch_entry(self, managed: ManagedComponent, descriptor: VocoderCUDAGraphDescriptor) -> None:
        with self._capture_lock:
            if descriptor in managed.entries:
                managed.entries.move_to_end(descriptor)

    def _available_descriptors(
        self,
        managed: ManagedComponent,
    ) -> frozenset[VocoderCUDAGraphDescriptor]:
        with self._capture_lock:
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
        descriptor: VocoderCUDAGraphDescriptor,
    ) -> VocoderCUDAGraphEntry | None:
        if not managed.enable_lazy_capture:
            return None
        if descriptor in managed.failed_descriptors:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        with self._capture_lock:
            existing = managed.entries.get(descriptor)
            if existing is not None:
                managed.entries.move_to_end(descriptor)
                return existing
            if descriptor in managed.failed_descriptors:
                return None
            if managed.max_graphs == 0:
                return None
            with self._runtime_capture_scope():
                entry = self._capture_and_register(managed, descriptor)
            if entry is not None:
                logger.info(
                    "Lazy-captured vocoder CUDA Graph Component %s Descriptor %r",
                    managed.component.component_id,
                    descriptor,
                )
            return entry

    def _make_runtime_miss_handler(
        self,
        managed: ManagedComponent,
    ) -> Callable[[VocoderRuntimeResolution], VocoderCUDAGraphEntry | None]:
        if not managed.enable_lazy_capture:
            return lambda resolution: None

        def on_runtime_miss(resolution: VocoderRuntimeResolution) -> VocoderCUDAGraphEntry | None:
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
                else VocoderRuntimeResolution(
                    runtime_key=resolution.runtime_key,
                    descriptor=descriptor,
                )
            )
            with entry.replay_lock:
                try:
                    routine.copy_runtime_inputs(args, kwargs, entry.buffers)
                    entry.graph.replay()
                    output = routine.output_after_replay(args, kwargs, entry.buffers, entry.captured_output)
                    if clone_output:
                        output = clone_tensor_tree(output)
                except Exception:
                    recorder.record_replay_error(graph_resolution)
                    raise
                if managed.enable_lazy_capture:
                    self._touch_entry(managed, descriptor)
                recorder.record_graph_hit(graph_resolution)
            return output

        return runtime_callable

    def capture_and_bind(self) -> int:
        if not self._prepared:
            raise RuntimeError("VocoderCUDAGraphManager must be prepared before capture")
        if self._capture_finished:
            raise RuntimeError("Vocoder CUDA Graph capture has already completed")
        self._capture_finished = True

        capture_start = time.perf_counter()
        free_before = self._synchronized_free_memory()
        prepared: dict[str, ManagedComponent] = {}

        # Phase 1: every Component remains eager until all startup captures finish.
        selected = [
            component for component in self.components if self._component_configs[component.component_id].enabled
        ]
        for component in selected:
            component_config = self._component_configs[component.component_id]
            if component._bound_handle is not None:
                raise RuntimeError(f"Component already bound before capture: {component.component_id}")
            managed = ManagedComponent(
                component=component,
                entries=OrderedDict(),
                enable_lazy_capture=component_config.enable_lazy_capture,
            )
            progress = (
                tqdm(
                    total=len(component.descriptors),
                    desc=f"Capture {component.component_id}",
                    unit="graph",
                    leave=True,
                )
                if component.descriptors
                else None
            )
            try:
                for descriptor in component.descriptors:
                    try:
                        self._capture_and_register(managed, descriptor)
                    finally:
                        assert progress is not None
                        progress.update(1)
            finally:
                if progress is not None:
                    progress.close()
            logger.info(
                "Vocoder CUDA Graph Component %s captured %d/%d startup Descriptors",
                component.component_id,
                len(managed.entries),
                len(component.descriptors),
            )
            managed.max_graphs = len(managed.entries) + component_config.max_extra_graphs
            if not managed.entries and not managed.enable_lazy_capture:
                continue
            prepared[component.component_id] = managed

        # Phase 2: runtime assembly/binding failures are programming or
        # lifecycle errors and must propagate after capture has completed.
        active: dict[str, ManagedComponent] = {}
        for component_id, managed in prepared.items():
            component = managed.component
            runtime_callable = self._build_runtime_callable(
                managed,
                self.stats_sink.recorder_for(component_id),
            )
            # Bind one opaque runtime endpoint to the stable model-owned
            # Component; GraphEntry/Descriptor internals stay manager-owned.
            component._bind_handle(
                VocoderGraphHandle(
                    runtime_callable,
                    lambda managed=managed: self._available_descriptors(managed),
                )
            )
            active[component_id] = managed

        self.managed_components = active
        free_after = self._synchronized_free_memory()
        captured_memory = max(0, free_before - free_after)
        logger.info(
            "Vocoder CUDA Graph capture finished in %.2fs, bound=%s, memory=%.2f MiB",
            time.perf_counter() - capture_start,
            list(active),
            captured_memory / (1 << 20),
        )
        return captured_memory

    def clear(self) -> None:
        for managed in self.managed_components.values():
            managed.component._restore_eager()
            while managed.entries:
                self._evict_oldest(managed)
            managed.failed_descriptors.clear()
        self.managed_components.clear()
        self._runtime_capture_stream = None
        if self.stats_sink.enabled:
            logger.info("Vocoder CUDA Graph runtime stats: %s", self.stats_sink.snapshot())
