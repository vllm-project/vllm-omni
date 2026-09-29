# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import logging
from collections import Counter, OrderedDict
from collections.abc import Callable, Set
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NamedTuple, cast
from unittest.mock import patch

import pytest
import torch
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.interfaces.model_local_cudagraph import (
    BaseModelLocalCUDAGraphRoutine,
    ModelLocalCaptureMode,
    ModelLocalCUDAGraphComponent,
    ModelLocalCUDAGraphDescriptor,
    ModelLocalGraphHandle,
    ModelLocalRuntimeKey,
    ModelLocalRuntimeResolution,
    SupportsModelLocalCUDAGraph,
)
from vllm_omni.worker.model_local_cudagraph_manager import (
    ManagedComponent,
    ModelLocalCUDAGraphEntry,
    ModelLocalCUDAGraphManager,
    ModelLocalGraphStatsSink,
    clone_tensor_tree,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _NestedOutput(NamedTuple):
    tensor: torch.Tensor
    metadata: dict[str, object]


def test_clone_tensor_tree_clones_supported_tensor_containers() -> None:
    tensor = torch.tensor([1.0, 2.0])
    output = {
        "tensor": tensor,
        "tuple": (tensor, [tensor]),
        "namedtuple": _NestedOutput(tensor, {"label": "audio"}),
    }

    cloned = clone_tensor_tree(output)
    assert isinstance(cloned, dict)

    assert isinstance(cloned["namedtuple"], _NestedOutput)
    assert cloned["namedtuple"].metadata == {"label": "audio"}
    cloned_tensors = (
        cloned["tensor"],
        cloned["tuple"][0],
        cloned["tuple"][1][0],
        cloned["namedtuple"].tensor,
    )
    for cloned_tensor in cloned_tensors:
        torch.testing.assert_close(cloned_tensor, tensor)
        assert cloned_tensor.data_ptr() != tensor.data_ptr()


@dataclass
class _Buffers:
    input: torch.Tensor
    output: torch.Tensor


class _Graph:
    def __init__(self, buffers: _Buffers, *, fail: bool = False) -> None:
        self.buffers = buffers
        self.fail = fail

    def reset(self) -> None:
        self.reset_called = True

    def replay(self) -> None:
        if self.fail:
            raise RuntimeError("replay failed")
        self.buffers.output.copy_(self.buffers.input * 2)


class _Routine(BaseModelLocalCUDAGraphRoutine):
    def __init__(self) -> None:
        self.eager_calls = 0
        self.validate_calls = 0
        self._runnable: Callable[[torch.Tensor], torch.Tensor] = lambda value: value * 2

    @property
    def runnable(self) -> Callable[..., Any]:
        return self._runnable

    def eager_call(self, value: torch.Tensor) -> torch.Tensor:
        self.eager_calls += 1
        return self._runnable(value)

    def validate_runtime_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.validate_calls += 1
        if kwargs or len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError("invalid invocation")
        if args[0].numel() == 0:
            raise ValueError("empty input")

    def resolve_runtime(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        available: Set[ModelLocalCUDAGraphDescriptor],
    ) -> ModelLocalRuntimeResolution:
        del kwargs
        size = int(args[0].numel())
        descriptor = min(
            (item for item in available if isinstance(item.variant, int) and item.variant >= size),
            key=lambda item: item.variant if isinstance(item.variant, int) else 0,
            default=None,
        )
        return ModelLocalRuntimeResolution(ModelLocalRuntimeKey(size), descriptor)

    def allocate_buffers(self, descriptor: ModelLocalCUDAGraphDescriptor, device: torch.device) -> _Buffers:
        assert isinstance(descriptor.variant, int)
        size = descriptor.variant
        return _Buffers(torch.zeros(size, device=device), torch.zeros(size, device=device))

    def forward_for_capture(self, buffers: object) -> torch.Tensor:
        assert isinstance(buffers, _Buffers)
        buffers.output.copy_(buffers.input * 2)
        return buffers.output

    def copy_runtime_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        buffers: object,
    ) -> None:
        del kwargs
        assert isinstance(buffers, _Buffers)
        buffers.input.zero_()
        buffers.input[: args[0].numel()].copy_(args[0])

    def output_after_replay(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        buffers: object,
        captured_output: object,
    ) -> torch.Tensor:
        del kwargs, buffers
        assert isinstance(captured_output, torch.Tensor)
        return captured_output[: args[0].numel()]


class _LifecycleRoutine(_Routine):
    def __init__(self, *, fail_forward: bool = False, context_only: bool = False) -> None:
        super().__init__()
        self.events: list[str] = []
        self.fail_forward = fail_forward
        self.context_only = context_only

    def prepare_for_capture(self, buffers: object) -> None:
        del buffers
        if self.context_only:
            raise AssertionError("manager called prepare_for_capture directly")
        self.events.append("prepare")

    def forward_for_capture(self, buffers: object) -> torch.Tensor:
        if self.context_only:
            return _Routine.forward_for_capture(self, buffers)
        self.events.append("forward")
        if self.fail_forward:
            raise RuntimeError("capture forward failed")
        return super().forward_for_capture(buffers)

    def after_capture(self, buffers: object) -> None:
        del buffers
        if self.context_only:
            raise AssertionError("manager called after_capture directly")
        self.events.append("after")

    @contextmanager
    def capture_context(self, descriptor: ModelLocalCUDAGraphDescriptor, buffers: object):
        if self.context_only:
            del descriptor, buffers
            self.events.append("context-enter")
            try:
                yield
            finally:
                self.events.append("context-exit")
            return
        with super().capture_context(descriptor, buffers):
            yield


class _ScopedLifecycleRoutine(_LifecycleRoutine):
    @contextmanager
    def capture_context(self, descriptor: ModelLocalCUDAGraphDescriptor, buffers: object):
        self.events.append("scope-enter")
        try:
            with super().capture_context(descriptor, buffers):
                yield
        finally:
            self.events.append("scope-exit")


class _TestManager(ModelLocalCUDAGraphManager):
    def __init__(self, *, config: dict[str, Any] | None = None, log_stats: bool = False) -> None:
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(model_local_cudagraph=config),
            compilation_config=SimpleNamespace(cudagraph_num_of_warmups=0),
            observability_config=SimpleNamespace(cudagraph_metrics=log_stats),
        )
        super().__init__(vllm_config=vllm_config, device=torch.device("cpu"))
        self._use_default_component_config = config is None
        self.components_during_capture: list[tuple[bool, ...]] = []
        self.fail_replay_for: set[tuple[str, object]] = set()
        self.fail_capture_for: set[tuple[str, object]] = set()
        self.capture_attempts: Counter[tuple[str, object]] = Counter()
        self.capture_pools: list[object | None] = []

    def prepare(self, model: SupportsModelLocalCUDAGraph) -> None:
        if self._use_default_component_config:
            self.config = {component.component_id: {} for component in model.get_model_local_cudagraph_components()}
        super().prepare(model)

    def capture_entry(
        self,
        component: ModelLocalCUDAGraphComponent,
        descriptor: ModelLocalCUDAGraphDescriptor,
        *,
        graph_pool: object | None = None,
    ) -> ModelLocalCUDAGraphEntry | None:
        self.capture_pools.append(graph_pool)
        self.components_during_capture.append(tuple(item._bound_handle is not None for item in self.components))
        key = (component.component_id, descriptor.variant)
        self.capture_attempts[key] += 1
        if not component.validate_descriptor(descriptor):
            return None
        if key in self.fail_capture_for:
            return None
        buffers = component.routine.allocate_buffers(descriptor, self.device)
        assert isinstance(buffers, _Buffers)
        output = component.routine.forward_for_capture(buffers)
        graph = _Graph(
            buffers,
            fail=(component.component_id, descriptor.variant) in self.fail_replay_for,
        )
        return ModelLocalCUDAGraphEntry(
            descriptor=descriptor,
            graph=cast(torch.cuda.CUDAGraph, graph),
            buffers=buffers,
            captured_output=output,
        )


class _Model:
    supports_model_local_cudagraph = True
    model_local_cudagraph_shared_config_keys = frozenset({"shared_shape_policy"})

    def __init__(self, components: tuple[ModelLocalCUDAGraphComponent, ...]) -> None:
        self.components = components

    def get_model_local_cudagraph_components(self) -> tuple[ModelLocalCUDAGraphComponent, ...]:
        return self.components


def _component(
    component_id: str,
    *sizes: int,
    capture_order_key: Callable[[ModelLocalCUDAGraphDescriptor], Any] | None = None,
    capture_mode: ModelLocalCaptureMode = ModelLocalCaptureMode.PRECAPTURE,
) -> tuple[ModelLocalCUDAGraphComponent, _Routine]:
    routine = _Routine()
    component = ModelLocalCUDAGraphComponent(
        component_id,
        routine,
        [ModelLocalCUDAGraphDescriptor(size) for size in sizes],
        supported_config_keys=frozenset({"bucket_policy"}),
        capture_order_key=capture_order_key,
        capture_mode=capture_mode,
    )
    return component, routine


def _manager_for_capture(*, warmups: int) -> ModelLocalCUDAGraphManager:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_local_cudagraph=None),
        compilation_config=SimpleNamespace(cudagraph_num_of_warmups=warmups),
    )
    return ModelLocalCUDAGraphManager(vllm_config=vllm_config, device=torch.device("cpu"))


def _mock_cuda_capture(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(torch.cuda, "graph", lambda *_args, **_kwargs: nullcontext())


def test_handle_exposes_runtime_call_and_read_only_graph_coverage() -> None:
    handle = ModelLocalGraphHandle(lambda value, *, offset=0: value + offset)

    assert handle(2, offset=3) == 5
    assert handle.available_descriptors == frozenset()
    assert not hasattr(handle, "capture")
    assert not hasattr(handle, "replay")
    assert not hasattr(handle, "entries")


def test_component_descriptor_validation_defaults_to_capture() -> None:
    component, _ = _component("decode", 2)

    assert component.validate_descriptor(ModelLocalCUDAGraphDescriptor(2))


def test_capture_context_wraps_each_warmup_and_capture_forward(monkeypatch) -> None:
    routine = _LifecycleRoutine()
    component = ModelLocalCUDAGraphComponent("decode", routine, [ModelLocalCUDAGraphDescriptor(2)])
    manager = _manager_for_capture(warmups=2)
    _mock_cuda_capture(monkeypatch)

    manager.capture_entry(component, ModelLocalCUDAGraphDescriptor(2), graph_pool=object())

    assert routine.events == ["prepare", "forward", "after"] * 3


def test_capture_context_runs_cleanup_when_forward_raises() -> None:
    routine = _LifecycleRoutine(fail_forward=True)
    component = ModelLocalCUDAGraphComponent("decode", routine, [ModelLocalCUDAGraphDescriptor(2)])
    manager = _manager_for_capture(warmups=1)

    with pytest.raises(RuntimeError, match="capture forward failed"):
        manager.capture_entry(component, ModelLocalCUDAGraphDescriptor(2))

    assert routine.events == ["prepare", "forward", "after"]


def test_graph_capture_failure_resets_graph_and_uses_warmup_stream(monkeypatch) -> None:
    routine = _LifecycleRoutine()
    component = ModelLocalCUDAGraphComponent("decode", routine, [ModelLocalCUDAGraphDescriptor(2)])
    manager = _manager_for_capture(warmups=1)
    capture_stream = SimpleNamespace(synchronize=lambda: None)
    graph = SimpleNamespace(reset_called=False)

    def reset_graph():
        graph.reset_called = True

    graph.reset = reset_graph
    graph_streams = []

    def graph_context(*_args, **kwargs):
        graph_streams.append(kwargs["stream"])
        return nullcontext()

    original_forward = routine.forward_for_capture
    forward_calls = 0

    def forward(buffers):
        nonlocal forward_calls
        forward_calls += 1
        if forward_calls == 2:
            routine.events.append("forward")
            raise RuntimeError("graph capture failed")
        return original_forward(buffers)

    monkeypatch.setattr(routine, "forward_for_capture", forward)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: capture_stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)

    with pytest.raises(RuntimeError, match="graph capture failed"):
        manager.capture_entry(component, ModelLocalCUDAGraphDescriptor(2), graph_pool=object())

    assert routine.events == ["prepare", "forward", "after"] * 2
    assert graph_streams == [capture_stream]
    assert graph.reset_called


def test_manager_uses_capture_context_instead_of_prepare_or_after(monkeypatch) -> None:
    routine = _LifecycleRoutine(context_only=True)
    component = ModelLocalCUDAGraphComponent("decode", routine, [ModelLocalCUDAGraphDescriptor(2)])
    manager = _manager_for_capture(warmups=1)
    _mock_cuda_capture(monkeypatch)

    manager.capture_entry(component, ModelLocalCUDAGraphDescriptor(2), graph_pool=object())

    assert routine.events == ["context-enter", "context-exit"] * 2


def test_capture_context_override_composes_default_lifecycle(monkeypatch) -> None:
    routine = _ScopedLifecycleRoutine()
    component = ModelLocalCUDAGraphComponent("decode", routine, [ModelLocalCUDAGraphDescriptor(2)])
    manager = _manager_for_capture(warmups=1)
    _mock_cuda_capture(monkeypatch)

    manager.capture_entry(component, ModelLocalCUDAGraphDescriptor(2), graph_pool=object())

    assert routine.events == ["scope-enter", "prepare", "forward", "after", "scope-exit"] * 2


def test_stats_sink_bounds_detail_items_but_preserves_aggregate_counters() -> None:
    sink = ModelLocalGraphStatsSink(enabled=True, max_log_items=2)
    for variant in (1, 2, 3):
        resolution = ModelLocalRuntimeResolution(
            runtime_key=ModelLocalRuntimeKey(variant),
            descriptor=ModelLocalCUDAGraphDescriptor(variant),
        )
        sink.record("hit", "decode", resolution)

    snapshot = sink.snapshot()
    assert snapshot["calls"] == {"decode": 3}
    assert snapshot["outcomes"] == {("decode", "hit"): 3}
    assert snapshot["descriptors"] == {("decode", 2): 1, ("decode", 3): 1}
    assert snapshot["runtime_keys"] == {("decode", 2): 1, ("decode", 3): 1}


def test_stats_sink_logs_every_100_component_calls() -> None:
    sink = ModelLocalGraphStatsSink(enabled=True)
    resolution = ModelLocalRuntimeResolution(ModelLocalRuntimeKey(2), ModelLocalCUDAGraphDescriptor(2))

    with patch.object(logging.Logger, "info") as log_info:
        for _ in range(99):
            sink.record("hit", "decode", resolution)
        log_info.assert_not_called()
        sink.record("hit", "decode", resolution)

    log_info.assert_called_once()
    assert log_info.call_args.args[1] == 100
    assert log_info.call_args.args[2]["outcomes"] == {("decode", "hit"): 100}


def test_runtime_lazy_capture_logs_only_new_entries(monkeypatch) -> None:
    component, _ = _component("decode", 2)
    manager = _TestManager()
    descriptor = ModelLocalCUDAGraphDescriptor(3)
    fake_buffers = _Buffers(torch.zeros(1), torch.zeros(1))
    fake_entry = ModelLocalCUDAGraphEntry(
        descriptor=descriptor,
        graph=cast(torch.cuda.CUDAGraph, _Graph(fake_buffers)),
        buffers=fake_buffers,
        captured_output=fake_buffers.output,
    )
    managed = ManagedComponent(
        component=component,
        entries=OrderedDict(),
        capture_mode=ModelLocalCaptureMode.PRECAPTURE_LAZY,
        max_graphs=1,
    )
    calls: list[ModelLocalCUDAGraphDescriptor] = []

    def capture(managed_component, requested_descriptor):
        assert managed_component is managed
        calls.append(requested_descriptor)
        return fake_entry

    monkeypatch.setattr(manager, "_capture_and_register", capture)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(manager, "_runtime_capture_scope", nullcontext)
    with patch.object(logging.Logger, "info") as log_info:
        result = manager._runtime_capture_and_register(managed, descriptor)

    assert result is fake_entry
    assert calls == [descriptor]
    log_info.assert_called_once_with(
        "Lazy-captured model-local CUDA Graph Component %s Descriptor %r",
        "decode",
        descriptor,
    )

    managed.entries[descriptor] = fake_entry
    calls.clear()
    with patch.object(logging.Logger, "info") as log_info:
        result = manager._runtime_capture_and_register(managed, descriptor)

    assert result is fake_entry
    assert calls == []
    log_info.assert_not_called()


def test_capture_binds_only_after_all_components_are_captured_and_clear_restores_eager() -> None:
    first, first_routine = _component("first", 4)
    second, _ = _component("second", 8)
    manager = _TestManager()
    manager.prepare(_Model((first, second)))

    manager.capture_and_bind()

    assert manager.components_during_capture
    assert all(not any(bound) for bound in manager.components_during_capture)
    assert set(manager.managed_components) == {"first", "second"}
    assert isinstance(first._bound_handle, ModelLocalGraphHandle)
    assert isinstance(second._bound_handle, ModelLocalGraphHandle)
    assert first.available_descriptors == frozenset({ModelLocalCUDAGraphDescriptor(4)})
    assert second.available_descriptors == frozenset({ModelLocalCUDAGraphDescriptor(8)})
    value = torch.tensor([1.0, 2.0])
    first_output = first(value)
    assert torch.equal(first_output, value * 2)
    assert first_routine.eager_calls == 0

    # clone_output=True prevents the next replay from overwriting a retained result.
    retained = first_output.clone()
    first(torch.tensor([4.0, 5.0]))
    assert torch.equal(first_output, retained)

    manager.clear()
    assert first._bound_handle is None
    assert first.available_descriptors == frozenset()
    assert torch.equal(first(value), value * 2)
    assert first_routine.eager_calls == 1


def test_unified_fallback_uses_segmented_graph_after_all_components_bind() -> None:
    segmented, segmented_routine = _component("segmented", 4)

    class _UnifiedRoutine(_Routine):
        def eager_call(self, value: torch.Tensor) -> torch.Tensor:
            self.eager_calls += 1
            return segmented(value) + 1

        def forward_for_capture(self, buffers: object) -> torch.Tensor:
            assert isinstance(buffers, _Buffers)
            buffers.output.copy_(segmented(buffers.input) + 1)
            return buffers.output

    unified_routine = _UnifiedRoutine()
    unified = ModelLocalCUDAGraphComponent("unified", unified_routine, [ModelLocalCUDAGraphDescriptor(2)])
    manager = _TestManager()
    manager.prepare(_Model((segmented, unified)))
    manager.capture_and_bind()

    # Unified capture called segmented after its capture, but before binding.
    assert segmented_routine.eager_calls == 1
    assert all(not any(bound) for bound in manager.components_during_capture)

    segmented_entry = manager.managed_components["segmented"].entries[ModelLocalCUDAGraphDescriptor(4)]
    replay_calls = 0
    original_replay = segmented_entry.graph.replay

    def replay() -> None:
        nonlocal replay_calls
        replay_calls += 1
        original_replay()

    segmented_entry.graph.replay = replay
    value = torch.tensor([1.0, 2.0, 3.0])  # Misses the unified graph, fits the segmented graph.
    torch.testing.assert_close(unified(value), value * 2 + 1)
    assert unified_routine.eager_calls == 1
    assert segmented_routine.eager_calls == 1
    assert replay_calls == 1


def test_coverage_miss_falls_back_but_validation_and_replay_errors_propagate() -> None:
    component, routine = _component("decode", 2)
    manager = _TestManager(log_stats=True)
    manager.fail_replay_for.add(("decode", 2))
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()

    fallback_input = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(component(fallback_input), fallback_input * 2)
    assert routine.eager_calls == 1

    with pytest.raises(ValueError, match="empty input"):
        component(torch.tensor([]))
    assert routine.eager_calls == 1

    with pytest.raises(RuntimeError, match="replay failed"):
        component(torch.tensor([1.0]))
    assert routine.eager_calls == 1
    outcomes = manager.stats_sink.snapshot()["outcomes"]
    assert isinstance(outcomes, dict)
    assert outcomes[("decode", "fallback")] == 1
    assert outcomes[("decode", "replay_error")] == 1


def test_copy_and_postprocess_errors_propagate_without_eager_retry() -> None:
    component, routine = _component("decode", 2)
    manager = _TestManager(log_stats=True)
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()

    def fail_copy(*_args, **_kwargs):
        raise RuntimeError("copy failed")

    routine.copy_runtime_inputs = fail_copy
    with pytest.raises(RuntimeError, match="copy failed"):
        component(torch.tensor([1.0]))
    assert routine.eager_calls == 0

    component, routine = _component("decode-postprocess", 2)
    manager = _TestManager(log_stats=True)
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()

    def fail_output(*_args, **_kwargs):
        raise RuntimeError("postprocess failed")

    routine.output_after_replay = fail_output
    with pytest.raises(RuntimeError, match="postprocess failed"):
        component(torch.tensor([1.0]))
    assert routine.eager_calls == 0


def test_config_validation_catches_unknown_component_and_extension_keys() -> None:
    component, _ = _component("decode", 2)

    manager = _TestManager(config={"unknown": 1})
    with pytest.raises(ValueError, match="Unknown model-local CUDA Graph component"):
        manager.prepare(_Model((component,)))

    manager = _TestManager(config={"missing": {}})
    with pytest.raises(ValueError, match="Unknown model-local CUDA Graph component"):
        manager.prepare(_Model((component,)))

    manager = _TestManager(config={"decode": {"unknown_bucket_policy": [2]}})
    with pytest.raises(ValueError, match="Unknown config key"):
        manager.prepare(_Model((component,)))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("max_extra_graphs", True, "must be a non-negative integer"),
        ("max_extra_graphs", -1, "must be a non-negative integer"),
        ("max_extra_graphs", 1.0, "must be a non-negative integer"),
    ],
)
def test_component_policy_validation_rejects_invalid_types(key, value, message) -> None:
    component, _ = _component("decode", 2, capture_mode=ModelLocalCaptureMode.PRECAPTURE_LAZY)
    manager = _TestManager(config={"decode": {key: value}})

    with pytest.raises(TypeError, match=message):
        manager.prepare(_Model((component,)))


def test_component_registry_rejects_duplicate_ids_and_descriptors() -> None:
    first, _ = _component("decode", 2)
    duplicate_id, _ = _component("decode", 3)
    manager = _TestManager()
    with pytest.raises(ValueError, match="Duplicate model-local CUDA Graph component_id"):
        manager.prepare(_Model((first, duplicate_id)))

    duplicate_descriptor = ModelLocalCUDAGraphComponent(
        "duplicate",
        _Routine(),
        [ModelLocalCUDAGraphDescriptor(2), ModelLocalCUDAGraphDescriptor(2)],
    )
    manager = _TestManager()
    with pytest.raises(ValueError, match="Duplicate Descriptor"):
        manager.prepare(_Model((duplicate_descriptor,)))


def test_omitted_component_remains_on_original_eager_callable() -> None:
    component, routine = _component("decode", 2)
    manager = _TestManager(config={})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()

    assert not manager.managed_components
    assert component._bound_handle is None
    component(torch.tensor([1.0]))
    assert routine.eager_calls == 1


def test_pure_lazy_profiles_descriptors_but_does_not_capture_at_startup(monkeypatch) -> None:
    component, routine = _component("decode", 2, capture_mode=ModelLocalCaptureMode.PURE_LAZY)
    manager = _TestManager(config={"decode": {"max_extra_graphs": 1}})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()

    assert manager.capture_attempts[("decode", 2)] == 0
    assert component.available_descriptors == frozenset()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(manager, "_runtime_capture_scope", nullcontext)
    value = torch.ones(2)
    assert torch.equal(component(value), value * 2)
    assert component.available_descriptors == frozenset({ModelLocalCUDAGraphDescriptor(2)})
    assert manager.capture_attempts[("decode", 2)] == 1
    larger = torch.ones(3)
    assert torch.equal(component(larger), larger * 2)
    assert manager.capture_attempts[("decode", 3)] == 0
    assert routine.eager_calls == 1


def test_pure_lazy_requires_profiling_descriptors() -> None:
    component, _ = _component("decode", capture_mode=ModelLocalCaptureMode.PURE_LAZY)
    manager = _TestManager(config={"decode": {}})
    with pytest.raises(ValueError, match="needs descriptors for memory profiling"):
        manager.prepare(_Model((component,)))


def test_descriptor_rejection_falls_back_to_eager_without_capture(monkeypatch) -> None:
    component, routine = _component("decode", 2, capture_mode=ModelLocalCaptureMode.PRECAPTURE_LAZY)
    manager = _TestManager(config={"decode": {"max_extra_graphs": 1}})
    validation_calls = []

    def reject(descriptor):
        validation_calls.append(descriptor)
        return False

    component.validate_descriptor = reject
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(manager, "_runtime_capture_scope", nullcontext)

    value = torch.tensor([1.0, 2.0])
    assert torch.equal(component(value), value * 2)
    assert torch.equal(component(value), value * 2)
    assert routine.eager_calls == 2
    assert manager.capture_attempts[("decode", 2)] == 3
    assert validation_calls == [ModelLocalCUDAGraphDescriptor(2)] * 3


def test_successful_lazy_miss_registers_descriptor_and_replays_current_call(monkeypatch) -> None:
    component, routine = _component("decode", 2, capture_mode=ModelLocalCaptureMode.PRECAPTURE_LAZY)
    manager = _TestManager(config={"decode": {}})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(manager, "_runtime_capture_scope", lambda: nullcontext())

    value = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(component(value), value * 2)
    assert len(manager.managed_components["decode"].entries) == 2
    assert manager.capture_attempts[("decode", 3)] == 1
    assert routine.eager_calls == 0


@pytest.mark.parametrize("mode", [ModelLocalCaptureMode.PRECAPTURE_LAZY, ModelLocalCaptureMode.PURE_LAZY])
def test_zero_lazy_graph_limit_keeps_all_captured_descriptors(monkeypatch, mode) -> None:
    component, routine = _component("decode", 2, capture_mode=mode)
    manager = _TestManager(config={"decode": {}})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(manager, "_runtime_capture_scope", nullcontext)

    for size in (3, 4, 5):
        value = torch.ones(size)
        assert torch.equal(component(value), value * 2)

    assert manager.managed_components["decode"].max_graphs is None
    expected = {2, 3, 4, 5} if mode is ModelLocalCaptureMode.PRECAPTURE_LAZY else {3, 4, 5}
    assert {descriptor.variant for descriptor in component.available_descriptors} == expected
    assert routine.eager_calls == 0


def test_lazy_capture_rejects_nested_outer_capture(monkeypatch) -> None:
    component, routine = _component("decode", 2, capture_mode=ModelLocalCaptureMode.PRECAPTURE_LAZY)
    manager = _TestManager(config={"decode": {"max_extra_graphs": 1}})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    value = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(component(value), value * 2)
    assert routine.eager_calls == 1
    assert manager.capture_attempts[("decode", 3)] == 0


def test_lazy_capacity_falls_back_without_evicting_startup_entries(monkeypatch) -> None:
    component, routine = _component("decode", 2, 3, capture_mode=ModelLocalCaptureMode.PRECAPTURE_LAZY)
    manager = _TestManager(config={"decode": {"max_extra_graphs": 1}})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(manager, "_runtime_capture_scope", lambda: nullcontext())

    component(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    component(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    entries = manager.managed_components["decode"].entries
    assert [descriptor.variant for descriptor in entries] == [3, 2, 4]
    assert component.available_descriptors == frozenset(
        {
            ModelLocalCUDAGraphDescriptor(3),
            ModelLocalCUDAGraphDescriptor(2),
            ModelLocalCUDAGraphDescriptor(4),
        }
    )
    assert manager.capture_attempts[("decode", 5)] == 0
    assert routine.eager_calls == 1


def test_binding_failure_propagates() -> None:
    first, _ = _component("first", 2)
    second, _ = _component("second", 2)
    manager = _TestManager()
    manager.prepare(_Model((first, second)))

    captured_graphs = []
    original_capture = manager.capture_entry

    def capture(component, descriptor):
        entry = original_capture(component, descriptor)
        captured_graphs.append(entry.graph)
        return entry

    manager.capture_entry = capture
    original_bind = second._bind_handle

    def fail_bind(handle) -> None:
        original_bind(handle)
        raise RuntimeError("bind failed")

    second._bind_handle = fail_bind
    with pytest.raises(RuntimeError, match="bind failed"):
        manager.capture_and_bind()

    assert first._bound_handle is None
    assert second._bound_handle is None
    assert manager.managed_components == {}
    assert all(graph.reset_called for graph in captured_graphs)


def test_startup_capture_failure_releases_previous_entries() -> None:
    component, _ = _component("decode", 2, 3)
    manager = _TestManager()
    manager.prepare(_Model((component,)))
    captured_graphs = []
    original_capture = manager.capture_entry

    def capture(component, descriptor):
        if descriptor.variant == 2:
            raise RuntimeError("second startup capture failed")
        entry = original_capture(component, descriptor)
        captured_graphs.append(entry.graph)
        return entry

    manager.capture_entry = capture
    with pytest.raises(RuntimeError, match="second startup capture failed"):
        manager.capture_and_bind()

    assert component._bound_handle is None
    assert manager.managed_components == {}
    assert len(captured_graphs) == 1
    assert captured_graphs[0].reset_called


def test_prepare_and_capture_are_single_use_lifecycle_operations() -> None:
    component, _ = _component("decode", 2)
    manager = _TestManager()
    model = _Model((component,))
    manager.prepare(model)
    with pytest.raises(RuntimeError, match=r"prepare\(\).*more than once"):
        manager.prepare(model)

    manager.capture_and_bind()
    with pytest.raises(RuntimeError, match="capture has already been attempted"):
        manager.capture_and_bind()


def test_descriptor_rejection_isolated_to_sibling_component() -> None:
    first, _ = _component("first", 2, 3)
    second, _ = _component("second", 4)
    manager = _TestManager()
    first.validate_descriptor = lambda descriptor: descriptor.variant != 2
    manager.prepare(_Model((first, second)))
    manager.capture_and_bind()

    assert list(manager.managed_components["first"].entries) == [ModelLocalCUDAGraphDescriptor(3)]
    assert list(manager.managed_components["second"].entries) == [ModelLocalCUDAGraphDescriptor(4)]


def test_capture_failure_propagates() -> None:
    class _FailingRoutine(_Routine):
        def allocate_buffers(self, descriptor, device):
            del descriptor, device
            raise RuntimeError("capture allocation failed")

    component = ModelLocalCUDAGraphComponent(
        "decode",
        _FailingRoutine(),
        [ModelLocalCUDAGraphDescriptor(2)],
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_local_cudagraph={"decode": {}}),
        compilation_config=SimpleNamespace(cudagraph_num_of_warmups=0),
    )
    manager = ModelLocalCUDAGraphManager(vllm_config=vllm_config, device=torch.device("cpu"))
    manager.prepare(_Model((component,)))

    with pytest.raises(RuntimeError, match="capture allocation failed"):
        manager.capture_and_bind()


def test_capture_and_bind_reports_total_memory_delta(monkeypatch):
    component, _ = _component("decode", 2)
    manager = _TestManager()
    manager.device = torch.device("cuda")
    free_memory = iter((100, 70))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(manager, "_synchronized_free_memory", lambda: next(free_memory))

    def capture(component, descriptor):
        buffers = component.routine.allocate_buffers(descriptor, torch.device("cpu"))
        output = component.routine.forward_for_capture(buffers)
        return ModelLocalCUDAGraphEntry(
            descriptor=descriptor,
            graph=cast(torch.cuda.CUDAGraph, _Graph(buffers)),
            buffers=buffers,
            captured_output=output,
        )

    monkeypatch.setattr(manager, "capture_entry", capture)
    manager.prepare(_Model((component,)))

    assert manager.capture_and_bind() == 30


def test_capture_and_bind_orders_descriptors_largest_first():
    component, _ = _component("decode", 2, 8, 4, capture_order_key=lambda descriptor: descriptor.variant)
    manager = _TestManager()
    manager.prepare(_Model((component,)))

    manager.capture_and_bind()

    assert [variant for component_id, variant in manager.capture_attempts if component_id == "decode"] == [8, 4, 2]


def test_profile_memory_uses_first_capture_and_per_graph_increment(monkeypatch):
    first, _ = _component(
        "first",
        2,
        3,
        4,
        capture_order_key=lambda descriptor: descriptor.variant,
        capture_mode=ModelLocalCaptureMode.PRECAPTURE_LAZY,
    )
    second, _ = _component("second", 5)
    manager = _TestManager(config={"first": {"max_extra_graphs": 2}, "second": {}})
    manager.prepare(_Model((first, second)))
    free_memory = iter((100, 90, 90, 87, 87, 80))
    captured = []
    capture_pools = []
    captured_variants = []

    def capture(component, descriptor, *, graph_pool=None):
        del component
        capture_pools.append(graph_pool)
        captured_variants.append(descriptor.variant)
        buffers = _Buffers(torch.zeros(1), torch.zeros(1))
        entry = ModelLocalCUDAGraphEntry(
            descriptor=descriptor,
            graph=cast(torch.cuda.CUDAGraph, _Graph(buffers)),
            buffers=buffers,
            captured_output=buffers.output,
        )
        captured.append(entry)
        return entry

    monkeypatch.setattr(manager, "capture_entry", capture)
    profiling_pool = object()
    monkeypatch.setattr(current_platform, "graph_pool_handle", lambda: profiling_pool)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(manager, "_synchronized_free_memory", lambda: next(free_memory))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    # first: 10 + max(3, 1 MiB) * (3 - 1 + 2); second: 7.
    assert manager.profile_memory() == 10 + (1 << 20) * 4 + 7
    assert len(captured) == 3
    assert capture_pools == [profiling_pool] * 3
    assert captured_variants == [4, 3, 5]
    assert all(entry.graph.reset_called for entry in captured)


def test_precapture_rejects_lazy_graph_capacity() -> None:
    component, _ = _component("decode", 2)
    manager = _TestManager(config={"decode": {"max_extra_graphs": 4}})
    with pytest.raises(ValueError, match="requires a lazy capture mode"):
        manager.prepare(_Model((component,)))
