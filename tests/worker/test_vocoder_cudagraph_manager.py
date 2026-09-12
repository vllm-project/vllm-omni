# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import logging
from collections import Counter, OrderedDict
from collections.abc import Callable, Set
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace
from typing import Any, NamedTuple, cast
from unittest.mock import patch

import pytest
import torch

from vllm_omni.model_executor.models.interfaces.vocoder_cudagraph import (
    BaseVocoderCUDAGraphRoutine,
    VocoderCUDAGraphComponent,
    VocoderCUDAGraphDescriptor,
    VocoderGraphHandle,
    VocoderRuntimeKey,
    VocoderRuntimeResolution,
)
from vllm_omni.worker.vocoder_cudagraph_manager import (
    ManagedComponent,
    VocoderCUDAGraphEntry,
    VocoderCUDAGraphManager,
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


class _Routine(BaseVocoderCUDAGraphRoutine):
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
        available: Set[VocoderCUDAGraphDescriptor],
    ) -> VocoderRuntimeResolution:
        del kwargs
        size = int(args[0].numel())
        descriptor = min(
            (item for item in available if isinstance(item.variant, int) and item.variant >= size),
            key=lambda item: item.variant if isinstance(item.variant, int) else 0,
            default=None,
        )
        return VocoderRuntimeResolution(VocoderRuntimeKey(size), descriptor)

    def allocate_buffers(self, descriptor: VocoderCUDAGraphDescriptor, device: torch.device) -> _Buffers:
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


class _TestManager(VocoderCUDAGraphManager):
    def __init__(self, *, config: dict[str, Any] | None = None) -> None:
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(vocoder_cudagraph_config=config),
            compilation_config=SimpleNamespace(cudagraph_num_of_warmups=0),
        )
        super().__init__(vllm_config=vllm_config, device=torch.device("cpu"))
        self.components_during_capture: list[tuple[bool, ...]] = []
        self.fail_replay_for: set[tuple[str, object]] = set()
        self.fail_capture_for: set[tuple[str, object]] = set()
        self.capture_attempts: Counter[tuple[str, object]] = Counter()
        self.capture_started: Event | None = None
        self.capture_release: Event | None = None

    def capture_entry(
        self,
        component: VocoderCUDAGraphComponent,
        descriptor: VocoderCUDAGraphDescriptor,
    ) -> VocoderCUDAGraphEntry | None:
        self.components_during_capture.append(tuple(item._bound_handle is not None for item in self.components))
        key = (component.component_id, descriptor.variant)
        self.capture_attempts[key] += 1
        if key == ("decode", 3) and self.capture_started is not None and self.capture_release is not None:
            self.capture_started.set()
            self.capture_release.wait(timeout=5)
        if key in self.fail_capture_for:
            return None
        buffers = component.routine.allocate_buffers(descriptor, self.device)
        assert isinstance(buffers, _Buffers)
        output = component.routine.forward_for_capture(buffers)
        graph = _Graph(
            buffers,
            fail=(component.component_id, descriptor.variant) in self.fail_replay_for,
        )
        return VocoderCUDAGraphEntry(
            descriptor=descriptor,
            graph=cast(torch.cuda.CUDAGraph, graph),
            buffers=buffers,
            captured_output=output,
        )


class _Model:
    supports_vocoder_cudagraph = True
    vocoder_cudagraph_shared_config_keys = frozenset({"shared_shape_policy"})

    def __init__(self, components: tuple[VocoderCUDAGraphComponent, ...]) -> None:
        self.components = components

    def get_vocoder_cudagraph_components(self) -> tuple[VocoderCUDAGraphComponent, ...]:
        return self.components


def _component(component_id: str, *sizes: int) -> tuple[VocoderCUDAGraphComponent, _Routine]:
    routine = _Routine()
    component = VocoderCUDAGraphComponent(
        component_id,
        routine,
        [VocoderCUDAGraphDescriptor(size) for size in sizes],
        supported_config_keys=frozenset({"bucket_policy"}),
    )
    return component, routine


def test_handle_exposes_runtime_call_and_read_only_graph_coverage() -> None:
    handle = VocoderGraphHandle(lambda value, *, offset=0: value + offset)

    assert handle(2, offset=3) == 5
    assert handle.available_descriptors == frozenset()
    assert not hasattr(handle, "capture")
    assert not hasattr(handle, "replay")
    assert not hasattr(handle, "entries")


def test_runtime_lazy_capture_logs_only_new_entries(monkeypatch) -> None:
    component, _ = _component("decode", 2)
    manager = _TestManager()
    descriptor = VocoderCUDAGraphDescriptor(3)
    fake_buffers = _Buffers(torch.zeros(1), torch.zeros(1))
    fake_entry = VocoderCUDAGraphEntry(
        descriptor=descriptor,
        graph=cast(torch.cuda.CUDAGraph, _Graph(fake_buffers)),
        buffers=fake_buffers,
        captured_output=fake_buffers.output,
    )
    managed = ManagedComponent(
        component=component,
        entries=OrderedDict(),
        enable_lazy_capture=True,
        max_graphs=1,
    )
    calls: list[VocoderCUDAGraphDescriptor] = []

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
        "Lazy-captured vocoder CUDA Graph Component %s Descriptor %r",
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
    assert isinstance(first._bound_handle, VocoderGraphHandle)
    assert isinstance(second._bound_handle, VocoderGraphHandle)
    assert first.available_descriptors == frozenset({VocoderCUDAGraphDescriptor(4)})
    assert second.available_descriptors == frozenset({VocoderCUDAGraphDescriptor(8)})
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


def test_coverage_miss_falls_back_but_validation_and_replay_errors_propagate() -> None:
    component, routine = _component("decode", 2)
    manager = _TestManager(config={"log_stats": True})
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
    manager = _TestManager(config={"log_stats": True})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()

    def fail_copy(*_args, **_kwargs):
        raise RuntimeError("copy failed")

    routine.copy_runtime_inputs = fail_copy
    with pytest.raises(RuntimeError, match="copy failed"):
        component(torch.tensor([1.0]))
    assert routine.eager_calls == 0

    component, routine = _component("decode-postprocess", 2)
    manager = _TestManager(config={"log_stats": True})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()

    def fail_output(*_args, **_kwargs):
        raise RuntimeError("postprocess failed")

    routine.output_after_replay = fail_output
    with pytest.raises(RuntimeError, match="postprocess failed"):
        component(torch.tensor([1.0]))
    assert routine.eager_calls == 0


def test_config_validation_catches_unknown_shared_component_and_extension_keys() -> None:
    component, _ = _component("decode", 2)

    manager = _TestManager(config={"unknown": 1})
    with pytest.raises(ValueError, match="Unknown vocoder_cudagraph"):
        manager.prepare(_Model((component,)))

    manager = _TestManager(config={"components": {"missing": {"enabled": False}}})
    with pytest.raises(ValueError, match="Unknown vocoder CUDA Graph component"):
        manager.prepare(_Model((component,)))

    manager = _TestManager(config={"components": {"decode": {"unknown_bucket_policy": [2]}}})
    with pytest.raises(ValueError, match="Unknown config key"):
        manager.prepare(_Model((component,)))


def test_component_registry_rejects_duplicate_ids_and_descriptors() -> None:
    first, _ = _component("decode", 2)
    duplicate_id, _ = _component("decode", 3)
    manager = _TestManager()
    with pytest.raises(ValueError, match="Duplicate vocoder CUDA Graph component_id"):
        manager.prepare(_Model((first, duplicate_id)))

    duplicate_descriptor = VocoderCUDAGraphComponent(
        "duplicate",
        _Routine(),
        [VocoderCUDAGraphDescriptor(2), VocoderCUDAGraphDescriptor(2)],
    )
    manager = _TestManager()
    with pytest.raises(ValueError, match="Duplicate Descriptor"):
        manager.prepare(_Model((duplicate_descriptor,)))


def test_disabled_component_remains_on_original_eager_callable() -> None:
    component, routine = _component("decode", 2)
    manager = _TestManager(config={"components": {"decode": {"enabled": False}}})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()

    assert not manager.managed_components
    assert component._bound_handle is None
    component(torch.tensor([1.0]))
    assert routine.eager_calls == 1


def test_failed_descriptor_is_negative_cached_across_runtime_misses() -> None:
    component, routine = _component("decode", 2)
    manager = _TestManager(
        config={
            "components": {
                "decode": {
                    "enable_lazy_capture": True,
                    "max_extra_graphs": 1,
                }
            }
        }
    )
    manager.fail_capture_for.add(("decode", 2))
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()

    value = torch.tensor([1.0, 2.0])
    assert torch.equal(component(value), value * 2)
    assert torch.equal(component(value), value * 2)
    assert routine.eager_calls == 2
    assert manager.capture_attempts[("decode", 2)] == 1


def test_successful_lazy_miss_registers_descriptor_and_replays_current_call(monkeypatch) -> None:
    component, routine = _component("decode", 2)
    manager = _TestManager(config={"components": {"decode": {"enable_lazy_capture": True, "max_extra_graphs": 1}}})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(manager, "_runtime_capture_scope", lambda: nullcontext())

    value = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(component(value), value * 2)
    assert len(manager.managed_components["decode"].entries) == 2
    assert manager.capture_attempts[("decode", 3)] == 1
    assert routine.eager_calls == 0


def test_lazy_capture_rejects_nested_outer_capture(monkeypatch) -> None:
    component, routine = _component("decode", 2)
    manager = _TestManager(config={"components": {"decode": {"enable_lazy_capture": True, "max_extra_graphs": 1}}})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    value = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(component(value), value * 2)
    assert routine.eager_calls == 1
    assert manager.capture_attempts[("decode", 3)] == 0


def test_lazy_entries_share_one_lru_with_startup_entries(monkeypatch) -> None:
    component, _ = _component("decode", 2, 3)
    manager = _TestManager(config={"components": {"decode": {"enable_lazy_capture": True, "max_extra_graphs": 1}}})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(manager, "_runtime_capture_scope", lambda: nullcontext())

    component(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    component(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    entries = manager.managed_components["decode"].entries
    assert [descriptor.variant for descriptor in entries] == [3, 4, 5]
    assert component.available_descriptors == frozenset(
        {
            VocoderCUDAGraphDescriptor(3),
            VocoderCUDAGraphDescriptor(4),
            VocoderCUDAGraphDescriptor(5),
        }
    )


def test_concurrent_lazy_misses_capture_one_entry(monkeypatch) -> None:
    component, _ = _component("decode", 2)
    manager = _TestManager(config={"components": {"decode": {"enable_lazy_capture": True, "max_extra_graphs": 1}}})
    manager.prepare(_Model((component,)))
    manager.capture_and_bind()
    manager.capture_started = Event()
    manager.capture_release = Event()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(manager, "_runtime_capture_scope", lambda: nullcontext())

    value = torch.tensor([1.0, 2.0, 3.0])
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(component, value) for _ in range(2)]
        assert manager.capture_started.wait(timeout=5)
        manager.capture_release.set()
        assert [future.result(timeout=5).tolist() for future in futures] == [[2.0, 4.0, 6.0]] * 2

    assert manager.capture_attempts[("decode", 3)] == 1


def test_binding_failure_propagates() -> None:
    first, _ = _component("first", 2)
    second, _ = _component("second", 2)
    manager = _TestManager()
    manager.prepare(_Model((first, second)))

    def fail_bind(_handle) -> None:
        raise RuntimeError("bind failed")

    first._bind_handle = fail_bind
    with pytest.raises(RuntimeError, match="bind failed"):
        manager.capture_and_bind()


def test_prepare_and_capture_are_single_use_lifecycle_operations() -> None:
    component, _ = _component("decode", 2)
    manager = _TestManager()
    model = _Model((component,))
    manager.prepare(model)
    with pytest.raises(RuntimeError, match=r"prepare\(\).*more than once"):
        manager.prepare(model)

    manager.capture_and_bind()
    with pytest.raises(RuntimeError, match="capture has already completed"):
        manager.capture_and_bind()


def test_capture_failure_isolated_to_descriptor_and_sibling_component() -> None:
    first, _ = _component("first", 2, 3)
    second, _ = _component("second", 4)
    manager = _TestManager()
    manager.fail_capture_for.add(("first", 2))
    manager.prepare(_Model((first, second)))
    manager.capture_and_bind()

    assert list(manager.managed_components["first"].entries) == [VocoderCUDAGraphDescriptor(3)]
    assert list(manager.managed_components["second"].entries) == [VocoderCUDAGraphDescriptor(4)]


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
        return VocoderCUDAGraphEntry(
            descriptor=descriptor,
            graph=cast(torch.cuda.CUDAGraph, _Graph(buffers)),
            buffers=buffers,
            captured_output=output,
        )

    monkeypatch.setattr(manager, "capture_entry", capture)
    manager.prepare(_Model((component,)))

    assert manager.capture_and_bind() == 30


def test_profile_memory_uses_first_capture_and_per_graph_increment(monkeypatch):
    first, _ = _component("first", 2, 3, 4)
    second, _ = _component("second", 5)
    manager = _TestManager(config={"components": {"first": {"enable_lazy_capture": True, "max_extra_graphs": 2}}})
    manager.prepare(_Model((first, second)))
    free_memory = iter((100, 90, 90, 87, 87, 80))
    captured = []

    def capture(component, descriptor):
        del component
        buffers = _Buffers(torch.zeros(1), torch.zeros(1))
        entry = VocoderCUDAGraphEntry(
            descriptor=descriptor,
            graph=cast(torch.cuda.CUDAGraph, _Graph(buffers)),
            buffers=buffers,
            captured_output=buffers.output,
        )
        captured.append(entry)
        return entry

    monkeypatch.setattr(manager, "capture_entry", capture)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(manager, "_synchronized_free_memory", lambda: next(free_memory))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    # first: 10 + max(3, 1 MiB) * (3 - 1 + 2); second: 7.
    assert manager.profile_memory() == 10 + (1 << 20) * 4 + 7
    assert len(captured) == 3
    assert all(entry.graph.reset_called for entry in captured)


def test_profile_memory_ignores_extra_graphs_when_lazy_capture_is_disabled(monkeypatch):
    component, _ = _component("decode", 2)
    manager = _TestManager(config={"components": {"decode": {"max_extra_graphs": 4}}})
    manager.prepare(_Model((component,)))
    buffers = _Buffers(torch.zeros(1), torch.zeros(1))
    entry = VocoderCUDAGraphEntry(
        descriptor=component.descriptors[0],
        graph=cast(torch.cuda.CUDAGraph, _Graph(buffers)),
        buffers=buffers,
        captured_output=buffers.output,
    )
    monkeypatch.setattr(manager, "capture_entry", lambda *args, **kwargs: entry)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    free_memory = iter((100, 90))
    monkeypatch.setattr(manager, "_synchronized_free_memory", lambda: next(free_memory))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    assert manager.profile_memory() == 10
