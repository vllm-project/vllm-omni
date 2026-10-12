# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.lychee_fd import audio_cudnn as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class FakeLibrary:
    def __init__(self, *, finalize_failure=False, workspace_bytes=1024):
        self.descriptors = {}
        self.destroyed: list[int | None] = []
        self.destroyed_handles: list[int | None] = []
        self.finalize_failure = finalize_failure
        self.workspace_bytes = workspace_bytes
        self.execute_failure = False
        self.streams: list[int] = []
        self.cudnnGetVersion = Function(lambda: 92000)
        self.cudnnGetErrorString = Function(lambda status: b"controlled backend failure")
        self.cudnnCreate = Function(self.create_handle)
        self.cudnnDestroy = Function(self.destroy_handle)
        self.cudnnSetStream = Function(self.set_stream)
        self.cudnnBackendCreateDescriptor = Function(self.create_descriptor)
        self.cudnnBackendDestroyDescriptor = Function(self.destroy_descriptor)
        self.cudnnBackendSetAttribute = Function(lambda *args: 0)
        self.cudnnBackendGetAttribute = Function(self.get_attribute)
        self.cudnnBackendFinalize = Function(self.finalize)
        self.cudnnBackendExecute = Function(lambda *args: 1 if self.execute_failure else 0)

    def destroy_handle(self, handle: ctypes.c_void_p) -> int:
        self.destroyed_handles.append(handle.value)
        return 0

    def set_stream(self, handle: ctypes.c_void_p, stream: int) -> int:
        self.streams.append(stream)
        return 0

    def destroy_descriptor(self, descriptor: ctypes.c_void_p) -> int:
        self.destroyed.append(descriptor.value)
        return 0

    def create_handle(self, result):
        ctypes.cast(result, ctypes.POINTER(ctypes.c_void_p))[0] = 77
        return 0

    def create_descriptor(self, kind, result):
        index = len(self.descriptors) + 1
        self.descriptors[index] = kind
        ctypes.cast(result, ctypes.POINTER(ctypes.c_void_p))[0] = index
        return 0

    def get_attribute(self, descriptor, attribute, typ, size, count, result):
        ctypes.cast(count, ctypes.POINTER(ctypes.c_int64))[0] = 1
        ctypes.cast(result, ctypes.POINTER(ctypes.c_int64))[0] = self.workspace_bytes
        return 0

    def finalize(self, descriptor):
        return int(self.finalize_failure and self.descriptors[descriptor.value] == module._Descriptor.EXECUTION_PLAN)


class FakeTensor:
    def __init__(self, shape, device=torch.device("cuda", 0)):
        self.shape = shape
        self.device = device
        self.dtype = torch.bfloat16
        self.stream_records = []

    def contiguous(self):
        return self

    def record_stream(self, stream):
        self.stream_records.append(stream)

    def data_ptr(self):
        return id(self)


class FakeStream:
    cuda_stream = 123

    def __init__(self):
        self.waited = []
        self.synchronized = 0

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        self.synchronized += 1


class FakeEvent:
    def __init__(self):
        self.synchronized = 0
        self.recorded = None

    def record(self, stream):
        self.recorded = stream

    def synchronize(self):
        self.synchronized += 1


def fake_backend(monkeypatch, lib=None):
    lib = lib or FakeLibrary()
    stream = FakeStream()
    monkeypatch.setattr(module, "_load_torch_cudnn", lambda: (lib, "Torch/current/libcudnn.so.9"))
    monkeypatch.setattr(torch.accelerator, "device_index", lambda device: nullcontext())
    monkeypatch.setattr(module, "current_omni_platform", SimpleNamespace(get_device_capability=lambda device: (8, 0)))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch, "empty", lambda shape, **kwargs: FakeTensor(shape))
    return lib, stream


def test_cpu_and_tiny_config_keep_dtype_and_checkpoint_keys_without_loading_cudnn(monkeypatch):
    def forbidden():
        raise AssertionError("CPU path loaded cuDNN")

    monkeypatch.setattr(module, "_load_torch_cudnn", forbidden)
    conv = module.LycheeAudioConv1d(4, 4, 3, stride=2, padding=1).bfloat16()
    inputs = torch.randn(2, 4, 10)
    actual = conv(inputs)
    expected = F.conv1d(inputs, conv.weight.float(), conv.bias.float(), stride=2, padding=1)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    assert set(conv.state_dict()) == {"weight", "bias"}
    assert conv._released_plan is None
    conv.close()
    conv.close()


def test_plan_rejects_nonexplicit_cuda_device_before_library_load(monkeypatch):
    monkeypatch.setattr(module, "_load_torch_cudnn", lambda: pytest.fail("must not load cuDNN"))
    for device in ("cpu", "cuda"):
        with pytest.raises(ValueError, match="explicit CUDA"):
            module.LycheeAudioCudnnPlan(torch.device(device))


def test_wrong_cudnn_version_fails_before_library_search(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 91900)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: pytest.fail("must not load another cuDNN"))
    with pytest.raises(RuntimeError, match="requires cuDNN9.20.0"):
        module._load_torch_cudnn()


def test_plan_close_releases_each_descriptor_and_handle_once_in_reverse_order(monkeypatch):
    lib, _ = fake_backend(monkeypatch)
    plan = module.LycheeAudioCudnnPlan(torch.device("cuda", 0))
    allocated = [descriptor.value for descriptor in plan._descriptors]
    plan.close()
    plan.close()
    assert lib.destroyed == allocated[::-1]
    assert lib.destroyed_handles == [77]
    assert plan.workspace is None


def test_constructor_finalization_failure_rolls_back_handle_and_all_descriptors(monkeypatch):
    lib, _ = fake_backend(monkeypatch, FakeLibrary(finalize_failure=True))
    with pytest.raises(RuntimeError, match="controlled backend failure"):
        module.LycheeAudioCudnnPlan(torch.device("cuda", 0))
    assert lib.destroyed == list(lib.descriptors)[::-1]
    assert lib.destroyed_handles == [77]


def test_workspace_growth_fails_closed_and_releases_constructor_resources(monkeypatch):
    lib, _ = fake_backend(monkeypatch, FakeLibrary(workspace_bytes=module.LycheeAudioCudnnPlan.MAX_WORKSPACE_BYTES + 1))
    with pytest.raises(RuntimeError, match="workspace exceeds bound"):
        module.LycheeAudioCudnnPlan(torch.device("cuda", 0))
    assert lib.destroyed == list(lib.descriptors)[::-1]
    assert lib.destroyed_handles == [77]


def test_plan_stream_dependency_and_tensor_lifetime_are_owned_until_close(monkeypatch):
    lib, stream = fake_backend(monkeypatch)
    plan = module.LycheeAudioCudnnPlan(torch.device("cuda", 0))
    x, w = FakeTensor((1, 1280, 10)), FakeTensor((1280, 1280, 3))
    plan.execute(x, w, None)
    previous = plan._completion
    second = FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: second)
    plan.execute(x, w, None)
    assert second.waited == [previous]
    assert x.stream_records == [stream, second]
    assert w.stream_records == [stream, second]
    assert plan.workspace.stream_records == [stream, second]
    final = plan._completion
    plan.close()
    assert final.synchronized == 1
    assert lib.destroyed_handles == [77]


def test_execute_failure_fences_stream_then_releases_resources(monkeypatch):
    lib, stream = fake_backend(monkeypatch)
    plan = module.LycheeAudioCudnnPlan(torch.device("cuda", 0))
    lib.execute_failure = True
    with pytest.raises(RuntimeError, match="controlled backend failure"):
        plan.execute(FakeTensor((1, 1280, 10)), FakeTensor((1280, 1280, 3)), None)
    assert stream.synchronized == 1
    assert plan._closed
    assert len(lib.destroyed) == len(set(lib.destroyed)) == len(lib.descriptors)
    assert lib.destroyed_handles == [77]
    assert plan.workspace is None


def test_plan_rejects_device_and_shape_mismatch_before_launch(monkeypatch):
    lib, _ = fake_backend(monkeypatch)
    plan = module.LycheeAudioCudnnPlan(torch.device("cuda", 0))
    with pytest.raises(ValueError, match="B1,C1280,L10"):
        plan.execute(FakeTensor((2, 1280, 10)), FakeTensor((1280, 1280, 3)), None)
    with pytest.raises(ValueError, match="plan's CUDA device"):
        plan.execute(FakeTensor((1, 1280, 10), torch.device("cuda", 1)), FakeTensor((1280, 1280, 3)), None)
    assert not lib.streams
    plan.close()


def test_conv_explicit_close_is_idempotent():
    conv = module.LycheeAudioConv1d(4, 4, 3)
    calls = []
    conv._released_plan = SimpleNamespace(close=lambda: calls.append("closed"))
    conv.close()
    conv.close()
    assert calls == ["closed"]
    assert conv._released_plan is None


def test_workspace_allocation_failure_releases_native_constructor_resources(monkeypatch):
    lib, _ = fake_backend(monkeypatch)

    def fail_alloc(*args, **kwargs):
        raise RuntimeError("controlled workspace allocation failure")

    monkeypatch.setattr(torch, "empty", fail_alloc)
    with pytest.raises(RuntimeError, match="workspace allocation failure"):
        module.LycheeAudioCudnnPlan(torch.device("cuda", 0))
    assert lib.destroyed == list(lib.descriptors)[::-1]
    assert lib.destroyed_handles == [77]


def test_unproved_device_architecture_fails_before_handle_creation(monkeypatch):
    lib, _ = fake_backend(monkeypatch)
    monkeypatch.setattr(module.current_omni_platform, "get_device_capability", lambda device: (9, 0))
    with pytest.raises(RuntimeError, match="validated SM80"):
        module.LycheeAudioCudnnPlan(torch.device("cuda", 0))
    assert not lib.descriptors
    assert not lib.destroyed_handles


class SimulatedCudaInput:
    """CUDA routing metadata with CPU storage; these tests never launch CUDA."""

    is_cuda = True
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16

    def __init__(self, storage):
        self.storage = storage
        self.shape = storage.shape
        self.ndim = storage.ndim

    def __iter__(self):
        return iter(self.storage)


@pytest.mark.parametrize(
    ("capability", "cudnn_version", "cuda_platform"),
    [((9, 0), 92000, True), ((8, 0), 91900, True), (None, 92000, True), ((8, 0), 92000, False)],
)
def test_unmatched_runtime_uses_ordinary_convolution(monkeypatch, capability, cudnn_version, cuda_platform):
    monkeypatch.setattr(
        module,
        "current_omni_platform",
        SimpleNamespace(is_cuda=lambda: cuda_platform, get_device_capability=lambda device: capability),
    )
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: cudnn_version)
    monkeypatch.setattr(module, "LycheeAudioCudnnPlan", lambda device: pytest.fail("must not create released plan"))
    fallback_calls = []

    def ordinary_conv(conv, inputs, weight, bias):
        fallback_calls.append(inputs)
        return F.conv1d(inputs.storage, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups)

    monkeypatch.setattr(torch.nn.Conv1d, "_conv_forward", ordinary_conv)
    conv = module.LycheeAudioConv1d(1280, 1280, 3, stride=2, padding=1).bfloat16()
    storage = torch.randn(1, 1280, 10, dtype=torch.bfloat16)
    inputs = SimulatedCudaInput(storage)
    expected = F.conv1d(storage, conv.weight, conv.bias, stride=2, padding=1)
    actual = conv(inputs)
    torch.testing.assert_close(actual, expected)
    assert fallback_calls == [inputs]
    assert actual.shape == (1, 1280, 5)
    assert conv._released_plan is None
    conv.close()


def test_matched_runtime_keeps_row_owned_plan_and_close(monkeypatch):
    monkeypatch.setattr(
        module,
        "current_omni_platform",
        SimpleNamespace(is_cuda=lambda: True, get_device_capability=lambda device: (8, 0)),
    )
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 92000)
    plan_calls = []
    closed = []

    class Plan:
        def __init__(self, device):
            assert device == torch.device("cuda", 0)

        def execute(self, inputs, weight, bias):
            plan_calls.append(tuple(inputs.shape))
            return F.conv1d(inputs, weight, bias, stride=2, padding=1)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, "LycheeAudioCudnnPlan", Plan)
    conv = module.LycheeAudioConv1d(1280, 1280, 3, stride=2, padding=1).bfloat16()
    storage = torch.randn(2, 1280, 10, dtype=torch.bfloat16)
    expected = F.conv1d(storage, conv.weight, conv.bias, stride=2, padding=1)
    with torch.no_grad():
        actual = conv(SimulatedCudaInput(storage))
    torch.testing.assert_close(actual, expected)
    assert plan_calls == [(1, 1280, 10), (1, 1280, 10)]
    conv.close()
    conv.close()
    assert closed == [True]
