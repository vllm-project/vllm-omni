# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU ABI and rejection boundaries, without loading the NPU runtime."""

import importlib
import sys
from types import ModuleType

import msgspec
import pytest
import torch

from tests.distributed.weight_sharing.helpers import make_context

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def codec_module(contracts):
    return importlib.import_module(f"{contracts.__package__}.torch_npu_ipc_codec")


def _args():
    return (
        torch.Tensor,
        (64,),
        (1,),
        0,
        torch.storage.TypedStorage,
        torch.uint8,
        3,
        b"handle",
        64,
        256,
        False,
        b"counter",
        7,
        b"e" * 64,
        True,
    )


def test_npu_abi_and_consumer_local_device(codec_module, monkeypatch):
    calls = []
    module = ModuleType("torch_npu.multiprocessing.reductions")

    def rebuild(*args):
        calls.append(args)
        return "mapped"

    monkeypatch.setattr(module, "rebuild_npu_tensor", rebuild, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    descriptor = codec_module._encode_reduction(_args(), "s", 64)
    assert descriptor.codec == "npu-ipc-v1"
    assert descriptor.device_index == 3
    assert codec_module.TorchNpuIpcCodec().import_storage(descriptor, 1) == "mapped"
    expected = list(_args())
    expected[6] = 1
    assert calls == [tuple(expected)]


@pytest.mark.parametrize(
    "index,value",
    [
        (0, torch.nn.Parameter),
        (1, (32,)),
        (2, (2,)),
        (3, 1),
        (4, torch.UntypedStorage),
        (5, torch.float32),
        (8, 32),
        (10, True),
    ],
)
def test_reject_unknown_reducer_abi(codec_module, contracts, index, value):
    args = list(_args())
    args[index] = value
    with pytest.raises(contracts.UnsupportedError, match="ABI"):
        codec_module._encode_reduction(tuple(args), "s", 64)


@pytest.mark.parametrize(
    "changes",
    [
        {"codec": "cuda-ipc-v1"},
        {"storage_id": ""},
        {"device_index": -1},
        {"device_index": True},
        {"storage_nbytes": 0},
        {"allocation_offset_bytes": -1},
        {"ref_counter_offset": -1},
        {"handle": b""},
        {"handle": b"x" * 4097},
        {"ref_counter_handle": b""},
        {"event_handle": b""},
        {"event_handle": b"e"},
        {"event_handle": b"e" * 63},
        {"event_handle": b"e" * 65},
        {"event_handle": b"e" * 4096},
        {"event_sync_required": 1},
    ],
)
def test_reject_bad_descriptor_before_native_import(codec_module, contracts, monkeypatch, changes):
    descriptor = msgspec.structs.replace(codec_module._encode_reduction(_args(), "s", 64), **changes)
    # Any attempt to load the extension would fail, rather than masking a native call.
    monkeypatch.setitem(sys.modules, "torch_npu.multiprocessing.reductions", None)
    with pytest.raises((contracts.CompatibilityError, contracts.UnsupportedError)):
        codec_module.TorchNpuIpcCodec().import_storage(descriptor, 0)


def test_unused_event_handle_does_not_require_cann_event_size(codec_module):
    args = list(_args())
    args[13:15] = [b"unused", False]
    descriptor = codec_module._encode_reduction(tuple(args), "s", 64)
    assert descriptor.event_handle == b"unused"
    assert not descriptor.event_sync_required


@pytest.mark.parametrize("device", [-1, True, "0"])
def test_reject_invalid_consumer_device(codec_module, contracts, device):
    descriptor = codec_module._encode_reduction(_args(), "s", 64)
    with pytest.raises(contracts.CompatibilityError, match="device index"):
        codec_module.TorchNpuIpcCodec().import_storage(descriptor, device)


def test_cancel_uses_npu_counter_and_local_device(codec_module, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(
        torch.storage.TypedStorage,
        "_release_ipc_counter_npu",
        lambda *args, **kwargs: calls.append((args, kwargs)),
        raising=False,
    )
    codec_module.TorchNpuIpcCodec().cancel_unconsumed(codec_module._encode_reduction(_args(), "s", 64), 2)
    assert calls == [((b"counter", 7), {"device": 2})]


def test_runtime_checks_physical_identity_without_consumer_allocator(codec_module, contracts, monkeypatch):
    calls = []
    npu = ModuleType("torch_npu.npu")
    monkeypatch.setattr(npu, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(npu, "set_device", lambda index: calls.append(index), raising=False)
    module = ModuleType("torch_npu")
    monkeypatch.setattr(module, "npu", npu, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    monkeypatch.setattr(codec_module.sys, "platform", "linux")
    monkeypatch.setattr(codec_module, "_device_uuid", lambda index: "NPU-test")
    codec = codec_module.TorchNpuIpcCodec()
    with pytest.raises(contracts.CompatibilityError, match="physical NPU"):
        codec.validate_runtime(make_context(contracts))
    assert calls == []
    codec.validate_runtime(make_context(contracts, device_uuid="NPU-test"))
    assert calls == [0]


def test_non_linux_rejected_without_loading_npu(codec_module, contracts, monkeypatch):
    monkeypatch.setattr(codec_module.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    with pytest.raises(contracts.UnsupportedError, match="Linux"):
        codec_module.TorchNpuIpcCodec().validate_runtime(make_context(contracts))
