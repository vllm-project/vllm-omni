# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CPU storage/view tests and fixed CUDA ABI tests without a CUDA runtime."""

import gc
import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import ModuleType

import msgspec
import pytest
import torch

from tests.distributed.weight_sharing.helpers import make_context, make_tensor_specs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def codec_module(contracts):
    return importlib.import_module(f"{contracts.__package__}.torch_cuda_ipc_codec")


@pytest.fixture
def cuda_runtime(codec_module, monkeypatch):
    calls = []
    snapshot = {"allocator_settings": {"expandable_segments": False}, "segments": []}
    monkeypatch.setattr(codec_module.sys, "platform", "linux")
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda **kwargs: torch.device("cuda"))
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda index: calls.append(("select", index)))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda index: calls.append(("sync", index)))
    monkeypatch.setattr(codec_module, "_device_uuid", lambda index: "GPU-test")
    monkeypatch.setattr(torch._C, "_cuda_getAllocatorBackend", lambda: "native", raising=False)

    def memory_snapshot(*args):
        calls.append(("snapshot", None))
        return snapshot

    monkeypatch.setattr(torch._C, "_cuda_memorySnapshot", memory_snapshot, raising=False)
    return snapshot, calls


def _spec(contracts, bindings, aliases=None):
    return contracts.WeightSpec(
        "test-model", "checkpoint-1", "final-layout", make_tensor_specs(contracts, bindings, aliases)
    )


def _descriptor(contracts, codec_module):
    return contracts.StorageExport(
        "storage-0",
        codec_module._CODEC_ID,
        3,
        b"handle",
        64,
        256,
        b"counter",
        7,
        b"event",
        False,
    )


def test_storage_dedup_alias_mixed_dtype_and_noncontiguous_views_without_copy(contracts, codec_module):
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    parameter = torch.nn.Parameter(base[:2])
    floats = torch.arange(16, dtype=torch.float32)
    bindings = {
        "weight": parameter,
        "alias": parameter,
        "transpose": base[2:].T,
        "floats": floats,
        "bytes": floats.view(torch.uint8),
        "scalar": floats[4],
        "empty_view": floats[16:],
        "empty_storage": torch.empty(0),
        "overlap": floats.as_strided((3, 4), (0, 1)),
    }
    spec = _spec(contracts, bindings, {"alias": "weight"})
    prepared = codec_module._describe_bindings(bindings, spec)
    assert len(prepared.manifest.storages) == 3
    assert sum(storage.nbytes > 0 for storage in prepared.manifest.storages) == 2
    carrier = next(iter(prepared.carriers.values()))
    assert carrier.dtype == torch.uint8
    assert carrier.numel() == base.untyped_storage().nbytes()
    assert carrier.untyped_storage()._cdata == base.untyped_storage()._cdata
    assert not carrier.requires_grad

    restored = codec_module.TorchCudaIpcCodec().make_views(prepared.manifest, prepared.carriers)
    assert restored["alias"] is restored["weight"]
    for name, tensor in bindings.items():
        assert torch.equal(restored[name], tensor)
        assert restored[name].dtype == tensor.dtype
        assert restored[name].stride() == tensor.stride()
        assert restored[name].storage_offset() == tensor.storage_offset()
        assert restored[name].untyped_storage()._cdata == tensor.untyped_storage()._cdata


def test_distinct_storages_are_not_merged(contracts, codec_module):
    bindings = {"a": torch.ones(4), "b": torch.ones(4)}
    prepared = codec_module._describe_bindings(bindings, _spec(contracts, bindings))
    assert len(prepared.carriers) == 2


@pytest.mark.parametrize("kind", ["meta", "conjugate", "negative", "sparse", "subclass", "quantized"])
def test_unsupported_tensor_forms_fail_without_copy(contracts, codec_module, mocker, kind):
    if kind == "meta":
        tensor = torch.empty(4, device="meta")
    elif kind == "conjugate":
        tensor = torch.ones(4, dtype=torch.complex64).conj()
    elif kind == "negative":
        tensor = torch._neg_view(torch.ones(4))
    elif kind == "sparse":
        tensor = torch.ones(4).to_sparse()
    elif kind == "quantized":
        tensor = torch.quantize_per_tensor(torch.ones(4), scale=0.1, zero_point=0, dtype=torch.qint8)
    else:

        class TensorSubclass(torch.Tensor):
            pass

        tensor = torch.ones(4).as_subclass(TensorSubclass)
    spec = contracts.WeightSpec(
        "m", "r", "l", (contracts.TensorSpec("weight", str(tensor.dtype).removeprefix("torch."), (4,)),)
    )
    carrier = mocker.spy(codec_module, "_byte_carrier")
    with pytest.raises(contracts.UnsupportedError):
        codec_module._describe_bindings({"weight": tensor}, spec)
    carrier.assert_not_called()


def test_cpp_storage_lease_survives_python_carrier_and_external_view(codec_module):
    codec = codec_module.TorchCudaIpcCodec()
    carrier = codec_module._byte_carrier(torch.ones(8))
    lease = codec.weak_storage(carrier)
    external = torch.empty(0).set_(carrier.untyped_storage(), 0, (8,), (1,))
    del carrier
    gc.collect()
    assert not lease.expired()
    del external
    gc.collect()
    assert lease.expired()
    lease.close()
    lease.close()


@pytest.mark.parametrize("handle_size", [64, 8192])
def test_integer_device_reduction_schema(contracts, codec_module, handle_size):
    args = (
        torch.Tensor,
        (64,),
        (1,),
        0,
        torch.storage.TypedStorage,
        torch.uint8,
        3,
        b"h" * handle_size,
        64,
        256,
        False,
        b"counter",
        7,
        b"event",
        False,
    )
    descriptor = codec_module._encode_reduction(args, "storage-0", 64)
    assert descriptor == msgspec.structs.replace(_descriptor(contracts, codec_module), handle=b"h" * handle_size)
    incompatible = args[:6] + (torch.device("cuda", 3),) + args[7:]
    with pytest.raises(contracts.CompatibilityError, match="device or offset"):
        codec_module._encode_reduction(incompatible, "storage-0", 64)


@pytest.mark.parametrize("handle_size", [64, 8192, 1024 * 1024])
def test_import_remaps_to_local_device_and_uses_fixed_rebuild(contracts, codec_module, mocker, handle_size):
    rebuild = mocker.patch.object(torch.multiprocessing.reductions, "rebuild_cuda_tensor")
    codec = codec_module.TorchCudaIpcCodec()
    descriptor = msgspec.structs.replace(_descriptor(contracts, codec_module), handle=b"h" * handle_size)
    descriptor = msgspec.msgpack.decode(msgspec.msgpack.encode(descriptor), type=contracts.StorageExport)
    assert codec.import_storage(descriptor, 0) is rebuild.return_value
    rebuild.assert_called_once()
    args = rebuild.call_args.args
    assert args[0] is torch.Tensor
    assert args[4] is torch.storage.TypedStorage
    assert args[6] == 0  # provider used ordinal 3
    assert args[7] == descriptor.handle
    assert args[9] == 256
    assert args[12] == 7


def test_rebuild_error_is_propagated_without_counter_cancellation(contracts, codec_module, mocker):
    mocker.patch.object(
        torch.multiprocessing.reductions, "rebuild_cuda_tensor", side_effect=RuntimeError("possibly consumed")
    )
    release = mocker.patch.object(torch.storage.TypedStorage, "_release_ipc_counter")
    with pytest.raises(RuntimeError, match="possibly consumed"):
        codec_module.TorchCudaIpcCodec().import_storage(_descriptor(contracts, codec_module), 0)
    release.assert_not_called()


def test_cancel_uses_only_the_supplied_unconsumed_counter(contracts, codec_module, mocker):
    release = mocker.patch.object(torch.storage.TypedStorage, "_release_ipc_counter")
    codec_module.TorchCudaIpcCodec().cancel_unconsumed(_descriptor(contracts, codec_module), 0)
    release.assert_called_once_with(b"counter", 7, device=0)


def test_consumer_checks_runtime_once_and_sync_does_not_rescan(contracts, codec_module, cuda_runtime):
    _, calls = cuda_runtime
    context = make_context(contracts)
    codec = codec_module.TorchCudaIpcCodec()
    codec.validate_runtime(context)
    codec.synchronize(context)
    codec.synchronize(context)
    assert calls == [("select", 0), ("snapshot", None), ("sync", 0), ("sync", 0)]


@pytest.mark.parametrize("invalid", ["gpu", "allocator"])
def test_consumer_runtime_rejects_unsafe_environment(contracts, codec_module, cuda_runtime, monkeypatch, invalid):
    if invalid == "gpu":
        monkeypatch.setattr(codec_module, "_device_uuid", lambda index: "GPU-other")
    else:
        monkeypatch.setattr(torch._C, "_cuda_getAllocatorBackend", lambda: "cudaMallocAsync")
    with pytest.raises((contracts.CompatibilityError, contracts.UnsupportedError)):
        codec_module.TorchCudaIpcCodec().validate_runtime(make_context(contracts))


@pytest.mark.parametrize("codec", ["unknown-codec", "cuda-ipc-v2"])
def test_incompatible_codec_is_rejected_before_rebuild(contracts, codec_module, mocker, codec):
    descriptor = msgspec.structs.replace(_descriptor(contracts, codec_module), codec=codec)
    rebuild = mocker.patch.object(torch.multiprocessing.reductions, "rebuild_cuda_tensor")
    with pytest.raises(contracts.UnsupportedError, match="codec"):
        codec_module.TorchCudaIpcCodec().import_storage(descriptor, 0)
    rebuild.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"storage_id": ""},
        {"device_index": -1},
        {"storage_nbytes": 0},
        {"allocation_offset_bytes": -1},
        {"ref_counter_offset": -1},
        {"handle": b""},
        {"handle": b"h" * (1024 * 1024 + 1)},
        {"ref_counter_handle": b""},
        {"ref_counter_handle": b"r" * 4097},
        {"event_handle": b""},
        {"event_handle": b"e" * 4097},
        {"event_sync_required": 1},
    ],
)
def test_invalid_descriptor_rejected_before_rebuild(contracts, codec_module, mocker, change):
    rebuild = mocker.patch.object(torch.multiprocessing.reductions, "rebuild_cuda_tensor")
    descriptor = msgspec.structs.replace(_descriptor(contracts, codec_module), **change)
    with pytest.raises(contracts.CompatibilityError):
        codec_module.TorchCudaIpcCodec().import_storage(descriptor, 0)
    rebuild.assert_not_called()


def test_cuda_uuid_is_normalized_for_nvml_and_mig_prefix_rejected(contracts, codec_module):
    bare_uuid = "12345678-1234-5678-9abc-123456789abc"
    assert codec_module._canonical_gpu_uuid(bare_uuid) == f"GPU-{bare_uuid}"
    assert codec_module._canonical_gpu_uuid(f"GPU-{bare_uuid}") == f"GPU-{bare_uuid}"
    with pytest.raises(contracts.UnsupportedError, match="physical"):
        codec_module._canonical_gpu_uuid(f"MIG-{bare_uuid}")


_GPU_UUID = "GPU-12345678-1234-5678-9abc-123456789abc"
_OTHER_GPU_UUID = "GPU-87654321-4321-8765-9abc-123456789abc"


@pytest.fixture
def device_identity(codec_module, monkeypatch, mocker):
    nvml = mocker.Mock()
    nvml.NVMLError = type("NVMLError", (Exception,), {})
    nvml.nvmlDeviceIsMigDeviceHandle.return_value = False
    properties = mocker.Mock(return_value=(_GPU_UUID.removeprefix("GPU-"),))
    for name, attribute, value in (
        ("import_utils", "import_pynvml", lambda: nvml),
        ("platform_utils", "cuda_get_device_properties", properties),
    ):
        module = ModuleType(f"vllm.utils.{name}")
        setattr(module, attribute, value)
        monkeypatch.setitem(codec_module.sys.modules, module.__name__, module)
    process = mocker.Mock()
    process.getpid.return_value = 101
    monkeypatch.setattr(codec_module, "os", process, raising=False)
    monkeypatch.setattr(codec_module, "_validated_gpu_uuids", set(), raising=False)
    return properties, nvml, process


def test_device_uuid_caches_success_per_process_and_physical_gpu(codec_module, device_identity, mocker):
    properties, nvml, process = device_identity
    for _ in range(50):
        assert codec_module._device_uuid(0) == _GPU_UUID
    assert properties.call_count == 50
    nvml.nvmlInit.assert_called_once_with()
    nvml.nvmlShutdown.assert_called_once_with()
    # Different local ordinals for the same physical UUID reuse its validation.
    assert codec_module._device_uuid(1) == _GPU_UUID
    properties.assert_called_with(1, ("uuid",), init_cuda=True)
    properties.return_value = (_OTHER_GPU_UUID,)
    assert codec_module._device_uuid(0) == _OTHER_GPU_UUID
    process.getpid.return_value = 102
    assert codec_module._device_uuid(0) == _OTHER_GPU_UUID
    assert nvml.nvmlDeviceGetHandleByUUID.call_args_list == [
        mocker.call(_GPU_UUID),
        mocker.call(_OTHER_GPU_UUID),
        mocker.call(_OTHER_GPU_UUID),
    ]
    assert nvml.nvmlInit.call_count == nvml.nvmlShutdown.call_count == 3


@pytest.mark.parametrize("failure", ["init", "query", "mig", "shutdown"])
def test_device_uuid_does_not_cache_failed_validation(contracts, codec_module, device_identity, failure):
    _, nvml, _ = device_identity
    operation = {
        "init": nvml.nvmlInit,
        "query": nvml.nvmlDeviceGetHandleByUUID,
        "mig": nvml.nvmlDeviceIsMigDeviceHandle,
        "shutdown": nvml.nvmlShutdown,
    }[failure]
    if failure == "mig":
        operation.return_value = True
    else:
        operation.side_effect = nvml.NVMLError("injected NVML failure")
    with pytest.raises((contracts.UnsupportedError, nvml.NVMLError)):
        codec_module._device_uuid(0)
    operation.side_effect = None
    nvml.nvmlDeviceIsMigDeviceHandle.return_value = False
    for _ in range(2):
        assert codec_module._device_uuid(0) == _GPU_UUID
    assert nvml.nvmlInit.call_count == 2
    assert nvml.nvmlShutdown.call_count == (1 if failure == "init" else 2)


def test_cached_device_uuid_preserves_runtime_checks(contracts, codec_module, device_identity, request, monkeypatch):
    properties, nvml, _ = device_identity
    real_device_uuid = codec_module._device_uuid
    snapshot, _ = request.getfixturevalue("cuda_runtime")
    monkeypatch.setattr(codec_module, "_device_uuid", real_device_uuid)
    context = make_context(contracts, device_uuid=_GPU_UUID)
    codec = codec_module.TorchCudaIpcCodec()
    codec.validate_runtime(context)
    properties.return_value = (_OTHER_GPU_UUID,)
    with pytest.raises(contracts.CompatibilityError, match="physical GPU UUID differs"):
        codec.validate_runtime(context)
    properties.return_value = (_GPU_UUID,)
    snapshot["allocator_settings"]["expandable_segments"] = True
    codec.validate_runtime(context)
    monkeypatch.setattr(torch._C, "_cuda_getAllocatorBackend", lambda: "cudaMallocAsync")
    with pytest.raises(contracts.UnsupportedError, match="only the native"):
        codec.validate_runtime(context)
    assert nvml.nvmlInit.call_count == nvml.nvmlShutdown.call_count == 2


def test_device_uuid_serializes_first_validation(codec_module, device_identity, monkeypatch):
    _, nvml, _ = device_identity
    entered, release, second_waiting = Event(), Event(), Event()

    class ObservedLock:
        def __init__(self):
            self.lock = Lock()
            self.calls = 0

        def __enter__(self):
            self.calls += 1
            if self.calls == 2:
                second_waiting.set()
            self.lock.acquire()

        def __exit__(self, *_exc):
            self.lock.release()

    def initialize():
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(codec_module, "_gpu_uuid_lock", ObservedLock(), raising=False)
    nvml.nvmlInit.side_effect = initialize
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(codec_module._device_uuid, 0)
        try:
            assert entered.wait(5)
            second = pool.submit(codec_module._device_uuid, 0)
            assert second_waiting.wait(5)
        finally:
            release.set()
        assert first.result(timeout=5) == second.result(timeout=5) == _GPU_UUID
    nvml.nvmlInit.assert_called_once_with()
    nvml.nvmlShutdown.assert_called_once_with()


def test_allocator_snapshot_fails_closed(contracts, codec_module):
    snapshots: tuple[object, ...] = (None, {}, {"segments": None}, {"segments": [None]})
    for snapshot in snapshots:
        with pytest.raises(contracts.UnsupportedError):
            codec_module._allocator_segments(snapshot)
    segment = {
        "device": 0,
        "segment_pool_id": (0, 0),
        "blocks": [{"address": 1024, "size": 256, "state": "active_allocated"}],
    }
    assert codec_module._allocator_segments({"segments": [segment]}) == [segment]
    assert codec_module._native_blocks([segment], 0) == {1024: 256}
    assert codec_module._native_blocks([segment], 1) == {}
    for change in (
        {"segment_pool_id": (0, 1)},
        {"segment_pool_id": None},
        {"blocks": [{"address": 1024, "size": 256, "state": "inactive"}]},
    ):
        assert codec_module._native_blocks([dict(segment, **change)], 0) == {}
