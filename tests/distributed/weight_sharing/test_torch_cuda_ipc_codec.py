# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CPU storage/view tests and fixed CUDA ABI tests without a CUDA runtime."""

import gc
import importlib

import msgspec
import pytest
import torch

from tests.distributed.weight_sharing.helpers import make_context

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def api(weight_sharing):
    return importlib.import_module(f"{weight_sharing.__name__}.contracts")


@pytest.fixture
def codec_module(api):
    return importlib.import_module(f"{api.__package__}.torch_cuda_ipc_codec")


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


def _spec(api, bindings, aliases=None):
    aliases = aliases or {}
    return api.WeightSpec(
        "test-model",
        "checkpoint-1",
        "final-layout",
        tuple(
            api.TensorSpec(
                name,
                str(tensor.dtype).removeprefix("torch."),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                aliases.get(name),
            )
            for name, tensor in bindings.items()
        ),
    )


def _descriptor(api, codec_module):
    return api.StorageExport(
        "storage-0",
        codec_module._check_torch_version(),
        3,
        b"handle",
        64,
        256,
        b"counter",
        7,
        b"event",
        False,
    )


def test_storage_dedup_alias_mixed_dtype_and_noncontiguous_views_without_copy(api, codec_module):
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
    spec = _spec(api, bindings, {"alias": "weight"})
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


def test_distinct_storages_are_not_merged(api, codec_module):
    bindings = {"a": torch.ones(4), "b": torch.ones(4)}
    prepared = codec_module._describe_bindings(bindings, _spec(api, bindings))
    assert len(prepared.carriers) == 2


@pytest.mark.parametrize("kind", ["meta", "conjugate", "negative", "sparse", "subclass", "named"])
def test_unsupported_tensor_forms_fail_without_copy(api, codec_module, kind):
    if kind == "meta":
        tensor = torch.empty(4, device="meta")
    elif kind == "conjugate":
        tensor = torch.ones(4, dtype=torch.complex64).conj()
    elif kind == "negative":
        tensor = torch._neg_view(torch.ones(4))
    elif kind == "sparse":
        tensor = torch.ones(4).to_sparse()
    elif kind == "named":
        tensor = torch.ones(4, names=("feature",))
    else:

        class TensorSubclass(torch.Tensor):
            pass

        tensor = torch.ones(4).as_subclass(TensorSubclass)
    spec = api.WeightSpec("m", "r", "l", (api.TensorSpec("weight", str(tensor.dtype).removeprefix("torch."), (4,)),))
    with pytest.raises(api.UnsupportedError):
        codec_module._describe_bindings({"weight": tensor}, spec)


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


def test_integer_device_reduction_schema(api, codec_module):
    args = (
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
        b"event",
        False,
    )
    descriptor = codec_module._encode_reduction(args, "storage-0", 64)
    assert descriptor == _descriptor(api, codec_module)
    incompatible = args[:6] + (torch.device("cuda", 3),) + args[7:]
    with pytest.raises(api.CompatibilityError, match="device or offset"):
        codec_module._encode_reduction(incompatible, "storage-0", 64)


def test_import_remaps_to_local_device_and_uses_fixed_rebuild(api, codec_module, mocker):
    mocker.patch.object(codec_module, "_check_torch_version", return_value="torch-2.10-native-v1")
    rebuild = mocker.patch.object(torch.multiprocessing.reductions, "rebuild_cuda_tensor")
    codec = codec_module.TorchCudaIpcCodec()
    assert codec.import_storage(_descriptor(api, codec_module), 0) is rebuild.return_value
    rebuild.assert_called_once()
    args = rebuild.call_args.args
    assert args[0] is torch.Tensor
    assert args[4] is torch.storage.TypedStorage
    assert args[6] == 0  # provider used ordinal 3
    assert args[9] == 256
    assert args[12] == 7


def test_rebuild_error_is_propagated_without_counter_cancellation(api, codec_module, mocker):
    mocker.patch.object(codec_module, "_check_torch_version", return_value="torch-2.10-native-v1")
    mocker.patch.object(
        torch.multiprocessing.reductions, "rebuild_cuda_tensor", side_effect=RuntimeError("possibly consumed")
    )
    release = mocker.patch.object(torch.storage.TypedStorage, "_release_ipc_counter")
    with pytest.raises(RuntimeError, match="possibly consumed"):
        codec_module.TorchCudaIpcCodec().import_storage(_descriptor(api, codec_module), 0)
    release.assert_not_called()


def test_cancel_uses_only_the_supplied_unconsumed_counter(api, codec_module, mocker):
    mocker.patch.object(codec_module, "_check_torch_version", return_value="torch-2.10-native-v1")
    release = mocker.patch.object(torch.storage.TypedStorage, "_release_ipc_counter")
    codec_module.TorchCudaIpcCodec().cancel_unconsumed(_descriptor(api, codec_module), 0)
    release.assert_called_once_with(b"counter", 7, device=0)


@pytest.mark.parametrize("version", ["2.9.1", "2.11.0", "2.12.0", "2.14.0", "2.10.0.dev1", "2.13.0rc1"])
def test_unknown_abi_rejected(api, codec_module, monkeypatch, version):
    monkeypatch.setattr(torch, "__version__", version)
    with pytest.raises(api.UnsupportedError, match="2.10"):
        codec_module._check_torch_version()


@pytest.mark.parametrize("version", ["2.10.0+cpu", "2.10.1+cu130", "2.13.0+cu130"])
def test_supported_release_abi(codec_module, monkeypatch, version):
    monkeypatch.setattr(torch, "__version__", version)
    codec_module._check_torch_version()


def test_consumer_checks_runtime_once_and_sync_does_not_rescan(api, codec_module, cuda_runtime):
    _, calls = cuda_runtime
    context = make_context(api)
    codec = codec_module.TorchCudaIpcCodec()
    codec.validate_runtime(context)
    codec.synchronize(context)
    codec.synchronize(context)
    assert calls == [("select", 0), ("snapshot", None), ("sync", 0), ("sync", 0)]


@pytest.mark.parametrize("invalid", ["gpu", "allocator", "expandable"])
def test_consumer_runtime_rejects_unsafe_environment(api, codec_module, cuda_runtime, monkeypatch, invalid):
    snapshot, _ = cuda_runtime
    if invalid == "gpu":
        monkeypatch.setattr(codec_module, "_device_uuid", lambda index: "GPU-other")
    elif invalid == "allocator":
        monkeypatch.setattr(torch._C, "_cuda_getAllocatorBackend", lambda: "cudaMallocAsync")
    else:
        snapshot["allocator_settings"]["expandable_segments"] = True
    with pytest.raises((api.CompatibilityError, api.UnsupportedError)):
        codec_module.TorchCudaIpcCodec().validate_runtime(make_context(api))


def test_cross_minor_codec_is_rejected_before_rebuild(api, codec_module, monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.10.0")
    descriptor = _descriptor(api, codec_module)
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(api.UnsupportedError, match="codec"):
        codec_module._validate_descriptor(descriptor)


@pytest.mark.parametrize(
    "change",
    [
        {"storage_id": ""},
        {"device_index": -1},
        {"storage_nbytes": 0},
        {"allocation_offset_bytes": -1},
        {"ref_counter_offset": -1},
        {"handle": b""},
        {"ref_counter_handle": b""},
        {"event_handle": b""},
        {"event_sync_required": 1},
    ],
)
def test_invalid_descriptor_rejected_before_rebuild(api, codec_module, mocker, change):
    rebuild = mocker.patch.object(torch.multiprocessing.reductions, "rebuild_cuda_tensor")
    descriptor = msgspec.structs.replace(_descriptor(api, codec_module), **change)
    with pytest.raises(api.CompatibilityError):
        codec_module.TorchCudaIpcCodec().import_storage(descriptor, 0)
    rebuild.assert_not_called()


def test_cuda_uuid_is_normalized_for_nvml_and_mig_prefix_rejected(api, codec_module):
    bare_uuid = "12345678-1234-5678-9abc-123456789abc"
    assert codec_module._canonical_gpu_uuid(bare_uuid) == f"GPU-{bare_uuid}"
    assert codec_module._canonical_gpu_uuid(f"GPU-{bare_uuid}") == f"GPU-{bare_uuid}"
    with pytest.raises(api.UnsupportedError, match="physical"):
        codec_module._canonical_gpu_uuid(f"MIG-{bare_uuid}")


def test_allocator_snapshot_fails_closed(api, codec_module):
    snapshots: tuple[dict[str, object], ...] = (
        {},
        {"allocator_settings": {}},
        {"allocator_settings": {"expandable_segments": True}},
    )
    for snapshot in snapshots:
        with pytest.raises(api.UnsupportedError):
            codec_module._allocator_segments(snapshot)
    segment = {
        "device": 0,
        "is_expandable": False,
        "segment_pool_id": (0, 0),
        "blocks": [{"address": 1024, "size": 256, "state": "active_allocated"}],
    }
    assert codec_module._native_blocks([segment], 0) == {1024: 256}
    assert codec_module._native_blocks([segment], 1) == {}
    for change in ({"is_expandable": True}, {"segment_pool_id": (0, 1)}, {"segment_pool_id": None}):
        assert codec_module._native_blocks([dict(segment, **change)], 0) == {}
