# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import replace
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from vllm_omni.distributed import ipc_weight_sharing
from vllm_omni.distributed.ipc_weight_sharing import (
    ManifestMismatchError,
    ManifestValidationError,
    ProviderUnavailableError,
    TensorHandle,
    WeightManifest,
    WeightSharingConsumer,
    WeightSharingEndpoint,
    WeightSharingProvider,
    build_weight_manifest,
    compare_manifests,
    get_cuda_device_uuid,
    remap_cuda_handle_device,
)

pytestmark = [pytest.mark.core_model, pytest.mark.parallel, pytest.mark.cpu]


class _FakeStorage:
    def __init__(self, cdata: int, nbytes: int) -> None:
        self._cdata = cdata
        self._nbytes = nbytes

    def nbytes(self) -> int:
        return self._nbytes


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: str = "torch.float32",
        storage: _FakeStorage | None = None,
        stride: tuple[int, ...] | None = None,
        storage_offset: int = 0,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self._storage = storage or _FakeStorage(id(self), 4 * max(1, _product(shape)))
        self._stride = stride or _contiguous_stride(shape)
        self._storage_offset = storage_offset

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def storage_offset(self) -> int:
        return self._storage_offset

    def untyped_storage(self) -> _FakeStorage:
        return self._storage


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dimension in reversed(shape):
        stride.append(running)
        running *= dimension
    return tuple(reversed(stride))


def _manifest(
    *,
    tensors: dict[str, _FakeTensor] | None = None,
    device_index: int = 0,
    device_uuid: str = "GPU-test-0",
    parallel_rank: int = 0,
    parallel_size: int = 1,
) -> WeightManifest:
    tensors = tensors or {"layer.weight": _FakeTensor((2, 2))}
    return build_weight_manifest(
        tensors,
        model_id="test-model",
        checkpoint_revision="revision-a",
        component="decoder",
        device_index=device_index,
        device_uuid=device_uuid,
        parallel_rank=parallel_rank,
        parallel_size=parallel_size,
    )


def _replace_manifest_identity(manifest: WeightManifest, field_name: str, replacement: object) -> WeightManifest:
    if field_name not in {"device_index", "device_uuid", "parallel_rank"}:
        return replace(manifest, **{field_name: replacement})
    entries = tuple(replace(entry, **{field_name: replacement}) for entry in manifest.tensors)
    return replace(manifest, **{field_name: replacement, "tensors": entries})


def _provider_tensors(manifest: WeightManifest) -> dict[str, object]:
    return {name: object() for name in manifest.tensor_names}


def _constant_handle_exporter(
    tensors: Mapping[str, object],
    manifest: WeightManifest,
) -> dict[str, TensorHandle]:
    assert set(tensors) == set(manifest.tensor_names)
    return {"layer.weight": ("cuda-handle",)}


def _fake_handle_mapper(handle: TensorHandle, _device_index: int) -> TensorHandle:
    return handle


def test_manifest_round_trip_has_stable_fingerprint_and_storage_aliases() -> None:
    storage = _FakeStorage(cdata=17, nbytes=64)
    manifest = _manifest(
        tensors={
            "weight": _FakeTensor((2, 2), storage=storage),
            "weight_view": _FakeTensor((2,), storage=storage, stride=(1,), storage_offset=1),
        }
    )

    restored = WeightManifest.from_payload(manifest.to_payload())

    assert restored == manifest
    assert restored.fingerprint == manifest.fingerprint
    assert restored.tensors[0].storage_key == restored.tensors[1].storage_key


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("protocol_version", 2),
        ("model_id", "other-model"),
        ("checkpoint_revision", "revision-b"),
        ("device_uuid", "GPU-test-1"),
        ("parallel_rank", 1),
        ("parallel_size", 2),
    ],
)
def test_manifest_mismatch_rejects_identity_changes(field_name: str, replacement: object) -> None:
    manifest = _manifest(parallel_size=2 if field_name == "parallel_rank" else 1)
    actual = _replace_manifest_identity(manifest, field_name, replacement)

    mismatches = compare_manifests(manifest, actual)

    assert mismatches
    assert field_name in " ".join(mismatches)


def test_manifest_accepts_different_local_device_ordinal_for_same_gpu() -> None:
    provider_manifest = _manifest(device_index=0, device_uuid="GPU-physical-0")
    consumer_manifest = _manifest(device_index=1, device_uuid="GPU-physical-0")

    assert compare_manifests(consumer_manifest, provider_manifest) == ()


def test_remap_cuda_handle_device_uses_consumer_local_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDevice:
        def __init__(self, specification: str) -> None:
            self.type, _, index = specification.partition(":")
            self.index = int(index) if index else None

    class _TorchModule:
        device = _FakeDevice

    monkeypatch.setitem(sys.modules, "torch", _TorchModule())
    remapped = remap_cuda_handle_device(("tensor", _FakeDevice("cuda:0"), "handle"), 1)

    assert remapped[1].type == "cuda"
    assert remapped[1].index == 1


def test_provider_rejects_tensor_on_a_different_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest(device_index=0, device_uuid="GPU-physical-0")
    tensors = {"layer.weight": SimpleNamespace(is_cuda=True, device=SimpleNamespace(index=1))}
    monkeypatch.setattr(ipc_weight_sharing, "get_cuda_device_uuid", lambda _index: "GPU-physical-1")

    with pytest.raises(ManifestMismatchError, match="device_index"):
        ipc_weight_sharing._validate_provider_tensor_devices(tensors, manifest)


def test_manifest_mismatch_rejects_tensor_shape_dtype_and_parallel_metadata() -> None:
    expected = _manifest(parallel_size=2)
    entry = expected.tensors[0]
    actual = replace(
        expected,
        tensors=(
            replace(
                entry,
                shape=(4, 1),
                dtype="torch.float16",
            ),
        ),
    )

    mismatches = compare_manifests(expected, actual)

    assert any("shape" in mismatch for mismatch in mismatches)
    assert any("dtype" in mismatch for mismatch in mismatches)


def test_manifest_rejects_invalid_parallel_rank() -> None:
    with pytest.raises(ManifestValidationError, match="parallel rank"):
        _manifest(parallel_rank=2, parallel_size=2)


def test_endpoint_rejects_non_loopback_tcp_addresses() -> None:
    with pytest.raises(ValueError, match="loopback"):
        WeightSharingEndpoint.tcp(12345, b"test", host="10.0.0.1")


def test_cuda_device_uuid_accepts_pytorch_uuid_object(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CudaUuid:
        def __str__(self) -> str:
            return "c15ebbe9-0a83-b17e-20e8-9eb270c37248"

    class _CudaProperties:
        uuid = _CudaUuid()

    class _CudaModule:
        @staticmethod
        def get_device_properties(device_index: int) -> _CudaProperties:
            assert device_index == 0
            return _CudaProperties()

    class _TorchModule:
        cuda = _CudaModule()

    monkeypatch.setitem(sys.modules, "torch", _TorchModule())

    assert get_cuda_device_uuid(0) == "c15ebbe9-0a83-b17e-20e8-9eb270c37248"


def test_provider_consumer_handshake_and_heartbeat() -> None:
    manifest = _manifest()
    provider = WeightSharingProvider(
        WeightSharingEndpoint.tcp(0, b"unit-test-authkey"),
        manifest,
        _provider_tensors(manifest),
        handle_exporter=_constant_handle_exporter,
        heartbeat_interval=0.05,
    )
    provider.start()

    mapped = WeightSharingConsumer.connect(
        provider.endpoint,
        manifest,
        rebuild=lambda handle: f"mapped:{handle[0]}",
        handle_mapper=_fake_handle_mapper,
        validate_local_device=False,
        validate_mapped_tensors=False,
        heartbeat_interval=0.05,
    )
    try:
        assert mapped["layer.weight"] == "mapped:cuda-handle"
        mapped.check_liveness(timeout=0.5)
        assert mapped.provider_alive
    finally:
        mapped.close()
        provider.stop()


def test_provider_rejects_manifest_before_sending_handles() -> None:
    manifest = _manifest()
    mismatched = _replace_manifest_identity(manifest, "device_uuid", "GPU-other")

    def unexpected_export(
        _tensors: Mapping[str, object],
        _manifest: WeightManifest,
    ) -> Mapping[str, TensorHandle]:
        raise AssertionError("handles must not be exported before manifest validation")

    provider = WeightSharingProvider(
        WeightSharingEndpoint.tcp(0, b"unit-test-authkey"),
        manifest,
        _provider_tensors(manifest),
        handle_exporter=unexpected_export,
        heartbeat_interval=0.05,
    )
    provider.start()

    try:
        with pytest.raises(ManifestMismatchError, match="device_uuid"):
            WeightSharingConsumer.connect(
                provider.endpoint,
                mismatched,
                handle_mapper=_fake_handle_mapper,
                validate_local_device=False,
                validate_mapped_tensors=False,
                heartbeat_interval=0.05,
            )
    finally:
        provider.stop()


def test_provider_stop_unblocks_listener_thread() -> None:
    manifest = _manifest()
    provider = WeightSharingProvider(
        WeightSharingEndpoint.tcp(0, b"unit-test-authkey"),
        manifest,
        _provider_tensors(manifest),
        handle_exporter=_constant_handle_exporter,
        heartbeat_interval=0.05,
    )
    serve_thread = provider.start()

    provider.stop()

    assert not serve_thread.is_alive()


def test_provider_stop_rejects_connection_accepted_during_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest()
    provider = WeightSharingProvider(
        WeightSharingEndpoint.tcp(0, b"unit-test-authkey"),
        manifest,
        _provider_tensors(manifest),
        handle_exporter=_constant_handle_exporter,
        heartbeat_interval=0.05,
    )
    original_listener = provider._listener
    accept_started = Event()
    release_accept = Event()
    connection_closed = Event()
    serve_connection_called = Event()

    class _Connection:
        def close(self) -> None:
            connection_closed.set()

    connection = _Connection()

    class _Listener:
        def accept(self) -> _Connection:
            accept_started.set()
            assert release_accept.wait(timeout=1.0)
            return connection

        def close(self) -> None:
            pass

    original_listener.close()
    monkeypatch.setattr(provider, "_listener", _Listener())
    monkeypatch.setattr(provider, "_wake_listener", lambda: None)
    monkeypatch.setattr(provider, "_serve_connection", lambda _connection: serve_connection_called.set())

    serve_thread = provider.start()
    stop_thread: Thread | None = None
    try:
        assert accept_started.wait(timeout=1.0)
        stop_thread = Thread(target=provider.stop)
        stop_thread.start()
        assert provider._stop_event.wait(timeout=1.0)
        release_accept.set()
        stop_thread.join(timeout=1.0)

        assert not stop_thread.is_alive()
        assert not serve_thread.is_alive()
        assert connection_closed.is_set()
        assert not serve_connection_called.wait(timeout=0.1)
    finally:
        release_accept.set()
        provider.stop()
        if stop_thread is not None:
            stop_thread.join(timeout=1.0)
        serve_thread.join(timeout=1.0)


def test_consumer_close_does_not_report_provider_exit() -> None:
    manifest = _manifest()
    provider = WeightSharingProvider(
        WeightSharingEndpoint.tcp(0, b"unit-test-authkey"),
        manifest,
        _provider_tensors(manifest),
        handle_exporter=_constant_handle_exporter,
        heartbeat_interval=0.05,
    )
    provider.start()
    exited = Event()
    mapped = WeightSharingConsumer.connect(
        provider.endpoint,
        manifest,
        rebuild=lambda handle: handle,
        handle_mapper=_fake_handle_mapper,
        validate_local_device=False,
        validate_mapped_tensors=False,
        heartbeat_interval=0.05,
        on_provider_exit=lambda _error: exited.set(),
    )

    mapped.close()
    try:
        assert not exited.wait(timeout=0.1)
    finally:
        provider.stop()


def test_provider_exports_fresh_handles_for_each_consumer() -> None:
    manifest = _manifest()
    export_count = 0

    def export_handles(
        tensors: Mapping[str, object],
        current_manifest: WeightManifest,
    ) -> dict[str, TensorHandle]:
        nonlocal export_count
        assert set(tensors) == set(current_manifest.tensor_names)
        export_count += 1
        return {"layer.weight": (f"cuda-handle-{export_count}",)}

    provider = WeightSharingProvider(
        WeightSharingEndpoint.tcp(0, b"unit-test-authkey"),
        manifest,
        _provider_tensors(manifest),
        handle_exporter=export_handles,
        heartbeat_interval=0.05,
    )
    provider.start()
    mapped_weights = []
    try:
        for _ in range(2):
            mapped_weights.append(
                WeightSharingConsumer.connect(
                    provider.endpoint,
                    manifest,
                    rebuild=lambda handle: handle[0],
                    handle_mapper=_fake_handle_mapper,
                    validate_local_device=False,
                    validate_mapped_tensors=False,
                    heartbeat_interval=0.05,
                )
            )

        assert [mapped["layer.weight"] for mapped in mapped_weights] == [
            "cuda-handle-1",
            "cuda-handle-2",
        ]
        assert export_count == 2
    finally:
        for mapped in mapped_weights:
            mapped.close()
        provider.stop()


def test_consumer_reports_provider_exit() -> None:
    manifest = _manifest()
    provider = WeightSharingProvider(
        WeightSharingEndpoint.tcp(0, b"unit-test-authkey"),
        manifest,
        _provider_tensors(manifest),
        handle_exporter=_constant_handle_exporter,
        heartbeat_interval=0.05,
    )
    provider.start()
    exited = Event()

    mapped = WeightSharingConsumer.connect(
        provider.endpoint,
        manifest,
        rebuild=lambda handle: handle,
        handle_mapper=_fake_handle_mapper,
        validate_local_device=False,
        validate_mapped_tensors=False,
        heartbeat_interval=0.05,
        on_provider_exit=lambda _error: exited.set(),
    )
    provider.stop()
    try:
        deadline = monotonic() + 2.0
        while not exited.is_set() and monotonic() < deadline:
            sleep(0.02)
        assert exited.is_set()
        with pytest.raises(ProviderUnavailableError):
            mapped["layer.weight"]
    finally:
        mapped.close()
