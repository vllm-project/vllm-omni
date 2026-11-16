# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real sibling-process Ascend IPC; parent relays metadata, never tensors."""

import gc
import importlib
import sys
import weakref
from dataclasses import replace

import msgspec
import pytest
import torch

from tests.distributed.weight_sharing.helpers import (
    load_weight_sharing,
    make_context,
    make_tensor_specs,
    publish_to_parent,
    receive,
    relay_import,
    relay_release,
    release_from_parent,
    spawn_workers,
)
from tests.helpers.mark import hardware_marks

if sys.platform != "linux":
    pytest.skip("NPU IPC requires Linux", allow_module_level=True)
torch_npu = pytest.importorskip("torch_npu")
pytestmark = [pytest.mark.core_model, *hardware_marks(res={"npu": "A2"}, num_cards=1)]


@pytest.fixture(autouse=True)
def native_allocator(monkeypatch):
    if not torch_npu.npu.is_available():
        pytest.skip("NPU IPC requires an Ascend NPU")
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")


def _runtime(session="session"):
    package = load_weight_sharing()
    api = importlib.import_module(f"{package.__name__}.contracts")
    codec_module = importlib.import_module(f"{package.__name__}.torch_npu_ipc_codec")
    context = make_context(api, device_uuid=codec_module._device_uuid(0), session_id=session)
    return api, package.TorchIpcBackend(codec=package.TorchNpuIpcCodec()), context


def _bindings(device):
    base = torch.arange(128, dtype=torch.float32, device=device).reshape(16, 8)
    first = base[:8]
    return {
        "first": first,
        "alias": first,
        "transpose": base[8:].T,
        "independent": first.clone(),
        "fp16": torch.full((8, 8), 0.5, dtype=torch.float16, device=device),
        "bf16": torch.full((8, 8), 0.25, dtype=torch.bfloat16, device=device),
        "scalar": torch.full((), 3.0, device=device),
        "empty": torch.empty(0, device=device),
    }


def _spec(api):
    return api.WeightSpec("test", "revision", "tp1", make_tensor_specs(api, _bindings("cpu"), {"alias": "first"}))


def _provider(rounds, control):
    baseline = None
    for index in range(rounds):
        api, backend, context = _runtime(f"round-{index}")
        bindings = _bindings("npu:0")
        exported = backend.prepare_export(bindings, _spec(api), context)
        assert backend.session_status(exported).storage_count == 6
        assert len(backend.session_status(exported).exported_storage_ids) == 5
        del bindings
        publish_to_parent(backend, exported, control)
        release_from_parent(backend, exported, control)
        gc.collect()
        torch_npu.npu.ipc_collect()
        current = torch_npu.npu.memory_allocated(0)
        if baseline is not None:
            assert current <= baseline
        baseline = current
        receive(control, "next")


def _consumer(rounds, control):
    baseline = None
    for index in range(rounds):
        api, backend, context = _runtime(f"round-{index}")
        payload = receive(control)
        imported = backend.open_import(payload, _spec(api), context)
        control.send(imported.receipt)
        receive(control, "attached")
        assert imported["first"] is imported["alias"]
        assert imported["first"].untyped_storage()._cdata == imported["transpose"].untyped_storage()._cdata
        assert imported["first"].untyped_storage()._cdata != imported["independent"].untyped_storage()._cdata
        for name, expected in _bindings("cpu").items():
            torch.testing.assert_close(imported[name].cpu(), expected, rtol=0, atol=0)
            assert imported[name].stride() == expected.stride()
            assert imported[name].storage_offset() == expected.storage_offset()
        # Also execute an NPU operator against the reconstructed float view.
        result = imported["first"] @ torch.ones(8, 1, device="npu:0")
        torch.testing.assert_close(result.cpu(), _bindings("cpu")["first"].sum(1, keepdim=True))
        del result
        with pytest.raises(api.WeightSharingError, match="reused"):
            backend.open_import(payload, _spec(api), context)
        external = imported["transpose"][:, :2]
        parameter = torch.nn.Parameter(imported["independent"], requires_grad=False)
        assert backend.close_import(imported).status == "HELD"
        del external
        assert backend.close_import(imported).status == "HELD"
        del parameter
        gc.collect()
        closed = backend.close_import(imported)
        assert closed.status == "CLOSED"
        assert backend.close_import(imported) == closed
        control.send(closed.receipt)
        current = torch_npu.npu.memory_allocated(0)
        if baseline is not None:
            assert current <= baseline
        baseline = current


@pytest.mark.parametrize("rounds", [1, 20])
def test_roundtrip_and_lifetime(rounds):
    with spawn_workers((_provider, (rounds,)), (_consumer, (rounds,))) as workers:
        (_, provider), (_, consumer) = workers
        for _ in range(rounds):
            relay_import(provider, consumer)
            relay_release(provider, consumer)
            provider.send("next")


def _large_spec(api, nbytes):
    return api.WeightSpec("large", "revision", "flat", (api.TensorSpec("weight", "float32", (nbytes // 4,)),))


def _large_provider(nbytes, control):
    api, backend, context = _runtime()
    weights = {"weight": torch.ones(nbytes // 4, device="npu:0")}
    exported = backend.prepare_export(weights, _large_spec(api, nbytes), context)
    publish_to_parent(backend, exported, control)
    release_from_parent(backend, exported, control)
    del weights
    gc.collect()
    torch_npu.npu.ipc_collect()


def _large_consumer(nbytes, control):
    api, backend, context = _runtime()
    warmup = torch.ones(1024, device="npu:0")
    torch_npu.npu.synchronize()
    del warmup
    torch_npu.npu.empty_cache()
    payload = receive(control)
    before = torch_npu.npu.memory_allocated(0)
    torch_npu.npu.reset_peak_memory_stats(0)
    imported = backend.open_import(payload, _large_spec(api, nbytes), context)
    peak = torch_npu.npu.max_memory_allocated(0) - before
    assert peak < nbytes // 4
    control.send(imported.receipt)
    receive(control, "attached")
    assert imported["weight"].sum().item() == nbytes // 4
    duplicate = imported["weight"].clone()
    torch_npu.npu.synchronize()
    copied = torch_npu.npu.memory_allocated(0) - before
    assert copied >= nbytes
    del duplicate
    result = backend.close_import(imported)
    assert result.status == "CLOSED"
    assert torch_npu.npu.memory_allocated(0) == before
    control.send(result.receipt)
    control.send((peak, copied))


def test_import_does_not_allocate_another_weight_copy(record_property):
    nbytes = 128 * 1024 * 1024
    with spawn_workers((_large_provider, (nbytes,)), (_large_consumer, (nbytes,))) as workers:
        (_, provider), (_, consumer) = workers
        relay_import(provider, consumer)
        relay_release(provider, consumer)
        peak, copied = receive(consumer)
        record_property("import_peak_bytes", peak)
        record_property("clone_control_bytes", copied)


def _cancel(control):
    api, backend, context = _runtime()
    weights = _bindings("npu:0")
    for i in range(20):
        context = replace(context, identity=msgspec.structs.replace(context.identity, session_id=f"cancel-{i}"))
        exported = backend.prepare_export(weights, _spec(api), context)
        assert backend.release_export(exported).status == "CLOSED"
        assert backend.release_export(exported).status == "CLOSED"
    del weights
    gc.collect()
    torch_npu.npu.ipc_collect()
    control.send("closed")


def test_cancel_unclaimed_export():
    with spawn_workers((_cancel, ())) as workers:
        receive(workers[0][1], "closed")


def _partial_import_spec(api):
    return api.WeightSpec(
        "failure-test", "revision", "final", tuple(api.TensorSpec(name, "float32", (32,)) for name in ("a", "b"))
    )


def _partial_import_provider(control):
    api, backend, context = _runtime()
    weights = {name: torch.arange(32, dtype=torch.float32, device="npu:0") for name in ("a", "b")}
    sources = [weakref.ref(tensor) for tensor in weights.values()]
    exported = backend.prepare_export(weights, _partial_import_spec(api), context)
    del weights
    control.send(backend.claim_export(exported))
    receive(control, "failed")
    assert backend.release_export(exported).status == "HELD"
    backend.mark_failed(exported)
    assert backend.release_export(exported).status == "POISONED"
    gc.collect()
    assert all(source() is not None for source in sources)
    control.send("checked")
    receive(control, "exit")  # Keep provider allocations alive until the consumer exits.


def _partial_import_consumer(control):
    api, backend, context = _runtime()
    payload = receive(control)
    original_import = backend._codec.import_storage
    rebuilt = []
    uncertain = []

    def import_storage(descriptor, device_index):
        carrier = original_import(descriptor, device_index)
        rebuilt.append(weakref.ref(carrier))
        if len(rebuilt) == 2:
            # The native rebuild consumed the counter before the injected error.
            # Retain its uncertain result until this process terminates.
            uncertain.append(carrier)
            raise RuntimeError("injected failure after NPU rebuild")
        return carrier

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend._codec, "import_storage", import_storage)
        with pytest.raises(api.WeightSharingError) as error:
            backend.open_import(payload, _partial_import_spec(api), context)
        assert error.value.poisoned
        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == "injected failure after NPU rebuild"
        del error  # Do not let the exception traceback keep the first mapping alive.
        gc.collect()
        snapshot = backend.session_status(context.identity.session_id)
        assert snapshot.state == api.SessionState.POISONED
        assert snapshot.imported_storage_ids == ("storage-0",)
        assert snapshot.active_storage_id == "storage-1"
        assert len(rebuilt) == 2 and len(uncertain) == 1
        record = backend._sessions[snapshot.session_id]
        assert record.result is None  # No import receipt was produced.
        assert record.carriers["storage-0"] is rebuilt[0]()
        assert len(record.leases) == 1 and not record.leases[0].expired()
        result = backend.close_import(snapshot.session_id)
        assert result.status == "POISONED" and result.receipt is None and not result.local_complete
        with pytest.raises(api.WeightSharingError, match="reused"):
            backend.open_import(payload, _partial_import_spec(api), context)
        assert len(rebuilt) == 2  # Re-import must not call the native rebuilder again.
        assert record.carriers["storage-0"] is rebuilt[0]()
        first = record.carriers["storage-0"].view(torch.float32)
        torch.testing.assert_close(first.cpu(), torch.arange(32, dtype=torch.float32))
        control.send("checked")
        receive(control, "exit")


def test_partial_import_failure_retains_real_npu_mapping():
    with spawn_workers((_partial_import_provider, ()), (_partial_import_consumer, ())) as workers:
        (_, provider), (consumer_process, consumer) = workers
        consumer.send(receive(provider))
        receive(consumer, "checked")
        provider.send("failed")
        receive(provider, "checked")
        consumer.send("exit")
        consumer_process.join(timeout=45)
        assert consumer_process.exitcode == 0
        provider.send("exit")


def _unsupported(kind, control):
    api, backend, context = _runtime()
    weight = torch.ones(16, 16, device="npu:0")
    if kind == "format":
        torch_npu.npu.config.allow_internal_format = True
        weight = torch_npu.npu_format_cast(weight, 29)
        assert str(torch_npu.get_npu_format(weight)) == "FRACTAL_NZ"
    bindings = {"weight": weight}
    spec = api.WeightSpec("test", "revision", "tp1", make_tensor_specs(api, bindings))
    with pytest.raises(api.WeightSharingError) as error:
        backend.prepare_export(bindings, spec, context)
    assert isinstance(error.value.__cause__, api.UnsupportedError)
    assert backend.session_status(context.identity.session_id).exported_storage_ids == ()
    control.send("rejected")


@pytest.mark.parametrize("kind", ["format", "expandable"])
def test_unsupported_allocation_rejected_before_export(kind):
    env = {"PYTORCH_NPU_ALLOC_CONF": f"expandable_segments:{kind == 'expandable'}"}
    with spawn_workers((_unsupported, (kind,), env)) as workers:
        receive(workers[0][1], "rejected")
