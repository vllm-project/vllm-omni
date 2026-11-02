# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CUDA IPC correctness, asynchronous use, and allocation lifetimes."""

import gc
import threading
from dataclasses import replace
from pathlib import Path

import msgspec
import pytest
import torch

from tests.distributed.weight_sharing.cuda_helpers import (
    assert_export_allocations,
    publish_to_parent,
    receive,
    relay_import,
    relay_release,
    release_from_parent,
    require_cuda,
    runtime,
    spawn_workers,
)
from tests.distributed.weight_sharing.helpers import make_tensor_specs
from tests.helpers.mark import hardware_marks

require_cuda()
pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.fixture(autouse=True)
def supported_runtime(ipc_allocator):
    require_cuda(1)


def _bindings(device):
    base = torch.arange(256 * 256, dtype=torch.float32, device=device).remainder_(17).div_(16).reshape(256, 256)
    first = base[:128]
    return {
        "first": first,
        "alias": first,
        "transposed": base[128:].T,
        "independent": first.clone(),
        "fp16": torch.full((32, 64), 0.125, dtype=torch.float16, device=device),
        "bf16": torch.full((32, 64), 0.25, dtype=torch.bfloat16, device=device),
        "scalar": torch.full((), 3.0, device=device),
        "empty": torch.empty(0, device=device),
    }


def _spec(api):
    return api.WeightSpec(
        "small-test-model",
        "revision",
        "tp1-final",
        make_tensor_specs(api, _bindings("cpu"), {"alias": "first"}),
    )


def _iteration(context, index):
    return replace(context, identity=msgspec.structs.replace(context.identity, session_id=f"round-{index}"))


def _assert_released(baseline):
    gc.collect()
    current = (torch.accelerator.memory.memory_allocated(0), len(list(Path("/proc/self/fd").iterdir())))
    if baseline is not None:
        assert all(value <= limit for value, limit in zip(current, baseline)), (current, baseline)
    assert not any(thread.name == "weight-sharing-session" for thread in threading.enumerate())
    # cuBLAS/IPC may retain fixed caches after first use; subsequent rounds must plateau.
    return current if baseline is None else baseline


def _provider(rounds, control):
    api, backend, context = runtime()
    baseline = None
    stream = torch.Stream(device="cuda:0")
    spec = _spec(api)
    with stream:
        warmup = _bindings("cuda:0")
    stream.synchronize()
    del warmup
    for index in range(rounds):
        with stream:
            if index == 0:
                torch.cuda._sleep(500_000_000)
            bindings = _bindings("cuda:0")
        if index == 0:
            assert not stream.query(), "export synchronization test needs pending writes"
        exported = backend.prepare_export(bindings, spec, _iteration(context, index))
        assert stream.query()
        assert_export_allocations(bindings)
        del bindings
        snapshot = backend.session_status(exported)
        assert snapshot.storage_count == 6
        assert len(snapshot.exported_storage_ids) == 5
        assert snapshot.active_storage_id is None
        payload = backend.claim_export(exported)
        descriptors = msgspec.msgpack.decode(payload, type=api.ExportBundle).exports
        # Native allocator suballocations can share one handle, but not offsets.
        assert len({d.handle for d in descriptors}) < len(descriptors)
        assert len({(d.handle, d.allocation_offset_bytes) for d in descriptors}) == len(descriptors)
        control.send(payload)
        backend.confirm_import(receive(control))
        control.send("attached")
        backend.confirm_release(receive(control))
        assert backend.release_export(exported).status == "CLOSED"
        assert backend.release_export(exported).status == "CLOSED"
        assert backend.session_status(exported).state == api.SessionState.CLOSED
        baseline = _assert_released(baseline)
        control.send("closed")
        receive(control, "next")


def _check_views(imported):
    assert imported["first"] is imported["alias"]
    assert imported["first"].untyped_storage()._cdata == imported["transposed"].untyped_storage()._cdata
    assert imported["first"].untyped_storage()._cdata != imported["independent"].untyped_storage()._cdata
    assert imported["transposed"].storage_offset() == 128 * 256
    for name, expected in _bindings("cpu").items():
        actual = imported[name]
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _matmuls(bindings):
    return {
        name: bindings[name] @ torch.ones((bindings[name].shape[1], 4), dtype=bindings[name].dtype, device="cuda:0")
        for name in ("first", "fp16", "bf16")
    }


def _consumer(rounds, control):
    api, backend, context = runtime()
    baseline = None
    stream = torch.Stream(device="cuda:0")
    spec = _spec(api)
    # Initialize cuBLAS before the pending-read check, so CPU library startup
    # cannot consume the delay intended to keep the GPU stream busy.
    with stream:
        warmup = _matmuls(_bindings("cuda:0"))
    stream.synchronize()
    del warmup
    for index in range(rounds):
        payload = receive(control)
        iteration = _iteration(context, index)
        imported = backend.open_import(payload, spec, iteration)
        control.send(imported.receipt)
        receive(control, "attached")
        _check_views(imported)
        with pytest.raises(api.WeightSharingError):
            backend.open_import(payload, spec, iteration)
        external_view = imported["transposed"][:, :8]
        external_parameter = torch.nn.Parameter(imported["independent"], requires_grad=False)
        with stream:
            if index == 0:
                torch.cuda._sleep(500_000_000)
            outputs = _matmuls(imported)
        if index == 0:
            assert not stream.query(), "release synchronization test needs pending reads"
        # close_import must synchronize outstanding reads, even when returning HELD.
        assert backend.close_import(imported).status == "HELD"
        assert stream.query()
        del external_view
        assert backend.close_import(imported).status == "HELD"
        del external_parameter
        gc.collect()
        result = backend.close_import(imported)
        assert result.status == "CLOSED" and result.local_complete and not result.peer_confirmed
        assert backend.close_import(imported) == result
        assert backend.session_status(imported).imported_storage_ids == ()
        for name, output in outputs.items():
            expected = _bindings("cpu")[name].float().sum(dim=1, keepdim=True).expand(-1, 4)
            torch.testing.assert_close(output.float().cpu(), expected, rtol=0, atol=0)
        del output, outputs
        baseline = _assert_released(baseline)
        control.send(result.receipt)


@pytest.mark.parametrize("rounds", [1, 50], ids=["roundtrip", "repeated-lifecycle"])
def test_two_spawned_processes_share_storage_and_return_references(rounds):
    with spawn_workers((_provider, (rounds,)), (_consumer, (rounds,))) as workers:
        (_, provider), (_, consumer) = workers
        for _ in range(rounds):
            relay_import(provider, consumer)
            relay_release(provider, consumer)
            provider.send("next")


def _large_spec(api, nbytes):
    return api.WeightSpec("large-test", "revision", "flat", (api.TensorSpec("weight", "float32", (nbytes // 4,)),))


def _large_provider(nbytes, control):
    api, backend, context = runtime()
    weights = {"weight": torch.ones(nbytes // 4, device="cuda:0")}
    assert_export_allocations(weights)
    exported = backend.prepare_export(weights, _large_spec(api, nbytes), context)
    publish_to_parent(backend, exported, control)
    release_from_parent(backend, exported, control)


def _memory_consumer(nbytes, control):
    api, backend, context = runtime()
    # Warm the context and allocator before measuring. Run on an otherwise idle GPU.
    warmup = torch.ones(1024, device="cuda:0")
    assert_export_allocations({"warmup": warmup})
    torch.accelerator.synchronize(0)
    del warmup
    torch.accelerator.memory.empty_cache()
    payload = receive(control)
    memory = torch.accelerator.memory
    before = memory.memory_allocated(0)
    free_before, _ = memory.get_memory_info(0)
    memory.reset_peak_memory_stats(0)
    imported = backend.open_import(payload, _large_spec(api, nbytes), context)
    control.send(imported.receipt)
    receive(control, "attached")
    free_after, _ = memory.get_memory_info(0)
    peak = memory.max_memory_allocated(0) - before
    # Allow fixed mapping/context overhead, but not a second weight-sized allocation.
    assert peak < nbytes // 4
    assert free_before - free_after < nbytes // 4
    assert imported["weight"].sum().item() == nbytes // 4
    duplicate = imported["weight"].clone()
    torch.accelerator.synchronize(0)
    copied = memory.memory_allocated(0) - before
    assert copied >= nbytes  # Positive control: this measurement detects a real copy.
    del duplicate
    result = backend.close_import(imported)
    assert result.status == "CLOSED"
    control.send(result.receipt)
    assert memory.memory_allocated(0) == before
    control.send((peak, free_before - free_after, copied))


@pytest.mark.parametrize("nbytes", [128 * 1024 * 1024, 256 * 1024 * 1024], ids=["128MiB", "256MiB"])
@pytest.mark.parametrize("same_allocator", [True, False], ids=["matching", "mixed"])
def test_import_does_not_allocate_a_second_weight_copy(record_property, nbytes, same_allocator, ipc_allocator):
    consumer_expandable = ipc_allocator if same_allocator else not ipc_allocator
    consumer_env = {"PYTORCH_ALLOC_CONF": f"backend:native,expandable_segments:{consumer_expandable}"}
    with spawn_workers((_large_provider, (nbytes,)), (_memory_consumer, (nbytes,), consumer_env)) as workers:
        (_, provider), (consumer_process, consumer) = workers
        relay_import(provider, consumer)
        relay_release(provider, consumer)
        peak, driver_delta, copied = receive(consumer)
        record_property("shared_weight_bytes", nbytes)
        record_property("provider_expandable", ipc_allocator)
        record_property("consumer_expandable", consumer_expandable)
        record_property("import_peak_bytes", peak)
        record_property("import_device_delta_bytes", driver_delta)
        record_property("copy_control_bytes", copied)
        consumer_process.join(timeout=45)
        assert consumer_process.exitcode == 0


def _unexposed_provider(rounds, control):
    api, backend, context = runtime()
    weights = {"weight": torch.arange(16, dtype=torch.float32, device="cuda:0")}
    assert_export_allocations(weights)
    spec = api.WeightSpec("m", "r", "l", (api.TensorSpec("weight", "float32", (16,), (1,)),))
    for index in range(rounds):
        exported = backend.prepare_export(weights, spec, _iteration(context, index))
        assert backend.release_export(exported).status == "CLOSED"
        assert backend.release_export(exported).status == "CLOSED"
        assert backend.session_status(exported).state == api.SessionState.CLOSED
    torch.testing.assert_close(weights["weight"].cpu(), torch.arange(16, dtype=torch.float32))
    control.send("closed")


@pytest.mark.parametrize("rounds", [1, 50], ids=["single", "repeated-lifecycle"])
def test_unexposed_export_is_cancelled_once(rounds):
    with spawn_workers((_unexposed_provider, (rounds,))) as workers:
        receive(workers[0][1], "closed")
