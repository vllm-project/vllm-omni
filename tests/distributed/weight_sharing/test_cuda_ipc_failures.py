# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CUDA lifetimes with explicit caller-reported control-plane failures."""

import os
import weakref

import pytest
import torch

from tests.distributed.weight_sharing.cuda_helpers import (
    assert_export_allocations,
    receive,
    require_cuda,
    runtime,
    spawn_workers,
)
from tests.helpers.mark import hardware_marks

require_cuda()
pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.fixture(autouse=True)
def supported_runtime(ipc_allocator):
    require_cuda(1)


def _spec(api):
    return api.WeightSpec(
        "failure-test", "revision", "final", tuple(api.TensorSpec(name, "float32", (32,)) for name in ("a", "b"))
    )


def _provider(fault, control):
    api, backend, context = runtime()
    cancellations = []

    def cancel(*args):
        cancellations.append(args)
        raise AssertionError("an uncertain CUDA reference must not be cancelled")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend._codec, "cancel_unconsumed", cancel)
        weights = {name: torch.arange(32, dtype=torch.float32, device="cuda:0") for name in ("a", "b")}
        assert_export_allocations(weights)
        sources = [weakref.ref(value) for value in weights.values()]
        exported = backend.prepare_export(weights, _spec(api), context)
        del weights
        control.send(backend.claim_export(exported))
        if fault == "provider_exit":
            os._exit(0)
        outcome = receive(control)
        if isinstance(outcome, bytes):
            backend.confirm_import(outcome)
            control.send("attached")
            receive(control, "failed")
        else:
            assert outcome == "failed"
        # A lost result never implies that the consumer did not use a handle.
        assert backend.release_export(exported).status == "HELD"
        backend.mark_failed(exported)
        snapshot = backend.session_status(exported)
        assert snapshot.state == api.SessionState.POISONED
        assert snapshot.exported_storage_ids == ("storage-0", "storage-1")
        assert all(reference() is not None for reference in sources)
        assert backend.release_export(exported).status == "POISONED"
        assert not cancellations
        control.send("checked")
        receive(control, "exit")  # Parent joins the consumer first.


def _consumer(fault, control):
    payload = receive(control)
    api, backend, context = runtime()
    original_import = backend._codec.import_storage
    imports = []
    uncertain = []

    def import_storage(descriptor, device_index):
        carrier = original_import(descriptor, device_index)
        imports.append(descriptor.storage_id)
        if fault == "partial_import" and len(imports) == 2:
            # Actual rebuild consumed the counter before the injected failure.
            uncertain.append(carrier)
            raise RuntimeError("injected failure after CUDA rebuild")
        return carrier

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend._codec, "import_storage", import_storage)
        if fault == "provider_exit":
            # The supervisor reports death instead of delivering stale handles.
            assert payload == "provider_failed"
            assert not imports
            with pytest.raises(api.WeightSharingError, match="unknown session"):
                backend.session_status(context.identity.session_id)
            control.send("checked")
            return
        if fault == "partial_import":
            with pytest.raises(api.WeightSharingError) as error:
                backend.open_import(payload, _spec(api), context)
            assert error.value.poisoned
        else:
            imported = backend.open_import(payload, _spec(api), context)
            control.send(imported.receipt)
            if fault == "import_ack":
                receive(control, "failed")
            else:
                receive(control, "attached")
                if fault == "consumer_exit":
                    control.send("mapped")
                    receive(control, "exit")
                    os._exit(0)
                result = backend.close_import(imported)
                assert result.local_complete and not result.peer_confirmed
                control.send(result.receipt)
                receive(control, "failed")
            backend.mark_failed(context.identity.session_id)

        snapshot = backend.session_status(context.identity.session_id)
        assert snapshot.state == api.SessionState.POISONED
        assert imports == ["storage-0", "storage-1"]
        if fault == "partial_import":
            assert snapshot.imported_storage_ids == ("storage-0",)
            assert snapshot.active_storage_id == "storage-1"
            assert len(uncertain) == 1
        elif fault == "import_ack":
            assert snapshot.imported_storage_ids == ("storage-0", "storage-1")
        else:
            assert snapshot.imported_storage_ids == ()
        record = backend._sessions[snapshot.session_id]
        if fault != "release_ack":
            assert record.leases and all(not lease.expired() for lease in record.leases)
        result = backend.close_import(snapshot.session_id)
        assert result.status == "POISONED" and result.receipt is None
        assert result.local_complete == (fault == "release_ack")
        with pytest.raises(api.WeightSharingError):
            backend.open_import(payload, _spec(api), context)
        assert imports == ["storage-0", "storage-1"]
        control.send("checked")
        receive(control, "exit")


def _run_failure(fault):
    with spawn_workers((_provider, (fault,)), (_consumer, (fault,))) as workers:
        (_, provider), (consumer_process, consumer) = workers
        consumer.send(receive(provider))
        if fault != "partial_import":
            receipt = receive(consumer)
            if fault == "import_ack":
                consumer.send("failed")  # Drop the import receipt.
            else:
                provider.send(receipt)
                receive(provider, "attached")
                consumer.send("attached")
                if fault == "consumer_exit":
                    receive(consumer, "mapped")
                    consumer.send("exit")
                    consumer_process.join(15)
                    assert consumer_process.exitcode == 0
                else:
                    assert isinstance(receive(consumer), bytes)  # Drop release receipt.
                    consumer.send("failed")
        if fault != "consumer_exit":
            receive(consumer, "checked")
        provider.send("failed")
        receive(provider, "checked")
        if fault != "consumer_exit":
            consumer.send("exit")
            consumer_process.join(15)
            assert consumer_process.exitcode == 0
        provider.send("exit")


@pytest.mark.parametrize("fault", ["partial_import", "import_ack", "release_ack"])
def test_uncertain_cuda_outcome_retains_resources(fault):
    _run_failure(fault)


def test_consumer_exit_does_not_release_provider_weights():
    _run_failure("consumer_exit")


def test_caller_withholds_bundle_after_provider_exit():
    with spawn_workers((_provider, ("provider_exit",)), (_consumer, ("provider_exit",))) as workers:
        (process, provider), (_, consumer) = workers
        assert isinstance(receive(provider), bytes)
        process.join(15)
        assert process.exitcode == 0
        consumer.send("provider_failed")
        receive(consumer, "checked")
