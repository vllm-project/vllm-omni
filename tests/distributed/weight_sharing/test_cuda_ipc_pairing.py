# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Physical GPU pairing with different process-local CUDA ordinals."""

import importlib

import pytest
import torch

from tests.distributed.weight_sharing.cuda_helpers import receive, require_cuda, runtime, spawn_workers
from tests.distributed.weight_sharing.helpers import load_weight_sharing
from tests.helpers.mark import hardware_marks

require_cuda()
pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=2)]


@pytest.fixture
def gpu_uuids():
    require_cuda(2)
    codec = importlib.import_module(f"{load_weight_sharing().__name__}.torch_cuda_ipc_codec")
    return tuple(codec._device_uuid(index) for index in range(2))


def _spec(api):
    return api.WeightSpec("paired-model", "revision", "tp2-final", (api.TensorSpec("weight", "float32", (8, 8)),))


def _provider(rank, control):
    api, backend, context = runtime(rank=rank, tp_size=2, device_index=rank)
    weights = {"weight": torch.full((8, 8), rank + 1.0, device=f"cuda:{rank}")}
    exported = backend.prepare_export(weights, _spec(api), context)
    control.send((context.identity.device_uuid, backend.claim_export(exported)))
    outcome = receive(control)
    if outcome == "rejected":
        backend.mark_failed(exported)
        assert backend.release_export(exported).status == "POISONED"
    else:
        backend.confirm_import(outcome)
        control.send("attached")
        backend.confirm_release(receive(control))
        assert backend.release_export(exported).status == "CLOSED"
    control.send("closed")


def _consumer(rank, mismatch, control):
    payload = receive(control)
    device_index = rank if mismatch == "device" else 1 - rank
    api, backend, context = runtime(
        rank=1 - rank if mismatch == "rank" else rank,
        tp_size=2,
        device_index=device_index,
    )
    if mismatch:
        calls = 0
        original_import = backend._codec.import_storage

        def counted_import(descriptor, index):
            nonlocal calls
            calls += 1
            return original_import(descriptor, index)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(backend._codec, "import_storage", counted_import)
            with pytest.raises(api.WeightSharingError):
                backend.open_import(payload, _spec(api), context)
        assert calls == 0
        with pytest.raises(api.WeightSharingError, match="unknown session"):
            backend.session_status(context.identity.session_id)
        control.send("rejected")
    else:
        imported = backend.open_import(payload, _spec(api), context)
        control.send(imported.receipt)
        receive(control, "attached")
        assert imported["weight"].device.index == device_index
        output = torch.ones((2, 8), device=f"cuda:{device_index}") @ imported["weight"]
        torch.testing.assert_close(output.cpu(), torch.full((2, 8), 8.0 * (rank + 1)))
        result = backend.close_import(imported)
        assert result.status == "CLOSED"
        control.send(result.receipt)


def _check_pairs(gpu_uuids, ranks, mismatch=None):
    workers = []
    for rank in ranks:
        workers.extend(
            [
                (_provider, (rank,), {"CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids)}),
                (_consumer, (rank, mismatch), {"CUDA_VISIBLE_DEVICES": ",".join(reversed(gpu_uuids))}),
            ]
        )
    with spawn_workers(*workers) as started:
        for rank, (_, provider), (_, consumer) in zip(ranks, started[::2], started[1::2]):
            device_uuid, payload = receive(provider)
            assert device_uuid == gpu_uuids[rank]
            consumer.send(payload)
            provider.send(receive(consumer))
            if not mismatch:
                receive(provider, "attached")
                consumer.send("attached")
                provider.send(receive(consumer))
            receive(provider, "closed")


def test_rank_pairs_share_with_reversed_visible_device_order(gpu_uuids):
    _check_pairs(gpu_uuids, (0, 1))


@pytest.mark.parametrize("mismatch", ["rank", "device"])
def test_wrong_rank_or_physical_gpu_is_rejected_before_import(gpu_uuids, mismatch):
    _check_pairs(gpu_uuids, (0,), mismatch)
