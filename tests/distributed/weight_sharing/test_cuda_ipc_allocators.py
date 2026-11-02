# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unsupported allocators and expandable allocations without IPC handles fail closed."""

import pytest
import torch

from tests.distributed.weight_sharing.cuda_helpers import (
    assert_export_allocations,
    receive,
    require_cuda,
    runtime,
    spawn_workers,
)
from tests.distributed.weight_sharing.helpers import FakeCodec, make_weights
from tests.helpers.mark import hardware_marks

require_cuda()

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.fixture(autouse=True)
def supported_runtime():
    require_cuda(1)


def _unsupported_allocator(mode, expected_error, control):
    api, backend, context = runtime()
    if mode == "private_pool":
        # Keep the pool alive until rejection has been checked.
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            weight = torch.ones(1024, dtype=torch.float32, device="cuda:0")
        segments = torch.cuda.memory._snapshot()["segments"]
        owners = [
            segment
            for segment in segments
            if segment["address"] <= weight.data_ptr() < segment["address"] + segment["total_size"]
        ]
        assert owners and all(tuple(segment["segment_pool_id"]) != (0, 0) for segment in owners)
    else:
        weight = torch.ones(1024, dtype=torch.float32, device="cuda:0")
        if mode == "ipc_disabled":
            assert_export_allocations({"weight": weight})

    spec = api.WeightSpec("guard-test", "revision", "flat", (api.TensorSpec("weight", "float32", (1024,)),))
    with pytest.raises(api.WeightSharingError) as error:
        backend.prepare_export({"weight": weight}, spec, context)
    assert isinstance(error.value.__cause__, RuntimeError if mode == "ipc_disabled" else api.UnsupportedError)
    assert expected_error in str(error.value.__cause__)
    assert backend.session_status(context.identity.session_id).exported_storage_ids == ()
    control.send("checked")


@pytest.mark.parametrize(
    ("mode", "allocator_conf", "expected_error"),
    [
        pytest.param("async_allocator", "backend:cudaMallocAsync", "only the native", id="cudaMallocAsync"),
        pytest.param(
            "ipc_disabled",
            "backend:native,expandable_segments:True",
            "IPC handles are disabled",
            id="expandable-ipc-disabled",
        ),
        pytest.param(
            "private_pool", "backend:native,expandable_segments:False", "outside private pools", id="private_pool"
        ),
    ],
)
def test_unsupported_allocator_rejected_before_export(monkeypatch, mode, allocator_conf, expected_error):
    # Allocator configuration is read on import/initialization, so each case
    # needs a fresh spawned process with its environment set before startup.
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    with spawn_workers(
        (
            _unsupported_allocator,
            (mode, expected_error),
            {
                "PYTORCH_ALLOC_CONF": allocator_conf,
                "TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC": "0" if mode == "ipc_disabled" else "1",
            },
        )
    ) as workers:
        ((_, control),) = workers
        receive(control, "checked")


def _async_consumer(control):
    api, backend, context = runtime()
    # A valid metadata bundle lets the actual consumer runtime check run before
    # importing anything. No real exporter is needed to reject its allocator.
    provider = type(backend)(codec=FakeCodec(api))
    bindings, spec = make_weights(api)
    payload = provider.claim_export(provider.prepare_export(bindings, spec, context))
    with pytest.raises(api.UnsupportedError, match="only the native"):
        backend.open_import(payload, spec, context)
    with pytest.raises(api.WeightSharingError, match="unknown session"):
        backend.session_status(context.identity.session_id)
    control.send("checked")


def test_consumer_rejects_async_allocator_before_import(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    with spawn_workers((_async_consumer, (), {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"})) as workers:
        receive(workers[0][1], "checked")
