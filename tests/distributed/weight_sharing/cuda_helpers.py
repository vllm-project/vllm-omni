# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared process harness for real CUDA tests; pipes carry control messages only."""

import importlib
import os
import sys
from dataclasses import replace

import pytest
import torch

from tests.distributed.weight_sharing.helpers import load_weight_sharing, make_context
from tests.distributed.weight_sharing.helpers import publish_to_parent as publish_to_parent
from tests.distributed.weight_sharing.helpers import receive as receive
from tests.distributed.weight_sharing.helpers import relay_import as relay_import
from tests.distributed.weight_sharing.helpers import relay_release as relay_release
from tests.distributed.weight_sharing.helpers import release_from_parent as release_from_parent
from tests.distributed.weight_sharing.helpers import spawn_workers as spawn_workers


def require_cuda(num_devices=0):
    # The import-time check must not initialize CUDA: spawned workers can have
    # different CUDA_VISIBLE_DEVICES settings, installed before Process.start().
    if sys.platform != "linux" or torch.version.cuda is None or torch.version.hip is not None:
        pytest.skip("CUDA IPC requires Linux and NVIDIA CUDA", allow_module_level=True)
    if num_devices and torch.accelerator.device_count() < num_devices:
        pytest.skip(f"CUDA IPC test requires {num_devices} NVIDIA GPU(s)")


def runtime(*, device_index=0, rank=0, tp_size=1, session_id="session"):
    package = load_weight_sharing()
    api = importlib.import_module(f"{package.__name__}.contracts")
    backend = importlib.import_module(f"{package.__name__}.backend")
    codec = importlib.import_module(f"{package.__name__}.torch_cuda_ipc_codec")
    context = make_context(
        api,
        device_uuid=codec._device_uuid(device_index),
        rank=rank,
        tp_size=tp_size,
        provider_id=f"ar-rank-{rank}",
        consumer_id=f"dit-rank-{rank}",
        session_id=session_id,
    )
    return api, backend.TorchCudaIpcBackend(), replace(context, device_index=device_index)


def assert_export_allocations(bindings):
    """Verify the real allocation kind, not just the worker's requested setting."""
    expandable = "expandable_segments:True" in os.environ.get("PYTORCH_ALLOC_CONF", "")
    snapshot = torch.cuda.memory._snapshot()
    assert snapshot["allocator_settings"]["expandable_segments"] is expandable
    for tensor in bindings.values():
        storage = tensor.untyped_storage()
        if not storage.nbytes():
            continue
        owners = [
            segment
            for segment in snapshot["segments"]
            if segment["address"] <= storage.data_ptr() < segment["address"] + segment["total_size"]
        ]
        assert len(owners) == 1 and owners[0]["is_expandable"] is expandable
