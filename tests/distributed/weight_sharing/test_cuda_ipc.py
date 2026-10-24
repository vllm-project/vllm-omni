# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small real CUDA IPC tests; no checkpoint or serving processes are required."""

import gc
import importlib
import multiprocessing
import sys
from pathlib import Path

import pytest
import torch
from packaging.version import Version

from tests.distributed.weight_sharing.helpers import load_weight_sharing, make_context
from tests.helpers.mark import hardware_marks

if sys.platform != "linux" or torch.version.cuda is None:
    pytest.skip("real CUDA IPC requires Linux and a CUDA-enabled PyTorch build", allow_module_level=True)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.fixture(autouse=True)
def supported_runtime():
    version = Version(torch.__version__)
    if (
        sys.platform != "linux"
        or version.release[:2] not in {(2, 10), (2, 13)}
        or version.is_prerelease
        or version.is_devrelease
    ):
        pytest.skip("CUDA IPC codec requires Linux and the PyTorch 2.10 or 2.13 release ABI")
    if torch.version.cuda is None or torch.version.hip is not None or not torch.accelerator.is_available():
        pytest.skip("NVIDIA CUDA is unavailable")


def _runtime(endpoint):
    package = load_weight_sharing()
    api = importlib.import_module(f"{package.__name__}.contracts")
    backend_module = importlib.import_module(f"{package.__name__}.backend")
    codec = importlib.import_module(f"{package.__name__}.torch_cuda_ipc_codec")
    context = make_context(api, endpoint, device_uuid=codec._device_uuid(0))
    return api, backend_module.TorchCudaIpcBackend(), context


def _spec(api):
    return api.WeightSpec(
        "small-test-model",
        "revision",
        "tp1-final",
        (
            api.TensorSpec("first", "float32", (128, 256), (256, 1)),
            api.TensorSpec("alias", "float32", (128, 256), (256, 1), "first"),
            api.TensorSpec("transposed", "float32", (256, 128), (1, 256)),
            api.TensorSpec("empty", "float32", (0,), (1,)),
        ),
    )


def _provider(endpoint, control):
    api, backend, context = _runtime(endpoint)
    base = torch.arange(256 * 256, dtype=torch.float32, device="cuda:0").reshape(256, 256)
    first = base[:128]
    bindings = {"first": first, "alias": first, "transposed": base[128:].T, "empty": torch.empty(0, device="cuda:0")}
    exported = backend.prepare_export(bindings, _spec(api), context)
    snapshot = backend.session_status(exported)
    assert snapshot.storage_count == 2
    assert len(snapshot.exported_storage_ids) == 1
    assert snapshot.active_storage_id is None
    control.send("ready")
    assert control.poll(45), "consumer did not finish"
    assert control.recv() == "release"
    assert backend.release_export(exported).status == "CLOSED"
    assert backend.release_export(exported).status == "CLOSED"
    closed = backend.session_status(exported)
    assert closed.state == api.SessionState.CLOSED
    assert closed.exported_storage_ids == snapshot.exported_storage_ids
    control.send("closed")


def _consumer(endpoint, control):
    api, backend, context = _runtime(endpoint)
    warmup = torch.empty(1, device="cuda:0")
    torch.accelerator.synchronize(0)
    before = torch.accelerator.memory.memory_allocated(0)
    imported = backend.open_import(endpoint, _spec(api), context)
    after = torch.accelerator.memory.memory_allocated(0)
    # Importing metadata/views must not allocate another full weight storage.
    assert after - before < 256 * 256 * 4
    assert imported["first"] is imported["alias"]
    assert imported["first"].untyped_storage()._cdata == imported["transposed"].untyped_storage()._cdata
    assert imported["transposed"].stride() == (1, 256)
    assert imported["transposed"].storage_offset() == 128 * 256
    expected = torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256)
    torch.testing.assert_close(imported["first"].cpu(), expected[:128])
    torch.testing.assert_close(imported["transposed"].cpu(), expected[128:].T)
    assert imported["empty"].numel() == 0
    with pytest.raises(api.WeightSharingError):
        backend.open_import(endpoint, _spec(api), context)

    external_view = imported["first"].view(-1)
    assert backend.close_import(imported).status == "HELD"
    del external_view
    gc.collect()
    assert backend.close_import(imported).status == "CLOSED"
    assert backend.close_import(imported).status == "CLOSED"
    assert backend.session_status(imported).imported_storage_ids == ()
    del warmup
    control.send("closed")


def test_two_spawned_processes_share_storage_and_return_references(tmp_path: Path):
    endpoint = str(tmp_path / "weights.sock")
    context = multiprocessing.get_context("spawn")
    provider_parent, provider_child = context.Pipe()
    consumer_parent, consumer_child = context.Pipe()
    provider = context.Process(target=_provider, args=(endpoint, provider_child))
    consumer = context.Process(target=_consumer, args=(endpoint, consumer_child))
    provider.start()
    try:
        assert provider_parent.poll(45), "provider did not publish"
        assert provider_parent.recv() == "ready"
        consumer.start()
        assert consumer_parent.poll(45), "consumer did not close"
        assert consumer_parent.recv() == "closed"
        provider_parent.send("release")
        assert provider_parent.poll(15)
        assert provider_parent.recv() == "closed"
        provider.join(timeout=15)
        consumer.join(timeout=15)
        assert provider.exitcode == 0
        assert consumer.exitcode == 0
    finally:
        for process in (provider, consumer):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
        for connection in (provider_parent, provider_child, consumer_parent, consumer_child):
            connection.close()


def test_unexposed_export_is_cancelled_once(tmp_path: Path):
    api, backend, context = _runtime(str(tmp_path / "unexposed.sock"))
    weights = {"weight": torch.arange(16, dtype=torch.float32, device="cuda:0")}
    spec = api.WeightSpec("m", "r", "l", (api.TensorSpec("weight", "float32", (16,), (1,)),))
    exported = backend.prepare_export(weights, spec, context)
    assert backend.release_export(exported).status == "CLOSED"
    assert backend.release_export(exported).status == "CLOSED"
    assert backend.session_status(exported).state == api.SessionState.CLOSED
    torch.testing.assert_close(weights["weight"].cpu(), torch.arange(16, dtype=torch.float32))
