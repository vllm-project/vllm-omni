# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import gc
import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_mlu")

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"mlu": "MLU580"}, num_cards=1),
    pytest.mark.skipif(not torch.mlu.is_available(), reason="MLU required"),
]

# These tensor helpers need no Omni engine or model registry. Load the actual
# source directly so the hardware regression also runs with vendor runtimes
# whose vLLM version differs from the Omni engine's dependency.
_SOURCE = Path(__file__).resolve().parents[3] / "vllm_omni/utils/device_copy.py"
_SPEC = importlib.util.spec_from_file_location("mlu_device_copy", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
device_copy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(device_copy)


@torch.inference_mode()
def test_mlu_copies_use_pinned_nonblocking_transfers(monkeypatch):
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    original = torch.Tensor.to
    transfers = []

    def recording_to(self, *args, **kwargs):
        if self.device.type == "cpu":
            transfers.append((self.is_pinned(), kwargs.get("non_blocking", False)))
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", recording_to)
        results = [device_copy.to_device_nonblocking(host, "mlu") for host in (source, source.pin_memory())]
        index = device_copy.index_to_device([3, 1, 2], torch.device("mlu", 0))
        staged = device_copy.DeviceStager()([[1, 2], [3, 4]], "mlu")
    assert transfers == [(True, True)] * 4
    for result in results:
        torch.testing.assert_close(result.cpu(), source)
        assert device_copy.to_device_nonblocking(result, result.device) is result
    assert index.tolist() == [3, 1, 2] and index.dtype == torch.long
    assert staged.cpu().tolist() == [[1, 2], [3, 4]]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_mlu_temporary_pinned_sources_survive_queued_copies(dtype):
    streams = [torch.mlu.Stream(), torch.mlu.Stream()]
    results = []
    for stream in streams:
        with torch.mlu.stream(stream):
            # Delay the copies while temporary pinned allocations are released
            # and other streams allocate from the same caching host allocator.
            torch.mlu._sleep(130_000_000)
    for value in range(64):
        with torch.mlu.stream(streams[value % len(streams)]):
            host = torch.full((32,), value, dtype=dtype)
            results.append(device_copy.to_device_nonblocking(host, "mlu"))
            results.append(device_copy.index_to_device([value] * 32, "mlu", dtype=dtype))
        torch.empty(32, dtype=dtype, pin_memory=True).fill_(-1)
    gc.collect()
    for stream in streams:
        stream.synchronize()
    for index, result in enumerate(results):
        torch.testing.assert_close(result.cpu(), torch.full((32,), index // 2, dtype=dtype))
    assert device_copy.index_to_device([], "mlu", dtype=dtype).shape == (0,)


def test_cpu_fallback_preserves_tensor_and_dtype():
    source = torch.arange(6, dtype=torch.float64).reshape(2, 3).T
    assert device_copy.to_device_nonblocking(source, "cpu") is source
    index = device_copy.index_to_device([2, 1], "cpu", dtype=torch.int32)
    assert index.tolist() == [2, 1] and index.dtype == torch.int32
