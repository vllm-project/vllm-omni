# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import gc

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_marks
from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


@torch.inference_mode()
def test_npu_copy_handles_strides_and_uses_nonblocking_transfer(monkeypatch):
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    original = torch.Tensor.to
    transfers = []

    def recording_to(self, *args, **kwargs):
        if self.device.type == "cpu":
            transfers.append((self.is_pinned(), kwargs.get("non_blocking", False)))
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", recording_to)
        results = [to_device_nonblocking(host, "npu") for host in (source, source.pin_memory())]
        for result in results:
            assert to_device_nonblocking(result, result.device) is result
    assert transfers == [(True, True), (True, True)]
    for result in results:
        torch.testing.assert_close(result.cpu(), source)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_npu_index_copy_keeps_temporary_sources_alive(dtype):
    # Queue work before each upload, release its temporary pinned source,
    # and repeatedly reuse the host allocator before reading any result.
    x = torch.randn(512, 512, device="npu")
    scratch = torch.empty_like(x)
    results = []
    for value in range(64):
        torch.mm(x, x, out=scratch)
        results.append(index_to_device([value] * 32, "npu", dtype=dtype))
    gc.collect()
    for value, result in enumerate(results):
        torch.testing.assert_close(result.cpu(), torch.full((32,), value, dtype=dtype))
    assert index_to_device([], "npu", dtype=dtype).shape == (0,)
