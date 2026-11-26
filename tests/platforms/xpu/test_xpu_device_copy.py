# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import gc

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.omni,
    *hardware_marks(res={"xpu": "B60"}, num_cards=1),
    pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="XPU required"),
]


@torch.inference_mode()
def test_xpu_copies_use_a_pinned_nonblocking_transfer(monkeypatch):
    # A pageable host copy waits for every kernel queued on the XPU stream,
    # which serialized the MRv2 talker's post_sample with its forward.
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    original = torch.Tensor.to
    transfers = []

    def recording_to(self, *args, **kwargs):
        if self.device.type == "cpu":
            # Tensor.to(device, dtype, non_blocking, copy): the first positional bool is non_blocking.
            positional = next((a for a in args if isinstance(a, bool)), False)
            transfers.append((self.is_pinned(), kwargs.get("non_blocking", positional)))
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", recording_to)
        results = [to_device_nonblocking(host, "xpu") for host in (source, source.pin_memory())]
        index = index_to_device([3, 1, 2], "xpu")
    assert transfers == [(True, True), (True, True), (True, True)]
    for result in results:
        torch.testing.assert_close(result.cpu(), source)
        assert to_device_nonblocking(result, result.device) is result
    assert index.tolist() == [3, 1, 2] and index.dtype == torch.long


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_xpu_index_copy_keeps_temporary_sources_alive(dtype):
    # Queue work before each upload, release its temporary pinned source,
    # and repeatedly reuse the host allocator before reading any result.
    x = torch.randn(512, 512, device="xpu")
    scratch = torch.empty_like(x)
    results = []
    for value in range(64):
        torch.mm(x, x, out=scratch)
        results.append(index_to_device([value] * 32, "xpu", dtype=dtype))
    gc.collect()
    for value, result in enumerate(results):
        torch.testing.assert_close(result.cpu(), torch.full((32,), value, dtype=dtype))
    assert index_to_device([], "xpu", dtype=dtype).shape == (0,)
