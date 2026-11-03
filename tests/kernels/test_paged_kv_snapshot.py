# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-layer paged KV gather writes directly into strided snapshot storage."""

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks
from vllm_omni.worker.kv_snapshot import PagedKVCopy

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required"),
]


@pytest.mark.parametrize("length", [1, 16, 23, 32])
@pytest.mark.parametrize("layout", ["stacked", "packed"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_snapshot(length, layout, dtype, monkeypatch):
    layers, pages, page_size, heads, dim = 3, 5, 16, 2, 64
    pairs = []
    for _ in range(layers):
        k = torch.randn(pages, page_size, heads, dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        if layout == "stacked":
            cache = torch.stack((k, v))
            pairs.append((cache[0], cache[1]))
        else:
            cache = torch.cat((k, v), dim=-1).transpose(1, 2).contiguous()
            pairs.append(tuple(cache.transpose(1, 2).split(dim, dim=-1)))
    copier = PagedKVCopy(pairs)
    table = torch.tensor([[0, 0], [4, 1]], device="cuda", dtype=torch.int32)
    capacity = 40  # Retain END/NAR space; head slicing also makes targets strided.
    key_storage = torch.full((layers, capacity, heads * 2, dim), -99, device="cuda", dtype=dtype)
    value_storage = torch.full_like(key_storage, -99)
    keys, values = key_storage[:, :, ::2], value_storage[:, :, ::2]
    expected = [(k[[4, 1]].flatten(0, 1)[:length], v[[4, 1]].flatten(0, 1)[:length]) for k, v in pairs]
    with monkeypatch.context() as patch:
        # Any per-copy CPU index construction/upload is a regression.
        patch.setattr(torch, "tensor", lambda *args, **kwargs: pytest.fail("unexpected tensor upload"))
        assert copier.copy(table, 1, length, keys, values)
    for layer, (k, v) in enumerate(expected):
        torch.testing.assert_close(keys[layer, :length], k, rtol=0, atol=0)
        torch.testing.assert_close(values[layer, :length], v, rtol=0, atol=0)
    assert (keys[:, length:] == -99).all()
    assert (values[:, length:] == -99).all()
    assert (key_storage[:, :, 1::2] == -99).all()
    assert (value_storage[:, :, 1::2] == -99).all()
    # Reuse the copier with a new page mapping and independently owned output.
    table[1].copy_(table[0])
    assert copier.copy(table, 1, length, keys, values)
    for layer, (k, v) in enumerate(pairs):
        ids = table[1].long()
        torch.testing.assert_close(keys[layer, :length], k[ids].flatten(0, 1)[:length], rtol=0, atol=0)
        torch.testing.assert_close(values[layer, :length], v[ids].flatten(0, 1)[:length], rtol=0, atol=0)
