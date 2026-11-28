# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU cache commits preserve selected rows, padding and graph inputs."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.moss_tts.codec_kv_state import commit_cache

pytestmark = [pytest.mark.core_model]


@pytest.fixture
def npu():
    pytest.importorskip("torch_npu")
    assert torch.npu.is_available()
    return torch.npu


def _slots(batch, pool, mode):
    live = batch if mode == "live" else (0 if mode == "padding" else max(1, batch // 2))
    slots = list(reversed(range(live))) + [pool - 1] * (batch - live)
    valid = [True] * live + [False] * (batch - live)
    return slots, valid


def _expected(cache, rows, slots, valid):
    expected = cache.cpu()
    host_rows = rows.cpu()
    for row, (slot, live) in enumerate(zip(slots, valid)):
        if live:
            expected[:, slot] = host_rows[:, row]
    return expected


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["live", "mixed", "padding"])
@pytest.mark.parametrize(
    "batch,heads,capacity,dim",
    [(1, 12, 880, 64), (2, 12, 640, 64), (4, 12, 520, 64), (16, 20, 140, 64), (32, 12, 880, 64), (64, 12, 880, 64)],
)
@torch.no_grad()
def test_cache_commit_preserves_pool(npu, dtype, mode, batch, heads, capacity, dim):
    pool = 65
    cache = torch.empty((2, pool, heads, capacity, dim), dtype=dtype, device="npu").uniform_(-1, 1)
    rows = torch.empty((2, batch, heads, capacity, dim), dtype=dtype, device="npu").uniform_(-2, 2)
    ids, flags = _slots(batch, pool, mode)
    expected = _expected(cache, rows, ids, flags)
    ptr = cache.data_ptr()
    commit_cache(
        rows,
        cache,
        torch.tensor(ids, device="npu"),
        torch.tensor(flags, device="npu"),
        torch.zeros(batch, dtype=torch.long, device="npu"),
        15,
    )
    npu.synchronize()
    assert cache.data_ptr() == ptr
    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_cache_commit_accepts_strided_rows(npu, dtype):
    cache = torch.randn((2, 7, 2, 11, 8), device="npu", dtype=dtype)
    rows = torch.randn((2, 4, 2, 8, 11), device="npu", dtype=dtype).transpose(-1, -2)
    assert not rows.is_contiguous()
    ids, flags = [4, 0, 6, 6], [True, True, False, False]
    expected = _expected(cache, rows, ids, flags)
    commit_cache(
        rows,
        cache,
        torch.tensor(ids, device="npu"),
        torch.tensor(flags, device="npu"),
        torch.zeros(4, dtype=torch.long, device="npu"),
        3,
    )
    npu.synchronize()
    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_cache_commit_replays_changed_slots_and_padding(npu, dtype):
    cache = torch.randn((2, 9, 2, 17, 8), device="npu", dtype=dtype)
    rows = torch.randn((2, 4, 2, 17, 8), device="npu", dtype=dtype)
    slots = torch.tensor([0, 3, 8, 8], device="npu")
    valid = torch.tensor([True, True, False, False], device="npu")
    offsets = torch.zeros(4, dtype=torch.long, device="npu")
    for _ in range(2):
        commit_cache(rows, cache, slots, valid, offsets, 3)
    npu.synchronize()
    graph = npu.NPUGraph()
    with npu.graph(graph):
        commit_cache(rows, cache, slots, valid, offsets, 3)
    npu.synchronize()
    patterns = [
        ([7, 1, 8, 8], [True, True, False, False]),
        ([8, 8, 8, 8], [False, False, False, False]),
        ([0, 5, 2, 6], [True, True, True, True]),
        ([1, 8, 8, 8], [True, False, False, False]),
    ]
    for iteration in range(24):
        ids, flags = patterns[iteration % len(patterns)]
        rows.uniform_(-2, 2)
        slots.copy_(torch.tensor(ids, device="npu"))
        valid.copy_(torch.tensor(flags, device="npu"))
        # Preserve accumulated cache state across replays: untouched slots
        # and the duplicate padding destination must remain unchanged.
        expected = _expected(cache, rows, ids, flags)
        graph.replay()
        npu.synchronize()
        torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
