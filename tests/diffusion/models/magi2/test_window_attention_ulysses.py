# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from tests.diffusion.models.magi2.test_window_attention_paged import _install_fa3, _platform
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.utils import fa
from vllm_omni.diffusion.models.magi2 import window_attention
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup, balanced_split_sizes
from vllm_omni.diffusion.models.magi2.window_attention import (
    Magi2WindowAttentionKernel,
    Magi2WindowAttentionPlan,
    build_window_attention_plan,
    ulysses_window_attention,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture(autouse=True)
def clear_provider_cache():
    fa._external_fa3_varlen.cache_clear()
    fa.flash_attn_3_varlen_unsupported.cache_clear()
    yield
    fa._external_fa3_varlen.cache_clear()
    fa.flash_attn_3_varlen_unsupported.cache_clear()


class _ThreadedAllToAll:
    """``dist.all_to_all_single`` between Ulysses ranks simulated as threads of one process."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size, timeout=60)
        self.sent: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
        self.calls = [0] * world_size

    def __call__(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        group: Any = None,
        async_op: bool = False,
    ) -> None:
        rank = group.rank
        self.sent[rank] = list(torch.split(input, input_split_sizes))
        self.barrier.wait()
        received = [self.sent[source][rank] for source in range(self.world_size)]
        assert [chunk.shape[0] for chunk in received] == list(output_split_sizes)
        output.copy_(torch.cat(received))
        self.calls[rank] += 1
        self.barrier.wait()


def _run_ranks(
    monkeypatch: pytest.MonkeyPatch, world_size: int, run: Callable[[Magi2ParallelGroup], torch.Tensor]
) -> tuple[list[torch.Tensor], _ThreadedAllToAll]:
    exchange = _ThreadedAllToAll(world_size)
    monkeypatch.setattr(dist, "all_to_all_single", exchange)
    outputs: list[torch.Tensor | None] = [None] * world_size
    errors: list[BaseException] = []

    def worker(rank: int) -> None:
        try:
            outputs[rank] = run(Magi2ParallelGroup(SimpleNamespace(rank=rank), world_size, rank))
        except BaseException as error:  # noqa: BLE001 - re-raised in the test thread
            errors.append(error)
            exchange.barrier.abort()

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return [output for output in outputs if output is not None], exchange


def _inputs(plan: Magi2WindowAttentionPlan, num_heads_q: int, num_heads_kv: int, dtype=torch.float32):
    generator = torch.Generator().manual_seed(num_heads_q * 10 + num_heads_kv)
    q = torch.randn(plan.num_tokens, num_heads_q, 16, generator=generator).to(dtype)
    k = torch.randn(plan.num_tokens, num_heads_kv, 16, generator=generator).to(dtype)
    v = torch.randn(plan.num_tokens, num_heads_kv, 16, generator=generator).to(dtype)
    return q, k, v


def _local(tensor: torch.Tensor, split_sizes: list[int], rank: int) -> torch.Tensor:
    start = sum(split_sizes[:rank])
    return tensor[start : start + split_sizes[rank]]


def _head_group_attention(q, k, v, plan, world_size: int, **kwargs: Any) -> torch.Tensor:
    """Single-rank window attention evaluated on each Ulysses rank's heads, as that rank evaluates it."""

    num_q, num_kv = q.shape[1] // world_size, k.shape[1] // world_size
    groups = [
        window_attention.window_attention(
            q[:, r * num_q : (r + 1) * num_q],
            k[:, r * num_kv : (r + 1) * num_kv],
            v[:, r * num_kv : (r + 1) * num_kv],
            plan,
            **kwargs,
        )
        for r in range(world_size)
    ]
    return torch.cat(groups, dim=1)


# Partitions of 309 tokens: the balanced ones the data proxy shards with, and arbitrary ones with empty ranks.
@pytest.mark.parametrize(
    "split_sizes",
    [[155, 154], [200, 109], [78, 77, 77, 77], [0, 150, 159, 0]],
    ids=["balanced-2", "uneven-2", "balanced-4", "empty-ranks-4"],
)
@pytest.mark.parametrize(("num_heads_q", "num_heads_kv"), [(8, 4), (16, 4)])
def test_ulysses_ranks_reproduce_the_single_rank_result(
    monkeypatch: pytest.MonkeyPatch, split_sizes: list[int], num_heads_q: int, num_heads_kv: int
) -> None:
    # The partial last t-block and the partial text page are both present.
    plan = build_window_attention_plan((9, 4, 8), 21)
    assert sum(split_sizes) == plan.num_tokens
    world_size = len(split_sizes)
    q, k, v = _inputs(plan, num_heads_q, num_heads_kv)
    expected = window_attention.window_attention(q, k, v, plan, softmax_scale=0.3)

    def run(group: Magi2ParallelGroup) -> torch.Tensor:
        local = (_local(tensor, split_sizes, group.rank) for tensor in (q, k, v))
        return ulysses_window_attention(*local, plan, split_sizes, softmax_scale=0.3, group=group)

    outputs, exchange = _run_ranks(monkeypatch, world_size, run)

    # Two exchanges per rank: q/k/v to heads, the output back to tokens.
    assert exchange.calls == [2] * world_size
    for rank, output in enumerate(outputs):
        assert output.shape == (split_sizes[rank], num_heads_q, 16)
    output = torch.cat(outputs)
    assert torch.equal(output, _head_group_attention(q, k, v, plan, world_size, softmax_scale=0.3))
    # The CPU reference's last bits depend on how many heads one call evaluates.
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


# Partitions of 1186 tokens; in the uneven one the third rank's slice crosses into the text.
@pytest.mark.parametrize(
    "split_sizes", [[297, 297, 296, 296], [100, 0, 1000, 86]], ids=["balanced", "uneven-with-empty-rank"]
)
def test_ulysses_ranks_reproduce_the_single_rank_paged_result(
    monkeypatch: pytest.MonkeyPatch, split_sizes: list[int]
) -> None:
    _platform(monkeypatch, "musa")
    calls = _install_fa3(monkeypatch)
    plan = build_window_attention_plan((11, 8, 12), 130)
    assert sum(split_sizes) == plan.num_tokens
    q, k, v = _inputs(plan, 8, 4)
    expected = window_attention.window_attention(q, k, v, plan)
    calls.clear()

    def run(group: Magi2ParallelGroup) -> torch.Tensor:
        local = (_local(tensor, split_sizes, group.rank) for tensor in (q, k, v))
        return ulysses_window_attention(*local, plan, split_sizes, group=group)

    outputs, _ = _run_ranks(monkeypatch, 4, run)

    # Every rank runs the video and the text call on its own two query heads and one key/value head.
    assert len(calls) == 8
    assert all(call["q"].shape[1] == 2 and call["k"].shape[2] == 1 for call in calls)
    output = torch.cat(outputs)
    assert torch.equal(output, _head_group_attention(q, k, v, plan, 4))
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


def test_window_attention_kernel_uses_the_ulysses_group(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = build_window_attention_plan((9, 4, 8), 21)
    q, k, v = _inputs(plan, 8, 2)
    expected = window_attention.window_attention(q, k, v, plan, softmax_scale=0.25)
    # Not the balanced partition, so the kernel must pass the metadata's split sizes through.
    split_sizes = [200, 109]
    rank_group = threading.local()
    monkeypatch.setattr(window_attention, "get_magi2_ulysses_group", lambda: rank_group.value)
    kernel = Magi2WindowAttentionKernel(0.25)
    metadata = AttentionMetadata(extra={"magi2_window_plan": plan, "magi2_split_sizes": split_sizes})

    def run(group: Magi2ParallelGroup) -> torch.Tensor:
        rank_group.value = group
        local = (_local(tensor, split_sizes, group.rank) for tensor in (q, k, v))
        return kernel(*local, metadata)

    outputs, _ = _run_ranks(monkeypatch, 2, run)

    assert torch.equal(torch.cat(outputs), _head_group_attention(q, k, v, plan, 2, softmax_scale=0.25))
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(("world_size", "num_heads_q", "num_heads_kv"), [(4, 8, 2), (2, 3, 1)])
def test_ulysses_refuses_heads_that_do_not_divide(
    monkeypatch: pytest.MonkeyPatch, world_size: int, num_heads_q: int, num_heads_kv: int
) -> None:
    monkeypatch.setattr(dist, "all_to_all_single", lambda *args, **kwargs: pytest.fail("no exchange may start"))
    monkeypatch.setattr(fa, "_external_fa3_varlen", lambda: pytest.fail("no kernel may run"))
    plan = build_window_attention_plan((9, 4, 8), 3)
    split_sizes = balanced_split_sizes(plan.num_tokens, world_size)
    q, k, v = (_local(tensor, split_sizes, 0) for tensor in _inputs(plan, num_heads_q, num_heads_kv))

    with pytest.raises(ValueError, match=rf"{num_heads_kv} key/value heads to divide by the Ulysses size {world_size}"):
        ulysses_window_attention(q, k, v, plan, split_sizes, group=Magi2ParallelGroup(None, world_size, 0))
