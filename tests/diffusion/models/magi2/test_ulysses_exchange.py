# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Ulysses exchanges must hand every rank the same bytes as the concatenating reference."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.magi2 import parallel
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup, balanced_split_sizes

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

_BITS = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}


def _reference_scatter_heads_gather_seqlen(tensors, split_sizes, group):
    """The Q/K/V exchange that packed through per-input copies and a concatenation."""
    tensors = list(tensors)
    if group.world_size == 1:
        return tensors
    local_tokens = split_sizes[group.rank]
    reshaped = []
    local_head_counts = []
    head_dim = tensors[0].shape[-1]
    for tensor in tensors:
        local_heads = tensor.shape[1] // group.world_size
        local_head_counts.append(local_heads)
        reshaped.append(
            tensor.view(local_tokens, group.world_size, local_heads, head_dim)
            .permute(1, 0, 2, 3)
            .reshape(group.world_size * local_tokens, local_heads, head_dim)
        )
    fused = torch.cat(reshaped, dim=1).contiguous()
    output = torch.empty((sum(split_sizes), fused.shape[1], head_dim), dtype=fused.dtype, device=fused.device)
    parallel.dist.all_to_all_single(
        output,
        fused,
        output_split_sizes=split_sizes,
        input_split_sizes=[local_tokens] * group.world_size,
        group=group.group,
    )
    return list(torch.split(output, local_head_counts, dim=1))


class _Exchange:
    """In-process all-to-all over every rank of one group.

    A first pass records each rank's send buffer; a second pass delivers, to
    each rank, the chunks every source addressed to it.
    """

    def __init__(self, world_size):
        self.world_size = world_size
        self.rank = 0
        self.sent = {}
        self.delivering = False

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes, group):
        assert input.is_contiguous() and output.is_contiguous()
        if not self.delivering:
            self.sent[self.rank] = (input.clone(), list(input_split_sizes), input.data_ptr())
            return
        chunks = [self.sent[source][0].split(self.sent[source][1])[self.rank] for source in range(self.world_size)]
        assert [chunk.shape[0] for chunk in chunks] == list(output_split_sizes)
        output.copy_(torch.cat(chunks))

    def run(self, monkeypatch, step):
        """Return ``step(rank)`` for every rank, after the exchange delivered its inputs."""
        monkeypatch.setattr(parallel, "dist", SimpleNamespace(all_to_all_single=self.all_to_all_single))
        for delivering in (False, True):
            self.delivering = delivering
            results = []
            for rank in range(self.world_size):
                self.rank = rank
                results.append(step(rank))
        return results


def _layout(tensor):
    # Strides of size-0/1 dimensions carry no layout information.
    return [stride for size, stride in zip(tensor.shape, tensor.stride()) if size > 1]


def _assert_bitwise_equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert _layout(actual) == _layout(expected)
    assert torch.equal(actual.view(_BITS[actual.dtype]), expected.view(_BITS[expected.dtype]))


def _global_tensor(tokens, heads, head_dim, dtype, generator):
    tensor = torch.randn(tokens, heads, head_dim, generator=generator)
    flat = tensor.view(-1)
    specials = torch.tensor([-0.0, torch.inf, -torch.inf, 1e-40, 3.0e38])
    count = min(specials.numel(), flat.numel())
    flat[torch.randperm(flat.numel(), generator=generator)[:count]] = specials[:count]
    return tensor.to(dtype)


def _shards(tensor, split_sizes, rank):
    start = sum(split_sizes[:rank])
    return tensor[start : start + split_sizes[rank]].contiguous()


_SPLITS = {
    # Balanced with a remainder, a rank without tokens, and a skewed split.
    2: ([7, 6], [0, 5], [9, 2]),
    4: (balanced_split_sizes(19, 4), [2, 1, 0, 0], [6, 3, 5, 1]),
    8: (balanced_split_sizes(43, 8), balanced_split_sizes(5, 8), [7, 2, 5, 3, 1, 4, 6, 2]),
}


@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("split_index", [0, 1, 2])
@pytest.mark.parametrize(
    "dtypes,local_heads",
    [
        ((torch.bfloat16,) * 3, (3, 3, 3)),
        ((torch.bfloat16,) * 3, (6, 6, 6)),
        ((torch.float32,) * 3, (2, 1, 1)),
        ((torch.bfloat16,) * 3, (1, 3, 2)),
        ((torch.bfloat16, torch.float32, torch.bfloat16), (2, 2, 2)),
    ],
)
def test_qkv_exchange_matches_concatenating_reference(monkeypatch, world_size, split_index, dtypes, local_heads):
    split_sizes = _SPLITS[world_size][split_index]
    head_dim = 16
    generator = torch.Generator().manual_seed(1000 * world_size + 10 * split_index + local_heads[0])
    tensors = [
        _global_tensor(sum(split_sizes), world_size * heads, head_dim, dtype, generator)
        for dtype, heads in zip(dtypes, local_heads, strict=True)
    ]

    def run(function):
        exchange = _Exchange(world_size)
        outputs = exchange.run(
            monkeypatch,
            lambda rank: function(
                [_shards(tensor, split_sizes, rank) for tensor in tensors],
                split_sizes,
                Magi2ParallelGroup(None, world_size, rank),
            ),
        )
        return exchange.sent, outputs

    sent, outputs = run(parallel.scatter_heads_gather_seqlen)
    expected_sent, expected_outputs = run(_reference_scatter_heads_gather_seqlen)

    promoted = torch.float32 if torch.float32 in dtypes else dtypes[0]
    for rank in range(world_size):
        # The send buffer and its split sizes are byte-identical to the reference.
        _assert_bitwise_equal(sent[rank][0], expected_sent[rank][0])
        assert sent[rank][1] == expected_sent[rank][1]
        # So are the receive-buffer views FlashAttention reads.
        assert len(outputs[rank]) == len(expected_outputs[rank]) == len(tensors)
        for actual, expected, tensor, heads in zip(
            outputs[rank], expected_outputs[rank], tensors, local_heads, strict=True
        ):
            _assert_bitwise_equal(actual, expected)
            assert actual.dtype == promoted
            # Rank ``rank`` holds every token of its head shard.
            _assert_bitwise_equal(
                actual.contiguous(), tensor[:, rank * heads : (rank + 1) * heads].to(promoted).contiguous()
            )


@pytest.mark.cpu
def test_qkv_exchange_keeps_input_validation():
    group = Magi2ParallelGroup(None, 2, 0)
    q = torch.randn(3, 4, 8)
    with pytest.raises(ValueError, match="divide evenly"):
        parallel.scatter_heads_gather_seqlen([q, torch.randn(3, 3, 8)], [3, 3], group)
    with pytest.raises(ValueError, match=r"\[local_tokens, heads, dim\]"):
        parallel.scatter_heads_gather_seqlen([q, torch.randn(2, 4, 8)], [3, 3], group)
    with pytest.raises(ValueError, match="world size"):
        parallel.scatter_heads_gather_seqlen([q], [3, 3, 0], group)
    with pytest.raises(ValueError, match="same device"):
        parallel.scatter_heads_gather_seqlen([q, torch.randn(3, 4, 8, device="meta")], [3, 3], group)
    single = Magi2ParallelGroup(None, 1, 0)
    assert parallel.scatter_heads_gather_seqlen([q], [3], single)[0] is q
