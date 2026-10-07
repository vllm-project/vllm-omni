# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Ulysses exchanges must hand every rank the same bytes as the concatenating reference."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.models.magi2 import attention, modeling_magi2, parallel
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.configuration_magi2 import Magi2MoEConfig, Magi2PreviewConfig
from vllm_omni.diffusion.models.magi2.layers import ModalityDispatcher
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2Attention, Magi2TransformerLayer
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup, balanced_split_sizes

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.fixture(autouse=True)
def _deterministic_inductor(monkeypatch):
    # Reduction configs are otherwise benchmarked per compile, which can change the bits.
    # Dynamo resets ``deterministic`` after every traced frame; the config filter stays on.
    monkeypatch.setattr(torch._inductor.config, "deterministic", True)
    monkeypatch.setattr(torch._inductor.config.test_configs, "force_filter_reduction_configs", True)


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


def _reference_scatter_seqlen_gather_heads(tensor, split_sizes, group):
    """The output exchange that copied the received head shards to ``[S_rank, world*H, D]``."""
    if group.world_size == 1:
        return tensor
    local_tokens = split_sizes[group.rank]
    output = torch.empty(
        (group.world_size * local_tokens, tensor.shape[1], tensor.shape[2]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    parallel.dist.all_to_all_single(
        output,
        tensor,
        output_split_sizes=[local_tokens] * group.world_size,
        input_split_sizes=split_sizes,
        group=group.group,
    )
    return (
        output.view(group.world_size, local_tokens, tensor.shape[1], tensor.shape[2])
        .permute(1, 0, 2, 3)
        .reshape(local_tokens, group.world_size * tensor.shape[1], tensor.shape[2])
    )


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


def _heads_flattened(view):
    # With one head per rank the flattened view needs no copy and keeps the send-buffer strides.
    return view.flatten(1, 2).contiguous()


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


def _rank_inputs(tensors, split_sizes, rank):
    return [_shards(tensor, split_sizes, rank) for tensor in tensors]


def _run_reference(monkeypatch, tensors, split_sizes, world_size):
    exchange = _Exchange(world_size)
    outputs = exchange.run(
        monkeypatch,
        lambda rank: _reference_scatter_heads_gather_seqlen(
            _rank_inputs(tensors, split_sizes, rank), split_sizes, Magi2ParallelGroup(None, world_size, rank)
        ),
    )
    return exchange, outputs


@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("split_index", [0, 1, 2])
@pytest.mark.parametrize("local_heads", [(3, 3, 3), (6, 6, 6), (1, 3, 2)])
def test_packed_head_shards_are_sent_in_place(monkeypatch, world_size, split_index, local_heads):
    split_sizes = _SPLITS[world_size][split_index]
    head_dim = 16
    generator = torch.Generator().manual_seed(3000 * world_size + 10 * split_index + local_heads[1])
    tensors = [
        _global_tensor(sum(split_sizes), world_size * heads, head_dim, torch.bfloat16, generator)
        for heads in local_heads
    ]
    packed: dict[int, list[torch.Tensor]] = {}

    def step(rank):
        views = parallel.pack_ulysses_head_shards(_rank_inputs(tensors, split_sizes, rank), world_size)
        # Keep the first pass's views alive so its send-buffer address stays unique.
        packed.setdefault(rank, views)
        return parallel.scatter_heads_gather_seqlen(views, split_sizes, Magi2ParallelGroup(None, world_size, rank))

    exchange = _Exchange(world_size)
    outputs = exchange.run(monkeypatch, step)
    reference, expected = _run_reference(monkeypatch, tensors, split_sizes, world_size)

    for rank in range(world_size):
        views = packed[rank]
        for view, tensor, heads in zip(views, _rank_inputs(tensors, split_sizes, rank), local_heads, strict=True):
            assert view.shape == (split_sizes[rank], world_size, heads, head_dim)
            _assert_bitwise_equal(_heads_flattened(view), tensor)
        # The exchange sent the packed buffer itself, and its bytes are the reference's.
        if split_sizes[rank]:
            assert exchange.sent[rank][2] == views[0].data_ptr()
        _assert_bitwise_equal(exchange.sent[rank][0], reference.sent[rank][0])
        assert exchange.sent[rank][1] == reference.sent[rank][1]
        for actual, wanted in zip(outputs[rank], expected[rank], strict=True):
            _assert_bitwise_equal(actual, wanted)


def _separate_head_shards(tensors, world_size):
    return [tensor.view(tensor.shape[0], world_size, -1, tensor.shape[2]) for tensor in tensors]


def _reordered_pack(tensors, world_size):
    k, q, v = parallel.pack_ulysses_head_shards([tensors[1], tensors[0], tensors[2]], world_size)
    return [q, k, v]


def _pack_for_two_ranks(tensors, world_size):
    return parallel.pack_ulysses_head_shards(tensors, 2)


def _pack_with_autograd(tensors, world_size):
    with torch.enable_grad():
        leaves = [tensor.detach().clone().requires_grad_() for tensor in tensors]
        return parallel.pack_ulysses_head_shards(leaves, world_size)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "make_inputs", [_separate_head_shards, _reordered_pack, _pack_for_two_ranks, _pack_with_autograd]
)
def test_head_shards_that_do_not_tile_one_send_buffer_are_copied(monkeypatch, make_inputs):
    world_size, split_sizes, head_dim = 4, [5, 4, 4, 4], 8
    generator = torch.Generator().manual_seed(41)
    tensors = [_global_tensor(sum(split_sizes), 8, head_dim, torch.float32, generator) for _ in range(3)]
    inputs: dict[int, list[torch.Tensor]] = {}

    def step(rank):
        views = make_inputs(_rank_inputs(tensors, split_sizes, rank), world_size)
        inputs.setdefault(rank, views)
        with torch.enable_grad():
            return parallel.scatter_heads_gather_seqlen(views, split_sizes, Magi2ParallelGroup(None, world_size, rank))

    exchange = _Exchange(world_size)
    outputs = exchange.run(monkeypatch, step)
    reference, expected = _run_reference(monkeypatch, tensors, split_sizes, world_size)
    for rank in range(world_size):
        assert exchange.sent[rank][2] != inputs[rank][0].data_ptr()
        _assert_bitwise_equal(exchange.sent[rank][0].detach(), reference.sent[rank][0])
        for actual, wanted in zip(outputs[rank], expected[rank], strict=True):
            _assert_bitwise_equal(actual.detach(), wanted)


@pytest.mark.cpu
def test_single_rank_exchange_flattens_head_shards():
    generator = torch.Generator().manual_seed(43)
    tensors = [torch.randn(5, 4, 8, generator=generator) for _ in range(3)]
    views = parallel.pack_ulysses_head_shards(tensors, 2)
    flattened = parallel.scatter_heads_gather_seqlen(views, [5], Magi2ParallelGroup(None, 1, 0))
    for actual, tensor in zip(flattened, tensors, strict=True):
        _assert_bitwise_equal(actual, tensor)


@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("split_index", [0, 1, 2])
@pytest.mark.parametrize("dtype,local_heads", [(torch.bfloat16, 3), (torch.bfloat16, 6), (torch.float32, 1)])
def test_output_exchange_matches_reference_and_hands_back_a_view(
    monkeypatch, world_size, split_index, dtype, local_heads
):
    split_sizes = _SPLITS[world_size][split_index]
    head_dim = 16
    generator = torch.Generator().manual_seed(2000 * world_size + 10 * split_index + local_heads)
    # Every rank attended over all tokens with its own head shard.
    attended = [_global_tensor(sum(split_sizes), local_heads, head_dim, dtype, generator) for _ in range(world_size)]

    def run(function):
        return _Exchange(world_size).run(
            monkeypatch,
            lambda rank: function(attended[rank], split_sizes, Magi2ParallelGroup(None, world_size, rank)),
        )

    shards = run(parallel.scatter_seqlen_gather_head_shards)
    flattened = run(parallel.scatter_seqlen_gather_heads)
    expected = run(_reference_scatter_seqlen_gather_heads)
    for rank, tokens in enumerate(split_sizes):
        start = sum(split_sizes[:rank])
        assert shards[rank].shape == (tokens, world_size, local_heads, head_dim)
        # The [world, S_rank, H, D] receive buffer itself, read token-major.
        receive = torch.empty(world_size, tokens, local_heads, head_dim).permute(1, 0, 2, 3)
        assert _layout(shards[rank]) == _layout(receive)
        _assert_bitwise_equal(shards[rank].flatten(1, 2), expected[rank])
        _assert_bitwise_equal(flattened[rank], expected[rank])
        # Head shard s of this rank's tokens is what rank s computed for them.
        heads = torch.cat([output[start : start + tokens] for output in attended], dim=1)
        _assert_bitwise_equal(expected[rank].contiguous(), heads)


@pytest.mark.cpu
def test_single_rank_output_exchange_is_a_view():
    tensor = torch.randn(5, 3, 8)
    single = Magi2ParallelGroup(None, 1, 0)
    assert parallel.scatter_seqlen_gather_heads(tensor, [5], single) is tensor
    shards = parallel.scatter_seqlen_gather_head_shards(tensor, [5], single)
    assert shards.shape == (5, 1, 3, 8) and shards.data_ptr() == tensor.data_ptr()


@pytest.mark.cpu
@pytest.mark.parametrize("head_shard_output", [False, True])
def test_kernel_hands_back_head_shards_only_on_request(monkeypatch, head_shard_output):
    group = Magi2ParallelGroup(None, 2, 1)
    monkeypatch.setattr(attention, "get_magi2_ulysses_group", lambda: group)
    monkeypatch.setattr(attention, "scatter_heads_gather_seqlen", lambda tensors, split_sizes, group: list(tensors))
    attended = torch.randn(6, 2, 8)
    monkeypatch.setattr(attention, "packed_attention_with_sink", lambda *args, **kwargs: attended)
    shards, flat = torch.empty(3, 2, 2, 8), torch.empty(3, 4, 8)
    gather_shards = Mock(return_value=shards)
    gather_flat = Mock(return_value=flat)
    monkeypatch.setattr(attention, "scatter_seqlen_gather_head_shards", gather_shards)
    monkeypatch.setattr(attention, "scatter_seqlen_gather_heads", gather_flat)

    q = torch.randn(3, 4, 8)
    cu = torch.tensor([0, 6], dtype=torch.int32)
    output = attention.ulysses_packed_attention_with_sink(
        q, q, q, VarlenHandler(cu, cu, 6, 6), [3, 3], group=group, head_shard_output=head_shard_output
    )

    called, idle = (gather_shards, gather_flat) if head_shard_output else (gather_flat, gather_shards)
    assert output is (shards if head_shard_output else flat)
    called.assert_called_once()
    idle.assert_not_called()
    assert called.call_args.args[0] is attended and called.call_args.args[1:] == ([3, 3], group)


def _attention_config(params_dtype):
    # Eight query and KV heads divide across 2, 4 and 8 Ulysses ranks.
    return Magi2PreviewConfig(
        num_layers=1,
        hidden_size=64,
        head_dim=8,
        num_query_groups=8,
        multimodal_layers=(0,),
        params_dtype=params_dtype,
    )


def _initialized(module, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return module


def _modality_dispatcher(tokens, num_modality, generator, device="cpu"):
    mapping = torch.randint(0, num_modality, (tokens,), generator=generator)
    return ModalityDispatcher(mapping.to(device), num_modality)


def _head_shards(tensor, world_size):
    """``[T, world*H, D]`` as the ``[T, world, H, D]`` view of a ``[world, T, H, D]`` receive buffer."""
    tokens, heads, head_dim = tensor.shape
    receive = tensor.view(tokens, world_size, heads // world_size, head_dim).transpose(0, 1).contiguous()
    return receive.transpose(0, 1)


@pytest.mark.cpu
@pytest.mark.parametrize("musa", [False, True])
def test_attention_requests_head_shards_on_musa(monkeypatch, musa):
    monkeypatch.setattr(modeling_magi2, "current_omni_platform", SimpleNamespace(is_musa=lambda: musa))
    module = Magi2Attention(_attention_config(torch.float32), num_modality=1)
    assert module.packed_attention.attention.head_shard_output is musa


@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("num_modality", [1, 3])
@pytest.mark.parametrize("params_dtype", [torch.float32, torch.bfloat16])
def test_output_projection_of_head_shards_is_bitwise_unchanged(world_size, num_modality, params_dtype):
    module = _initialized(Magi2Attention(_attention_config(params_dtype), num_modality=num_modality), 7)
    generator = torch.Generator().manual_seed(17 * world_size + num_modality)
    tokens = 13
    dispatcher = _modality_dispatcher(tokens, num_modality, generator)
    attended = torch.randn(tokens, module.num_heads_q, module.head_dim, generator=generator).to(params_dtype)
    gates = torch.randn(tokens, module.num_heads_q, 1, generator=generator).to(params_dtype)

    with torch.inference_mode():
        expected = module.output(attended, gates, dispatcher)
        actual = module.output(_head_shards(attended, world_size), gates, dispatcher)
    _assert_bitwise_equal(actual, expected)


def _musa_available():
    return hasattr(torch, "musa") and torch.musa.is_available()


@pytest.mark.musa
@pytest.mark.parametrize("tokens,world_size", [(3702, 8), (3651, 4), (14, 8)])
@pytest.mark.parametrize("num_modality", [1, 3])
def test_real_musa_compiled_output_projection_of_head_shards_is_bitwise_unchanged(tokens, world_size, num_modality):
    if not _musa_available():
        pytest.skip("requires a MUSA device")
    config = Magi2PreviewConfig()
    module = _initialized(Magi2Attention(config, num_modality=num_modality), 23).to("musa")
    generator = torch.Generator().manual_seed(tokens + num_modality)
    dispatcher = _modality_dispatcher(tokens, num_modality, generator, device="musa")
    attended = torch.randn(tokens, module.num_heads_q, module.head_dim, generator=generator)
    attended = attended.to(device="musa", dtype=config.params_dtype)
    gates = torch.randn(tokens, module.num_heads_q, 1, generator=generator).to(device="musa", dtype=config.params_dtype)

    torch._dynamo.reset()
    # The production regions compile statically with emulated precision casts.
    output = torch.compile(module.output, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True})
    with torch.inference_mode():
        expected = output(attended, gates, dispatcher)
        actual = output(_head_shards(attended, world_size), gates, dispatcher)
    _assert_bitwise_equal(actual.cpu(), expected.cpu())


@pytest.mark.cpu
@pytest.mark.parametrize(
    "musa,world_size,expected",
    [(False, 4, 1), (True, 1, 1), (True, 4, 4), (True, 8, 8), (True, 3, 1)],
)
def test_projection_packs_head_shards_on_musa(monkeypatch, musa, world_size, expected):
    monkeypatch.setattr(modeling_magi2, "current_omni_platform", SimpleNamespace(is_musa=lambda: musa))
    monkeypatch.setattr(modeling_magi2, "get_magi2_ulysses_group", lambda: Magi2ParallelGroup(None, world_size, 0))
    module = Magi2Attention(_attention_config(torch.float32), num_modality=1)
    assert module.ulysses_head_shards == expected


def _projection_inputs(module, tokens, num_modality, generator, device="cpu"):
    hidden = torch.randn(tokens, module.config.hidden_size, generator=generator).to(module.config.params_dtype)
    rope = torch.randn(tokens, module.head_dim, generator=generator)
    dispatcher = _modality_dispatcher(tokens, num_modality, generator, device)
    return hidden.to(device), rope.to(device), dispatcher


@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("num_modality", [1, 3])
@pytest.mark.parametrize("params_dtype", [torch.float32, torch.bfloat16])
def test_projection_head_shards_are_bitwise_unchanged(world_size, num_modality, params_dtype):
    module = _initialized(Magi2Attention(_attention_config(params_dtype), num_modality=num_modality), 5)
    generator = torch.Generator().manual_seed(19 * world_size + num_modality)
    inputs = _projection_inputs(module, 11, num_modality, generator)

    with torch.inference_mode():
        module.ulysses_head_shards = 1
        expected = module.project(*inputs)
        module.ulysses_head_shards = world_size
        actual = module.project(*inputs)

    for view, tensor in zip(actual[:3], expected[:3], strict=True):
        assert view.shape == (tensor.shape[0], world_size, tensor.shape[1] // world_size, tensor.shape[2])
        _assert_bitwise_equal(_heads_flattened(view), tensor)
    _assert_bitwise_equal(actual[3], expected[3])
    assert parallel._packed_head_shards(list(actual[:3]), world_size) is not None


def _dense_layer(num_modality):
    # Production widths; layer 0 is multimodal and layer 1 single-modality, both dense.
    config = Magi2PreviewConfig(num_layers=2, multimodal_layers=(0,), moe=Magi2MoEConfig(layers=()))
    return _initialized(Magi2TransformerLayer(config, 0 if num_modality == 3 else 1), 29).to("musa")


@pytest.mark.musa
@pytest.mark.parametrize("tokens,world_size", [(3702, 8), (3651, 4), (14, 8)])
@pytest.mark.parametrize("num_modality", [1, 3])
def test_real_musa_compiled_attention_input_head_shards_are_bitwise_unchanged(tokens, world_size, num_modality):
    if not _musa_available():
        pytest.skip("requires a MUSA device")
    layer = _dense_layer(num_modality)
    generator = torch.Generator().manual_seed(tokens * num_modality)
    width = layer.config.mhc.num_streams * layer.config.hidden_size
    hidden = torch.randn(tokens, width, generator=generator).to(device="musa", dtype=layer.config.params_dtype)
    rope = torch.randn(tokens, layer.config.head_dim, generator=generator).to("musa")
    dispatcher = _modality_dispatcher(tokens, num_modality, generator, device="musa")

    def compiled_region(head_shards):
        torch._dynamo.reset()
        layer.attention.ulysses_head_shards = head_shards
        # The production regions compile statically with emulated precision casts.
        region = torch.compile(
            layer._attention_input, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
        )
        with torch.inference_mode():
            return region(hidden, rope, dispatcher)

    expected_streams, expected_logits, *expected_qkv, expected_gates = compiled_region(1)
    streams, logits, *packed_qkv, gates = compiled_region(world_size)
    for view, tensor in zip(packed_qkv, expected_qkv, strict=True):
        _assert_bitwise_equal(view.flatten(1, 2).cpu(), tensor.cpu())
    # The rest of the region, the mHC norm and logits included, is unchanged too.
    for actual, expected in zip(
        (streams, *logits, gates), (expected_streams, *expected_logits, expected_gates), strict=True
    ):
        _assert_bitwise_equal(actual.cpu(), expected.cpu())
    # The compiled region hands back views of one send buffer, sent without a copy.
    assert parallel._packed_head_shards(packed_qkv, world_size) is not None
