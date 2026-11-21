# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tile64 key pairing and zero-pad correction for the Ascend block-sparse operator, on CPU.

The operator is replaced by a stand-in that follows its documented contract:
query blocks of 64 rows, key blocks of 128 rows, no per-key mask, and the
log-sum-exp of every query row returned next to the output.
"""

import sys
import types

import pytest
import torch

from vllm_omni.diffusion.attention.ops import block_sparse
from vllm_omni.diffusion.attention.ops.npu_tile64_block_sparse import (
    npu_tile64_block_sparse_attention,
    pair_key_tiles,
    require_npu_tile64_sparse,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

SCALE = 128**-0.5


def _fake_operator(calls):
    def npu_block_sparse_attention(query, key, value, block_sparse_mask, block_shape, **kwargs):
        calls.append({"block_shape": list(block_shape), "mask_shape": tuple(block_sparse_mask.shape), **kwargs})
        q_block, k_block = block_shape
        assert k_block % 128 == 0, "the operator only accepts key blocks in multiples of 128"
        assert block_sparse_mask.dtype == torch.int8
        assert block_sparse_mask.shape[-2:] == (-(-query.shape[2] // q_block), -(-key.shape[2] // k_block))
        token_mask = block_sparse_mask.bool().repeat_interleave(q_block, dim=-2).repeat_interleave(k_block, dim=-1)
        logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * kwargs["scale_value"]
        logits = logits.masked_fill(~token_mask, float("-inf"))
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        output = torch.matmul(torch.softmax(logits, dim=-1).nan_to_num(0.0), value.float())
        return output.to(query.dtype), lse

    return npu_block_sparse_attention


@pytest.fixture
def operator_calls(monkeypatch):
    calls: list[dict] = []
    module = types.ModuleType("torch_npu")
    module.npu_block_sparse_attention = _fake_operator(calls)
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return calls


def _tiled_inputs(sizes, heads=2, seed=0):
    """BSHD tensors padded to whole 64-token tiles, zero past each tile's valid rows."""
    generator = torch.Generator().manual_seed(seed)
    tiles = len(sizes)
    valid = torch.arange(64)[None, :] < torch.tensor(sizes)[:, None]
    tensors = []
    for _ in range(3):
        x = torch.randn(1, tiles * 64, heads, 128, generator=generator)
        tensors.append((x * valid.reshape(1, -1, 1, 1)).to(torch.bfloat16))
    return (*tensors, valid.reshape(-1))


def _reference(query, key, value, block_map, valid):
    """Exact attention over the valid keys of every selected tile and its pair partner."""
    tiles = block_map.shape[-1]
    partner = torch.arange(tiles) ^ 1
    widened = block_map.clone()
    in_range = partner < tiles
    widened[..., in_range] |= block_map[..., partner[in_range]]
    token_mask = widened.repeat_interleave(64, dim=-2).repeat_interleave(64, dim=-1) & valid[None, None, None, :]
    q, k, v = (t.float().transpose(1, 2) for t in (query, key, value))
    logits = torch.matmul(q, k.transpose(-1, -2)) * SCALE
    weights = torch.softmax(logits.masked_fill(~token_mask, float("-inf")), dim=-1).nan_to_num(0.0)
    return torch.matmul(weights, v).transpose(1, 2)


@pytest.mark.parametrize(
    "sizes",
    [[64, 64, 64, 64], [64, 17, 64, 1], [5, 64, 6, 64, 32], [64, 0, 64]],
    ids=["full_even", "ragged_even", "ragged_odd", "empty_tile_odd"],
)
def test_tile64_matches_attention_over_the_valid_keys_of_paired_tiles(operator_calls, sizes):
    query, key, value, valid = _tiled_inputs(sizes)
    tiles = len(sizes)
    generator = torch.Generator().manual_seed(1)
    block_map = torch.rand(1, 2, tiles, tiles, generator=generator) < 0.4
    block_map[..., 0] = True  # every query tile keeps at least the first key tile

    output = npu_tile64_block_sparse_attention(
        query, key, value, block_map, torch.tensor(sizes, dtype=torch.int32), SCALE
    )

    assert output.shape == query.shape and output.dtype == torch.bfloat16
    torch.testing.assert_close(output.float(), _reference(query, key, value, block_map, valid), atol=2e-2, rtol=2e-2)
    (call,) = operator_calls
    assert call["block_shape"] == [64, 128]
    assert call["mask_shape"] == (1, 2, tiles, (tiles + 1) // 2)
    assert call["q_input_layout"] == call["kv_input_layout"] == "BNSD"
    assert call["inner_precise"] == 0 and call["softmax_lse_flag"] == 1


def test_tile64_query_tile_without_a_valid_key_returns_zeros(operator_calls):
    sizes = [64, 64, 0, 0]
    query, key, value, _ = _tiled_inputs(sizes)
    block_map = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
    block_map[..., 0, 0] = True  # tile 0 attends real keys
    block_map[..., 1, 2] = True  # tile 1 attends only an all-padding key block
    # tiles 2 and 3 select nothing at all

    output = npu_tile64_block_sparse_attention(
        query, key, value, block_map, torch.tensor(sizes, dtype=torch.int32), SCALE
    )

    assert torch.isfinite(output.float()).all()
    assert output[:, :64].abs().sum() > 0
    assert output[:, 64:].abs().sum() == 0


def test_pair_key_tiles_folds_an_odd_tile_count_onto_whole_key_blocks():
    block_map = torch.tensor([[[[True, False, False, False, True], [False, False, True, False, False]]]])
    pair_map, pair_valid = pair_key_tiles(block_map, torch.tensor([64, 3, 0, 64, 10], dtype=torch.int32))

    assert pair_map.tolist() == [[[[True, False, True], [False, True, False]]]]
    assert pair_valid.tolist() == [67, 64, 10]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda q, k, v, m, s: (q.float(), k.float(), v.float(), m, s), "BF16"),
        (lambda q, k, v, m, s: (q[:, :-1], k[:, :-1], v[:, :-1], m, s), "complete 64-token tiles"),
        (lambda q, k, v, m, s: (q, k, v, m[..., :1], s), "block_map"),
        (lambda q, k, v, m, s: (q, k, v, m, s[:1]), "valid-token count"),
    ],
    ids=["dtype", "partial_tile", "map_shape", "sizes_shape"],
)
def test_tile64_rejects_inputs_the_operator_cannot_take(operator_calls, mutate, match):
    query, key, value, _ = _tiled_inputs([64, 64])
    args = mutate(
        query, key, value, torch.ones(1, 2, 2, 2, dtype=torch.bool), torch.tensor([64, 64], dtype=torch.int32)
    )
    with pytest.raises(ValueError, match=match):
        npu_tile64_block_sparse_attention(*args, SCALE)
    assert not operator_calls


def test_require_reports_a_torch_npu_without_the_operator(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    with pytest.raises(ImportError, match="npu_block_sparse_attention"):
        require_npu_tile64_sparse()


def _resolve(provider="auto", precision="bf16", dtype=torch.bfloat16, head_size=128, block_size=64):
    device = types.SimpleNamespace(type="npu", index=0)
    return block_sparse.resolve_block_sparse_provider.__wrapped__(
        provider, precision, device, dtype, head_size, block_size=block_size
    )


def test_resolver_selects_the_npu_provider_for_tile64(operator_calls):
    assert _resolve() == "npu"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"provider": "fastvideo"}, "CUDA-only"),
        ({"provider": "triton"}, "CUDA-only"),
        ({"block_size": 128}, "64-token tiles only"),
        ({"precision": "sage"}, "BF16"),
        ({"dtype": torch.float16}, "BF16"),
        ({"head_size": 64}, "BF16"),
    ],
)
def test_resolver_refuses_what_the_npu_provider_cannot_run(operator_calls, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _resolve(**kwargs)


def test_dispatch_routes_the_npu_provider(operator_calls):
    query, key, value, _ = _tiled_inputs([64, 64])
    block_map = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    sizes = torch.tensor([64, 64], dtype=torch.int32)

    output = block_sparse.block_sparse_attn_bshd(query, key, value, block_map, sizes, SCALE, provider="npu")

    assert output.shape == query.shape and len(operator_calls) == 1
    with pytest.raises(ValueError, match="64-token tiles"):
        block_sparse.block_sparse_attn_bshd(query, key, value, block_map, sizes, SCALE, provider="npu", block_size=128)
