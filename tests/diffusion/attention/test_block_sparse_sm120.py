# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise logical block128 routing through the real FlashInfer SM120 kernel."""

import math
import sys

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "RTX5090"})]


@pytest.fixture(autouse=True)
def sm120_runtime(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires an SM120/SM121 GPU")
    pytest.importorskip("flashinfer.cute_dsl.sparse.bsa_attn_sm120")
    # The framework must work without the experimental FastVideo SM120 port,
    # including when a shell still carries its former opt-in variable.
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.block_sparse_attn_sm120", None)
    monkeypatch.setenv("FASTVIDEO_VSA_SM120", "1")
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


def _reference(q, k, v, block_map, sizes, logical_blocks):
    """Independent token-level attention; never decomposes the mask into 64s."""
    length = logical_blocks * 128
    q, k, v = [x[:, :length].float().transpose(1, 2) for x in (q, k, v)]
    valid_keys = (torch.arange(128, device=q.device)[None] < sizes[:logical_blocks, None]).flatten()
    mask = block_map[..., :logical_blocks, :logical_blocks].repeat_interleave(128, -2).repeat_interleave(128, -1)
    mask = mask & valid_keys.view(1, 1, 1, -1)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1).nan_to_num()
    return (probabilities @ v).transpose(1, 2)


@pytest.mark.parametrize(
    "batch,heads,sizes_list,empty_rows,transport,scale",
    [
        (1, 14, [128] * 8, False, False, 1.0),
        (1, 14, [128, 17, 64, 65, 127, 0, 93, 128], False, False, 1.0),
        (1, 14, [128, 17, 64, 65, 127, 0, 93, 128], True, False, 1.0),
        (2, 7, [128, 17, 64, 65, 0], False, True, 1.0),
        (1, 56, [128, 17, 128], False, True, 1.0),
        (1, 14, [128, 17, 64, 65, 127], True, True, 4.0),
        (1, 1, [0, 0, 0], True, True, 1.0),
    ],
    ids=["dense", "partial-kv", "empty-query", "batch2-head7", "head56", "large-logits", "empty-kv"],
)
def test_flashinfer_sm120_preserves_token_attention(batch, heads, sizes_list, empty_rows, transport, scale):
    torch.manual_seed(8391)
    blocks = len(sizes_list)
    q, k, v = [torch.randn(batch, blocks * 128, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q *= scale
    k *= scale
    sizes = torch.tensor(sizes_list, device="cuda", dtype=torch.int32)
    valid = (torch.arange(128, device="cuda")[None] < sizes[:, None]).flatten()
    v[:, ~valid] = 100  # Invalid keys must not contribute, even with large values.
    block_map = torch.rand(batch, heads, blocks, blocks, device="cuda") < 0.35
    block_map[..., 0] = True
    block_map[..., 0, :] = True
    if empty_rows:
        block_map[..., -1, :] = False
    expected = _reference(q, k, v, block_map, sizes, blocks)
    if transport:
        q, k, v = [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 128), value=50) for x in (q, k, v)]
        block_map = torch.nn.functional.pad(block_map, (0, 1, 0, 1), value=True)
        sizes = torch.nn.functional.pad(sizes, (0, 1), value=0)
        block_map[..., -1, :] = False
        expected = torch.nn.functional.pad(expected, (0, 0, 0, 0, 0, 128))

    actual = block_sparse_attn_bshd(q, k, v, block_map, sizes, 128**-0.5, provider="flashinfer", block_size=128)
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.01)
    assert ((actual.float() - expected).norm() / expected.norm().clamp_min(1e-10)).item() < 0.005
    assert actual.shape == q.shape and actual.dtype == q.dtype
    if transport:
        assert torch.count_nonzero(actual[:, blocks * 128 :]).item() == 0


def test_flashinfer_sm120_cuda_graph_reloads_queries_routing_and_partial_sizes():
    torch.manual_seed(431)
    q, k, v = [torch.randn(1, 640, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    sizes = torch.tensor([128, 17, 64, 65, 127], device="cuda", dtype=torch.int32)
    block_map = torch.ones(1, 7, 5, 5, device="cuda", dtype=torch.bool)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            block_sparse_attn_bshd(q, k, v, block_map, sizes, 128**-0.5, provider="flashinfer", block_size=128)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = block_sparse_attn_bshd(q, k, v, block_map, sizes, 128**-0.5, provider="flashinfer", block_size=128)
    graph.replay()
    before = actual.clone()

    q.copy_(torch.randn_like(q))
    k.copy_(torch.randn_like(k))
    v.copy_(torch.randn_like(v))
    sizes.copy_(torch.tensor([17, 128, 0, 64, 65], device="cuda", dtype=torch.int32))
    block_map.copy_(torch.rand_like(block_map, dtype=torch.float32) < 0.4)
    block_map[..., 0] = True
    block_map[..., -1, :] = False
    graph.replay()
    expected = _reference(q, k, v, block_map, sizes, 5)
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.01)
    assert not torch.equal(actual, before)
