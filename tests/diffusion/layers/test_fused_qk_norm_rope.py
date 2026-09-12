# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton required"),
]

_HEAD_DIM = 128
_ROTARY_DIM = 96
_EPS = 1e-5


def _reference(q, k, q_weight, k_weight, rope_table):
    q = F.rms_norm(q, (_HEAD_DIM,), q_weight, _EPS)
    k = F.rms_norm(k, (_HEAD_DIM,), k_weight, _EPS)
    half = _ROTARY_DIM // 2
    cos = rope_table[..., :half].unsqueeze(1)
    sin = rope_table[..., half:].unsqueeze(1)

    def apply(x):
        first = x[..., :half]
        second = x[..., half:_ROTARY_DIM]
        return torch.cat(
            (
                first * cos - second * sin,
                second * cos + first * sin,
                x[..., _ROTARY_DIM:],
            ),
            dim=-1,
        )

    return apply(q), apply(k)


def _inputs(seq_len):
    torch.manual_seed(17)
    heads = 14
    qkv = torch.randn(
        seq_len,
        heads * _HEAD_DIM * 3,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q = qkv[:, : heads * _HEAD_DIM].view(seq_len, heads, _HEAD_DIM)
    k = qkv[:, heads * _HEAD_DIM : 2 * heads * _HEAD_DIM].view(
        seq_len,
        heads,
        _HEAD_DIM,
    )
    q_weight = torch.randn(_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    freqs = torch.randn(seq_len, _ROTARY_DIM // 2, device="cuda")
    rope_table = torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1).to(torch.bfloat16)
    return q, k, q_weight, k_weight, rope_table


@pytest.mark.parametrize("seq_len", [1, 257, 1024])
def test_fused_qk_norm_rope_matches_bf16_reference(seq_len):
    from vllm_omni.diffusion.layers.ops import fused_qk_norm_rope

    q, k, q_weight, k_weight, rope_table = _inputs(seq_len)

    expected_q, expected_k = _reference(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
    )
    actual_q, actual_k = fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        _EPS,
    )

    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)


def test_fused_qk_norm_rope_fullgraph_compile():
    from vllm_omni.diffusion.layers.ops import fused_qk_norm_rope

    inputs = _inputs(257)
    original = tuple(x.clone() for x in inputs)
    compiled = torch.compile(fused_qk_norm_rope, fullgraph=True)
    for actual, expected in zip(compiled(*inputs, _EPS), fused_qk_norm_rope(*inputs, _EPS)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(inputs, original):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fused_qk_norm_rope_cuda_graph_replay():
    from vllm_omni.diffusion.layers.ops import fused_qk_norm_rope

    inputs = _inputs(257)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Finish kernel compilation and allocator warmup before capture.
        for _ in range(3):
            fused_qk_norm_rope(*inputs, _EPS)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_qk_norm_rope(*inputs, _EPS)

    # Replay must read updated buffers rather than reuse capture-time results.
    inputs[0].add_(0.25)
    inputs[1].mul_(0.75)
    expected = fused_qk_norm_rope(*inputs, _EPS)
    graph.replay()
    for actual, reference in zip(captured, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
