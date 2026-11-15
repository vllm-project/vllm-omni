# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU DiT fused body matches the CUDA fused schedule on the shared math."""

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.dit_fused import (
    blocks_forward_chunk_fused,
    supports_fused_body,
)
from vllm_omni.platforms.npu.models.minicpmo_fused import (
    _attach_history,
    _causal_conv,
    blocks_forward_chunk_fused_npu,
    packed_conv_columns,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.act = nn.GELU(approximate="tanh")
        self.norm = nn.Identity()


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner)
        self.to_k = nn.Linear(dim, inner)
        self.to_v = nn.Linear(dim, inner)
        self.proj = nn.Linear(inner, dim)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = _Attention(dim, 4, dim // 4)
        self.mlp = _MLP(dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        conv1 = nn.Conv1d(dim, dim, 3)
        conv2 = nn.Conv1d(dim, dim, 3)
        self.conv = nn.Module()
        self.conv.block = nn.ModuleList(
            [nn.Identity(), conv1, nn.Identity(), nn.LayerNorm(dim, eps=1e-6), nn.Mish(), nn.Identity(), conv2]
        )


class _Estimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim = 32
        self.in_proj = nn.Linear(16, dim)
        self.blocks = nn.ModuleList([_Block(dim), _Block(dim)])
        self.final_layer = nn.Module()
        self.final_layer.norm_final = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.final_layer.linear = nn.Linear(dim, 8)
        self.final_layer.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))


def _run(body, estimator: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    rows, frames, cached = 4, 6, 5
    attn = estimator.blocks[0].attn
    channels = estimator.blocks[0].conv.block[1].in_channels
    width = int(estimator.blocks[0].conv.block[1].kernel_size[0]) - 1
    depth = len(estimator.blocks)
    estimator_input = torch.randn(rows, 16, frames)
    time_embedding = torch.randn(rows, 1, channels)
    mask = torch.rand(rows, frames, cached + frames) > 0.3
    mask[..., 0] = True
    cnn = [torch.randn(rows, 2 * channels, width) for _ in range(depth)]
    att = [torch.randn(rows, attn.num_heads, cached, 2 * attn.head_dim) for _ in range(depth)]
    cnn_out = torch.zeros(depth, rows, 2 * channels, width)
    att_out = torch.zeros(depth, rows, attn.num_heads, cached + frames, 2 * attn.head_dim)
    with torch.no_grad():
        out = body(estimator, estimator_input, time_embedding, mask, cnn, att, cnn_out, att_out, [6, 4])
    return out, cnn_out, att_out


def test_npu_fused_body_matches_cuda_fused_body() -> None:
    torch.manual_seed(0)
    estimator = _Estimator().eval()
    assert supports_fused_body(estimator)
    expected = _run(blocks_forward_chunk_fused, estimator)
    actual = _run(blocks_forward_chunk_fused_npu, estimator)
    for want, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_npu_fused_body_accepts_inference_parameters():
    torch.manual_seed(0)
    reference = _Estimator().eval()
    expected = _run(blocks_forward_chunk_fused, reference)
    with torch.inference_mode():
        estimator = _Estimator().eval()
        estimator.load_state_dict(reference.state_dict())
        actual = _run(blocks_forward_chunk_fused_npu, estimator)
    for want, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("inference", [False, True])
def test_packed_conv_columns_tracks_in_place_and_replaced_weights(inference):
    with torch.inference_mode(inference), torch.no_grad():
        conv = nn.Conv1d(8, 12, 3)
        columns = conv.weight.permute(0, 2, 1).reshape(12, 24)
        packed = packed_conv_columns(conv)
        torch.testing.assert_close(packed, columns)
        conv.weight.add_(0.25)
        assert packed_conv_columns(conv) is packed
        torch.testing.assert_close(packed, conv.weight.permute(0, 2, 1).reshape(12, 24))
        conv.weight = nn.Parameter(torch.randn_like(conv.weight))
        replacement = packed_conv_columns(conv)
        assert replacement is not packed
        torch.testing.assert_close(replacement, conv.weight.permute(0, 2, 1).reshape(12, 24))


def test_causal_conv_matches_conv1d() -> None:
    torch.manual_seed(2)
    rows, frames, width, channels = 2, 5, 2, 8
    conv = nn.Conv1d(channels, channels, width + 1).eval()
    history = torch.randn(rows, width + frames, channels)
    with torch.no_grad():
        got = _causal_conv(conv, history, frames)
        want = conv(history.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_attach_history_matches_buffer_fill() -> None:
    current = torch.randn(2, 5, 4)
    cache = torch.randn(2, 2, 4)  # already frames-major
    attached = _attach_history(cache, current, 2)
    torch.testing.assert_close(attached[:, :2], cache)
    torch.testing.assert_close(attached[:, 2:], current)
    torch.testing.assert_close(_attach_history(None, current, 2)[:, :2], torch.zeros(2, 2, 4))
    torch.testing.assert_close(_attach_history(None, current, 2)[:, 2:], current)


def test_npu_fused_body_rejects_slot_pool() -> None:
    estimator = _Estimator().eval()
    slots = (torch.zeros(2, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long))
    try:
        blocks_forward_chunk_fused_npu(
            estimator,
            torch.zeros(2, 16, 1),
            torch.zeros(2, 1, 32),
            None,
            [None, None],
            [None, None],
            torch.zeros(2, 2, 64, 2),
            torch.zeros(2, 2, 4, 1, 16),
            [1],
            slots=slots,
        )
    except RuntimeError as exc:
        assert "slot pool" in str(exc)
    else:
        raise AssertionError("slot pool should be rejected")
