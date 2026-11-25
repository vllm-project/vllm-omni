# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LiDAR VAE neighborhood attention: gather path on Ascend, FlexAttention path on CUDA."""

from __future__ import annotations

import math

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.cosmos3 import lidar_vae as lv

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

# (height, width, (kernel_h, kernel_w), (dilation_h, dilation_w)); every case keeps
# axis >= kernel * dilation, which is the layout contract the attention asserts.
_CASES = [
    (1, 1, (1, 1), (1, 1)),
    (4, 8, (3, 1), (1, 1)),
    (12, 18, (5, 3), (2, 1)),
    (16, 16, (3, 3), (4, 4)),
    (10, 12, (3, 3), (3, 2)),
]
_DEVICE_CASES = [(8, 16, (3, 3), (1, 1)), (12, 18, (5, 3), (2, 1))]


def _fixed_window_reference(query, key, value, *, kernel, dilation, scale):
    """Dense attention restricted to the keys the FlexAttention predicate admits."""
    batch, height, width, heads, dim = query.shape
    sequence = height * width
    rows = torch.arange(sequence)
    allowed = lv._mask_predicate(height, width, kernel, dilation)(0, 0, rows[:, None], rows[None, :])
    flat = [t.permute(0, 3, 1, 2, 4).reshape(batch, heads, sequence, dim).double() for t in (query, key, value)]
    scores = torch.einsum("bhqd,bhkd->bhqk", flat[0], flat[1]) * float(scale)
    weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    out = torch.einsum("bhqk,bhkd->bhqd", weights, flat[2])
    return out.reshape(batch, heads, height, width, dim).permute(0, 2, 3, 1, 4)


def _tensors(height, width, *, heads=2, dim=8, batch=1, device="cpu"):
    shape = (batch, height, width, heads, dim)
    generator = torch.Generator().manual_seed(7)
    return [torch.randn(shape, generator=generator, dtype=torch.float32, device=device) for _ in range(3)]


@pytest.mark.cpu
@pytest.mark.parametrize(("height", "width", "kernel", "dilation"), _CASES)
def test_gather_matches_flex_predicate(height: int, width: int, kernel, dilation) -> None:
    """The Ascend gather attends to exactly the window the FlexAttention predicate admits."""
    scale = 1.0 / math.sqrt(8)
    query, key, value = _tensors(height, width)
    out = lv._neighborhood_attention_gather(query, key, value, kernel=kernel, dilation=dilation, scale=scale)
    expected = _fixed_window_reference(query, key, value, kernel=kernel, dilation=dilation, scale=scale)
    torch.testing.assert_close(out.double(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.cpu
def test_gather_rejects_axis_shorter_than_window() -> None:
    """An axis shorter than ``kernel * dilation`` fails at index construction."""
    query, key, value = _tensors(2, 2, dim=4)
    with pytest.raises(ValueError, match="axis length"):
        lv._neighborhood_attention_gather(query, key, value, kernel=(3, 3), dilation=(2, 2), scale=1.0)


@pytest.mark.cpu
def test_cpu_fixture_keeps_token_limit() -> None:
    """The 4096-token cap still guards the eager CPU FlexAttention fixtures."""
    query, key, value = _tensors(64, 65, heads=1, dim=8)
    with pytest.raises(RuntimeError, match="4096 tokens"):
        lv.neighborhood_attention_2d(query, key, value, kernel_size=3, dilation=1, scale=1.0)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize(("height", "width", "kernel", "dilation"), _DEVICE_CASES)
@pytest.mark.parametrize(("heads", "dim"), [(2, 8), (4, 16)])
def test_npu_neighborhood_matches_cpu_reference(
    height: int, width: int, kernel, dilation, heads: int, dim: int
) -> None:
    """``neighborhood_attention_2d`` on Ascend reproduces the dense predicate to FP32 tolerance."""
    scale = 1.0 / math.sqrt(dim)
    cpu_qkv = _tensors(height, width, heads=heads, dim=dim)
    npu_qkv = [t.to("npu") for t in cpu_qkv]
    out = lv.neighborhood_attention_2d(*npu_qkv, kernel_size=kernel, dilation=dilation, scale=scale)
    expected = _fixed_window_reference(*cpu_qkv, kernel=kernel, dilation=dilation, scale=scale)
    torch.testing.assert_close(out.double().cpu(), expected, atol=1e-5, rtol=1e-5)


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("height", "width", "kernel", "dilation"), _DEVICE_CASES)
def test_cuda_flex_neighborhood_matches_reference(height: int, width: int, kernel, dilation) -> None:
    """Regression guard: the CUDA FlexAttention path still matches the dense predicate."""
    scale = 1.0 / math.sqrt(8)
    cpu_qkv = _tensors(height, width)
    cuda_qkv = [t.cuda() for t in cpu_qkv]
    out = lv.neighborhood_attention_2d(*cuda_qkv, kernel_size=kernel, dilation=dilation, scale=scale)
    expected = _fixed_window_reference(*cpu_qkv, kernel=kernel, dilation=dilation, scale=scale)
    torch.testing.assert_close(out.double().cpu(), expected, atol=1e-5, rtol=1e-5)
