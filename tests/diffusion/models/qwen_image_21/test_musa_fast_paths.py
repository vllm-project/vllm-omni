# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MUSA paths of the Qwen-Image-2.1 transformer fast paths."""

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import exact_bf16_pointwise as pointwise
from vllm_omni.diffusion.models.qwen_image_21 import (
    qwen_image_21_transformer as transformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _require_musa() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")


def _freqs(sequence: int, generator: torch.Generator) -> torch.Tensor:
    angles = torch.randn(sequence, 64, generator=generator) * 3
    return torch.polar(torch.ones_like(angles), angles)


def _qkv_split(sequence: int, heads: int, generator: torch.Generator, device: str) -> tuple[torch.Tensor, ...]:
    """Q, K and V as the attention block produces them: views into one fused projection."""
    qkv = torch.randn(1, sequence, 3 * heads * 128, generator=generator).to(device=device, dtype=torch.bfloat16)
    return tuple(t.unflatten(-1, (heads, 128)) for t in qkv.split(heads * 128, dim=-1))


def _reference_rotation(monkeypatch, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    with monkeypatch.context() as patch:
        patch.setattr(transformer, "rotary", lambda *args: None)
        return transformer._apply_qwen_image21_rotary_emb_native(x, freqs)


@pytest.mark.cpu
@torch.no_grad()
def test_cpu_tensors_keep_the_reference_rotation():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 8, 2, 128, generator=generator).to(torch.bfloat16)
    freqs = _freqs(8, generator)
    assert pointwise.rotary(x, freqs) is None


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_rope_kernel_matches_the_native_rotation(monkeypatch):
    _require_musa()
    generator = torch.Generator().manual_seed(1)
    query, _, _ = _qkv_split(512, 4, generator, "musa")
    freqs = _freqs(512, generator).to("musa")
    got = pointwise.rotary(query, freqs)
    assert got is not None
    torch.testing.assert_close(got, _reference_rotation(monkeypatch, query, freqs), rtol=0, atol=0)


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_rope_runs_inside_a_compiled_region():
    _require_musa()
    generator = torch.Generator().manual_seed(2)
    query, _, _ = _qkv_split(512, 4, generator, "musa")
    freqs = _freqs(512, generator).to("musa")
    compiled = torch.compile(transformer._apply_qwen_image21_rotary_emb_native, fullgraph=True)
    # Inductor's own fusion of the native rotation rounds differently, so only the
    # kernel reproduces the eager kernel output bit for bit.
    torch.testing.assert_close(compiled(query, freqs), pointwise.rotary(query, freqs), rtol=0, atol=0)


class _ScaledAttention(nn.Module):
    def forward(self, values, *args, **kwargs):
        return values * 0.37


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_shared_modulation_matches_per_block_modulation():
    _require_musa()
    block = transformer.QwenImage21TransformerBlock.__new__(transformer.QwenImage21TransformerBlock)
    nn.Module.__init__(block)
    block.img_norm1 = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
    block.img_norm2 = nn.LayerNorm(128, elementwise_affine=False, eps=1e-6)
    block.attn = _ScaledAttention()
    block.img_mlp = nn.SiLU()
    block.to("musa")
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(2, 17, 128, generator=generator).to("musa", torch.bfloat16)
    modulation = torch.randn(2, 512, generator=generator).to("musa", torch.bfloat16)
    select = transformer._select_modulation_rows
    prepared = tuple(
        1 + select(x, None) if i % 2 == 0 else select(x, None).tanh() for i, x in enumerate(modulation.chunk(4, dim=-1))
    )
    expected = block(values, modulation, None, target_token_mask=None)
    actual = block(values, modulation, None, target_token_mask=None, prepared_modulation=prepared)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
