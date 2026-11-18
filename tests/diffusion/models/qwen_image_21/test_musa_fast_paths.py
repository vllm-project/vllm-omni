# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MUSA paths of the Qwen-Image-2.1 transformer fast paths."""

from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import exact_bf16_rope_musa as musa_rope
from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _require_musa() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")


def _forbidden(name: str) -> Mock:
    return Mock(side_effect=AssertionError(f"{name} must not run here"))


def _freqs(sequence: int, generator: torch.Generator) -> torch.Tensor:
    angles = torch.randn(sequence, 64, generator=generator) * 3
    return torch.polar(torch.ones_like(angles), angles)


def _qkv_split(sequence: int, heads: int, generator: torch.Generator, device: str) -> tuple[torch.Tensor, ...]:
    """Q, K and V as the attention block produces them: views into one fused projection."""
    qkv = torch.randn(1, sequence, 3 * heads * 128, generator=generator).to(device=device, dtype=torch.bfloat16)
    return tuple(t.unflatten(-1, (heads, 128)) for t in qkv.split(heads * 128, dim=-1))


def _attention_without_init() -> transformer.QwenImage21Attention:
    # The rotation dispatch reads no module state.
    return transformer.QwenImage21Attention.__new__(transformer.QwenImage21Attention)


@pytest.mark.cpu
@torch.no_grad()
def test_cpu_tensors_keep_the_reference_rotation():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 8, 2, 128, generator=generator).to(torch.bfloat16)
    freqs = _freqs(8, generator)
    assert musa_rope.musa_rotary(x, freqs) is None
    native = transformer._apply_qwen_image21_rotary_emb_native(x, freqs)
    torch.testing.assert_close(_attention_without_init()._apply_rotary_emb(x, freqs), native, rtol=0, atol=0)
    # Inputs the MUSA kernel declines take the native rotation.
    torch.testing.assert_close(transformer._apply_qwen_image21_rotary_emb_musa(x, freqs), native, rtol=0, atol=0)


@pytest.mark.cpu
def test_musa_tensors_take_the_musa_rotation(monkeypatch):
    rotated = object()
    monkeypatch.setattr(transformer, "_apply_qwen_image21_rotary_emb_musa", lambda x, freqs: rotated)
    monkeypatch.setattr(transformer, "_apply_qwen_image21_rotary_emb_native", _forbidden("the native rotation"))
    x = Mock(spec=torch.Tensor, device=Mock(spec=torch.device, type="musa"))
    assert _attention_without_init()._apply_rotary_emb(x, None) is rotated


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_rope_kernel_matches_the_native_rotation(monkeypatch):
    _require_musa()
    generator = torch.Generator().manual_seed(1)
    query, _, _ = _qkv_split(512, 4, generator, "musa")
    freqs = _freqs(512, generator).to("musa")
    got = musa_rope.musa_rotary(query, freqs)
    assert got is not None
    expected = transformer._apply_qwen_image21_rotary_emb_native(query, freqs)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    monkeypatch.setattr(transformer, "_apply_qwen_image21_rotary_emb_native", _forbidden("the native rotation"))
    torch.testing.assert_close(_attention_without_init()._apply_rotary_emb(query, freqs), got, rtol=0, atol=0)


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_rope_runs_inside_a_compiled_region(monkeypatch):
    _require_musa()
    generator = torch.Generator().manual_seed(2)
    query, _, _ = _qkv_split(512, 4, generator, "musa")
    freqs = _freqs(512, generator).to("musa")
    eager = musa_rope.musa_rotary(query, freqs)
    compiled = torch.compile(musa_rope.musa_rotary, fullgraph=True)(query, freqs)
    assert compiled is not None
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
    # Tracing the native fallback would fail, so the compiled helper took the kernel.
    monkeypatch.setattr(transformer, "_apply_qwen_image21_rotary_emb_native", _forbidden("the native rotation"))
    helper = torch.compile(transformer._apply_qwen_image21_rotary_emb_musa, fullgraph=True)
    torch.testing.assert_close(helper(query, freqs), eager, rtol=0, atol=0)
