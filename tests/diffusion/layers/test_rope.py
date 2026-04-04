# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Equivalence tests for unified RoPE utility functions.

Proves that the shared functions in vllm_omni.diffusion.layers.rope produce
bit-exact results compared to the per-model implementations they replace.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch


def _load_rope_module():
    """Load rope.py directly, bypassing vllm_omni.__init__ dependencies."""
    rope_path = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        "vllm_omni",
        "diffusion",
        "layers",
        "rope.py",
    )
    rope_path = os.path.abspath(rope_path)

    # Mock only the imports rope.py needs that pull in heavy deps
    mocks = {
        "vllm_omni": types.ModuleType("vllm_omni"),
        "vllm_omni.diffusion": types.ModuleType("vllm_omni.diffusion"),
        "vllm_omni.diffusion.layers": types.ModuleType("vllm_omni.diffusion.layers"),
        "vllm_omni.diffusion.layers.custom_op": MagicMock(),
        "vllm.logger": MagicMock(init_logger=lambda name: MagicMock()),
    }
    with patch.dict(sys.modules, mocks):
        spec = importlib.util.spec_from_file_location("vllm_omni.diffusion.layers.rope", rope_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


_rope = _load_rope_module()
rotate_half = _rope.rotate_half
apply_rotary_pos_emb = _rope.apply_rotary_pos_emb

# ---------------------------------------------------------------------------
# Reference implementations (copied from model files for comparison)
# ---------------------------------------------------------------------------


def _ref_rotate_half_nextstep(x: torch.Tensor) -> torch.Tensor:
    """From nextstep_1_1/modeling_nextstep_llama.py"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_apply_rotary_pos_emb_nextstep(q, k, cos, sin, unsqueeze_dim=1):
    """From nextstep_1_1/modeling_nextstep_llama.py"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_ref_rotate_half_nextstep(q) * sin)
    k_embed = (k * cos) + (_ref_rotate_half_nextstep(k) * sin)
    return q_embed, k_embed


def _ref_apply_rotary_pos_emb_mla(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, mla=False):
    """From hunyuan_image_3/hunyuan_image_3_transformer.py"""
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if mla:
        b, h, s, d = q.shape
        q = q.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        b, h, s, d = k.shape
        k = k.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (_ref_rotate_half_nextstep(q) * sin)
    k_embed = (k * cos) + (_ref_rotate_half_nextstep(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _random_cos_sin(batch: int, seq: int, dim: int, dtype: torch.dtype = torch.float32):
    """Generate random cos/sin tensors shaped [batch, seq, dim]."""
    cos = torch.randn(batch, seq, dim, dtype=dtype)
    sin = torch.randn(batch, seq, dim, dtype=dtype)
    return cos, sin


# ---------------------------------------------------------------------------
# Tests: rotate_half equivalence
# ---------------------------------------------------------------------------


class TestRotateHalf:
    @pytest.mark.parametrize("shape", [(2, 8, 4, 32), (1, 16, 2, 64)])
    def test_matches_reference(self, shape: tuple[int, ...]) -> None:
        x = torch.randn(shape)
        result = rotate_half(x, interleaved=False)
        ref = _ref_rotate_half_nextstep(x)
        assert torch.equal(result, ref)


# ---------------------------------------------------------------------------
# Tests: apply_rotary_pos_emb (pair) equivalence
# ---------------------------------------------------------------------------


class TestApplyRotaryPosEmb:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_matches_nextstep(self, dtype: torch.dtype) -> None:
        """Standard q,k pair rotation matches nextstep_llama reference."""
        bsz, seq, heads, dim = 2, 16, 4, 32
        q = torch.randn(bsz, heads, seq, dim, dtype=dtype)
        k = torch.randn(bsz, heads, seq, dim, dtype=dtype)
        cos, sin = _random_cos_sin(bsz, seq, dim, dtype=dtype)

        q_new, k_new = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        q_ref, k_ref = _ref_apply_rotary_pos_emb_nextstep(q, k, cos, sin, unsqueeze_dim=1)

        if dtype == torch.float32:
            assert torch.equal(q_new, q_ref)
            assert torch.equal(k_new, k_ref)
        else:
            assert torch.allclose(q_new, q_ref, atol=1e-4, rtol=1e-3)
            assert torch.allclose(k_new, k_ref, atol=1e-4, rtol=1e-3)

    def test_with_position_ids(self) -> None:
        """position_ids indexing into cos/sin works correctly."""
        bsz, seq, heads, dim = 2, 16, 4, 32
        q = torch.randn(bsz, heads, seq, dim)
        k = torch.randn(bsz, heads, seq, dim)
        # cos/sin as lookup table: [max_positions, dim]
        max_pos = 32
        cos = torch.randn(max_pos, dim)
        sin = torch.randn(max_pos, dim)
        # position_ids: [bsz, seq] — gather produces [bsz, seq, dim]
        position_ids = torch.arange(seq).unsqueeze(0).expand(bsz, -1)

        q_new, k_new = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=1)
        q_ref, k_ref = _ref_apply_rotary_pos_emb_mla(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=1)

        assert torch.equal(q_new, q_ref)
        assert torch.equal(k_new, k_ref)

    def test_mla_matches_hunyuan(self) -> None:
        """MLA reshape variant matches hunyuan_image_3 reference."""
        bsz, heads, seq, dim = 2, 4, 16, 32
        q = torch.randn(bsz, heads, seq, dim)
        k = torch.randn(bsz, heads, seq, dim)
        cos, sin = _random_cos_sin(bsz, seq, dim)

        q_new, k_new = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1, mla=True)
        q_ref, k_ref = _ref_apply_rotary_pos_emb_mla(q, k, cos, sin, unsqueeze_dim=1, mla=True)

        assert torch.equal(q_new, q_ref)
        assert torch.equal(k_new, k_ref)
