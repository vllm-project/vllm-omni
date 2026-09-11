# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public/legacy imports and CPU contracts for the QK/RoPE migration."""

import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("legacy_first", [True, False])
def test_import_order_preserves_operation_identity(legacy_first: bool) -> None:
    # Fresh processes catch duplicate registration hidden by sys.modules caching.
    code = textwrap.dedent(f"""
        import importlib
        import torch

        legacy = "vllm_omni.diffusion.layers.fused_qk_norm_rope"
        public = "vllm_omni.diffusion.layers.ops"
        names = [legacy, public] if {legacy_first!r} else [public, legacy]
        first = importlib.import_module(names[0])
        registered = torch.ops.vllm_omni.fused_qk_norm_rope.default
        second = importlib.import_module(names[1])
        canonical = importlib.import_module(
            "vllm_omni.diffusion.layers.ops.rope.qk_norm_rope"
        )
        assert first.fused_qk_norm_rope is second.fused_qk_norm_rope
        assert first.fused_qk_norm_rope is canonical.fused_qk_norm_rope
        assert first.fused_qk_norm_rope.__module__ == canonical.__name__
        assert registered is torch.ops.vllm_omni.fused_qk_norm_rope.default
        assert canonical._fused_qk_norm_rope_impl.__module__ == canonical.__name__
        assert canonical._fused_qk_norm_rope_fake.__module__ == canonical.__name__
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 7])
def test_public_and_legacy_cpu_reference(dtype: torch.dtype, tokens: int) -> None:
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope as legacy
    from vllm_omni.diffusion.layers.ops import fused_qk_norm_rope

    generator = torch.Generator().manual_seed(17)
    head_dim, rotary_dim, q_heads, k_heads = 16, 12, 3, 1
    packed = torch.randn(tokens, (q_heads + 2 * k_heads) * head_dim, generator=generator, dtype=dtype)
    q, k, _ = packed.split([q_heads * head_dim, k_heads * head_dim, k_heads * head_dim], dim=-1)
    q = q.view(tokens, q_heads, head_dim)
    k = k.view(tokens, k_heads, head_dim)
    original = packed.clone()
    weights = [torch.randn(head_dim, generator=generator, dtype=dtype) for _ in range(2)]
    angles = torch.randn(tokens, rotary_dim // 2, generator=generator)
    table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
    eps = 1e-5
    half = rotary_dim // 2
    cos, sin = table[:, :half].unsqueeze(1), table[:, half:].unsqueeze(1)
    expected = []
    for x, weight in zip((q, k), weights):
        normalized = F.rms_norm(x, (head_dim,), weight, eps)
        first, second = normalized[..., :half], normalized[..., half:rotary_dim]
        expected.append(
            torch.cat((first * cos - second * sin, second * cos + first * sin, normalized[..., rotary_dim:]), dim=-1)
        )

    for entry in (fused_qk_norm_rope, legacy):
        actual = entry(q, k, weights[0], weights[1], table, eps)
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        torch.testing.assert_close(packed, original, rtol=0, atol=0)


def test_public_entry_preserves_invalid_input_error() -> None:
    from vllm_omni.diffusion.layers.ops import fused_qk_norm_rope

    with pytest.raises(ValueError, match="must be.*tokens, heads, head_dim"):
        fused_qk_norm_rope(
            torch.zeros(2, 16), torch.zeros(2, 16), torch.ones(16), torch.ones(16), torch.zeros(2, 12), 1e-5
        )


def test_registered_fake_preserves_gqa_metadata() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_omni.diffusion.layers.ops import fused_qk_norm_rope  # noqa: F401

    with FakeTensorMode():
        q, k = torch.empty(7, 3, 16), torch.empty(7, 1, 16)
        weight, table = torch.empty(16), torch.empty(7, 12)
        outputs = torch.ops.vllm_omni.fused_qk_norm_rope(q, k, weight, weight, table, 1e-5, 16, 12)
        for output, source in zip(outputs, (q, k)):
            assert output.shape == source.shape
            assert output.dtype == source.dtype
            assert output.device == source.device
