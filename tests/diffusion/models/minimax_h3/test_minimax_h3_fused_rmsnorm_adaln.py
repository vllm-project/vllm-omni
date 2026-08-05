# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.minimax_h3.fused_ops import (
    fused_rmsnorm_indexed_scale_shift_bf16,
)

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.diffusion]

_HIDDEN_SIZE = 5376
_EPS = 1e-5


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    normalized = F.rms_norm(x, (x.shape[-1],), weight, _EPS)
    return (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(x.dtype)


def _make_inputs(
    tokens: int,
    num_modulations: int = 16,
    *,
    index_dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(20260806 + tokens)
    x = torch.randn(tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16, generator=generator)
    scale = torch.randn(
        num_modulations,
        _HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    shift = torch.randn(
        num_modulations,
        _HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    indices = torch.randint(
        num_modulations,
        (tokens,),
        device="cuda",
        dtype=index_dtype,
        generator=generator,
    )
    if tokens:
        indices[0] = 0
        indices[-1] = num_modulations - 1
    return x, weight, shift, scale, indices


@pytest.mark.parametrize("tokens", [1, 2, 7, 31, 128, 257, 512, 1024, 2048, 8192])
def test_fused_rmsnorm_indexed_adaln_matches_reference_and_preserves_inputs(tokens: int):
    index_dtype = torch.int32 if tokens % 2 else torch.int64
    x, weight, shift, scale, indices = _make_inputs(tokens, index_dtype=index_dtype)
    snapshots = tuple(t.clone() for t in (x, weight, shift, scale, indices))

    expected = _reference(x, weight, shift, scale, indices)
    actual = fused_rmsnorm_indexed_scale_shift_bf16(x, weight, shift, scale, indices, _EPS)

    assert actual is not None
    assert actual.data_ptr() != x.data_ptr()
    assert actual.shape == x.shape
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    diff = (actual.float() - expected.float()).abs()
    print(f"tokens={tokens} max_abs={diff.max().item():.8f} mean_abs={diff.mean().item():.8f}")
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)
    for tensor, snapshot in zip((x, weight, shift, scale, indices), snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize("num_modulations", [1, 4, 16])
def test_fused_rmsnorm_indexed_adaln_modulation_rows(num_modulations: int):
    x, weight, shift, scale, indices = _make_inputs(31, num_modulations)
    if num_modulations > 1:
        indices.copy_(torch.arange(31, device="cuda") % num_modulations)
    actual = fused_rmsnorm_indexed_scale_shift_bf16(x, weight, shift, scale, indices, _EPS)
    assert actual is not None
    torch.testing.assert_close(actual, _reference(x, weight, shift, scale, indices), atol=0.03125, rtol=0.02)


def test_fused_rmsnorm_indexed_adaln_rejects_fallback_inputs():
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, dtype=torch.bfloat16)
    shift = torch.randn(2, 8, dtype=torch.bfloat16)
    scale = torch.randn(2, 8, dtype=torch.bfloat16)
    indices = torch.tensor([0, 1, 0])
    assert fused_rmsnorm_indexed_scale_shift_bf16(x, weight, shift, scale, indices, _EPS) is None

    assert (
        fused_rmsnorm_indexed_scale_shift_bf16(
            x.cuda().float(),
            weight.cuda(),
            shift.cuda(),
            scale.cuda(),
            indices.cuda(),
            _EPS,
        )
        is None
    )

    x_cuda = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert not x_cuda.is_contiguous()
    assert (
        fused_rmsnorm_indexed_scale_shift_bf16(
            x_cuda,
            weight.cuda(),
            shift.cuda(),
            scale.cuda(),
            indices.cuda(),
            _EPS,
        )
        is None
    )

    empty = torch.empty(2, 0, device="cuda", dtype=torch.bfloat16)
    assert (
        fused_rmsnorm_indexed_scale_shift_bf16(
            empty,
            torch.empty(0, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, 0, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, 0, device="cuda", dtype=torch.bfloat16),
            torch.zeros(2, device="cuda", dtype=torch.int64),
            _EPS,
        )
        is None
    )


def test_rmsnorm_indexed_adaln_cpu_fallback_and_model_integration():
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    norm = torch.nn.RMSNorm(8, eps=_EPS, dtype=torch.bfloat16)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    shift = torch.randn(2, 8, dtype=torch.bfloat16)
    scale = torch.randn(2, 8, dtype=torch.bfloat16)
    indices = torch.tensor([0, 1, 0])
    torch.testing.assert_close(
        h3._norm_modulate_scale_shift(norm, x, shift, scale, indices),
        _reference(x, norm.weight, shift, scale, indices),
        atol=0,
        rtol=0,
    )

    assert h3.MiniMaxH3DiTBlock.forward.__code__.co_names.count("_norm_modulate_scale_shift") == 2
    assert h3.MiniMaxH3FinalLayer.forward.__code__.co_names.count("_norm_modulate_scale_shift") == 1


def test_fused_rmsnorm_indexed_adaln_dynamic_compile():
    def run(
        x: torch.Tensor,
        weight: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        output = fused_rmsnorm_indexed_scale_shift_bf16(x, weight, shift, scale, indices, _EPS)
        assert output is not None
        return output

    compiled = torch.compile(run, dynamic=True, fullgraph=True)
    for tokens in (7, 31):
        inputs = _make_inputs(tokens)
        torch.testing.assert_close(compiled(*inputs), _reference(*inputs), atol=0.03125, rtol=0.02)
