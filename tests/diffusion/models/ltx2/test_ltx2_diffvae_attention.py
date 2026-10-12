# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native FNA against an independent, test-only explicit neighborhood mask."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from vllm_omni.diffusion.models.ltx2.ops import resolve_ltx2_vae_operators
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]


def _selected_fna():
    if not current_omni_platform.is_available():
        pytest.skip("No accelerator is available")
    operators = resolve_ltx2_vae_operators(current_omni_platform.get_torch_device())
    if operators is None or operators.fna is None:
        pytest.skip("LTX DiffVAE FNA is unavailable for this device or installation")
    return operators.fna


def _load_fna_ops():
    return pytest.importorskip("vllm_omni.diffusion.models.ltx2.ops.fna", exc_type=ImportError)


def _pack(x, tile):
    b, t, h, w, heads, dim = x.shape
    tt, th, tw = tile
    nt, nh, nw = (t + tt - 1) // tt, (h + th - 1) // th, (w + tw - 1) // tw
    padded = x.new_zeros((b, nt * tt, nh * th, nw * tw, heads, dim))
    padded[:, :t, :h, :w] = x
    # Outer and inner spatial tiles both flatten in W,H,T order.
    return (
        padded.view(b, nt, tt, nh, th, nw, tw, heads, dim)
        .permute(0, 5, 3, 1, 6, 4, 2, 7, 8)
        .reshape(b, nt * nh * nw * tt * th * tw, heads, dim)
    )


def _unpack(x, shape):
    t, h, w = shape
    b, _, heads, dim = x.shape
    nt, nh, nw = ((n + 3) // 4 for n in shape)
    return (
        x.view(b, nw, nh, nt, 4, 4, 4, heads, dim)
        .permute(0, 3, 6, 2, 5, 1, 4, 7, 8)
        .reshape(b, nt * 4, nh * 4, nw * 4, heads, dim)[:, :t, :h, :w]
    )


@pytest.mark.parametrize("batch,shape", [(1, (11, 11, 11)), (2, (13, 12, 17))])
@torch.inference_mode()
def test_fna_matches_explicit_inward_shift_mask(batch, shape):
    fna = _selected_fna()
    generator = torch.Generator(device="cuda").manual_seed(42)
    q, k, v = [
        torch.randn(batch, *shape, 4, 64, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    ]
    q = q * 0.125
    coordinates = torch.stack(
        torch.meshgrid(*(torch.arange(n, device="cuda") for n in shape), indexing="ij"), dim=-1
    ).reshape(-1, 3)
    start = (coordinates - 5).clamp_min(0).minimum(torch.tensor(shape, device="cuda") - 11)
    mask = ((coordinates[None] >= start[:, None]) & (coordinates[None] < start[:, None] + 11)).all(-1)
    with sdpa_kernel(SDPBackend.MATH):
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                *(x.flatten(1, 3).transpose(1, 2).float() for x in (q, k, v)), attn_mask=mask, scale=1.0
            )
            .transpose(1, 2)
            .reshape(q.shape)
        )
    actual = _unpack(
        fna(_pack(q, (4, 4, 4)), _pack(k, (4, 4, 8)), _pack(v, (4, 4, 8)), shape=shape, window=(11, 11, 11)),
        shape,
    ).float()
    assert actual.isfinite().all()
    assert float((actual - expected).norm() / expected.norm()) < 0.005
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.025)


def test_fna_rejects_cpu_without_fallback():
    fna_ops = _load_fna_ops()
    x = torch.zeros(1, 64, 4, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA BF16"):
        fna_ops.fna3d_tilelang(x, x, x, shape=(11, 11, 11), window=(11, 11, 11))
