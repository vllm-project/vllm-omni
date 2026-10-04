# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P0 correctness matrix for the fused AdaLayerNorm CUDA path.

Covers: fp32/bf16/fp16, B>1 (shared and per-sample modulation), C in
{1536, 3072, 4096}, varying L, elementwise_affine False/True with
NON-DEFAULT weight/bias, eps variants, legal non-contiguous inputs,
broadcast scale/shift shapes ((C,), (1, C), (1, 1, C), (B, 1, C)),
fallback completeness (fp64, oversized hidden size, cross-device
parameters), large-offset/small-variance stability, and the frozen
B > 1 (B, C) native broadcast-error contract.
"""

import pytest
import torch

from vllm_omni.diffusion.layers.adalayernorm import (
    AdaLayerNorm,
    _adaln_fused_forward,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]

TOL_STRICT = {"bf16": (2e-2, 2e-2), "fp16": (1e-2, 1e-2), "fp32": (1e-3, 1e-3)}
TOL_LOOSE = {"bf16": (5e-2, 2e-2), "fp16": (2e-2, 1e-2), "fp32": (5e-3, 5e-3)}
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def tol_key(dtype):
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return "fp32"


def fp32_reference(x, scale, shift, eps, weight=None, bias=None):
    xf = x.float()
    w = weight.float() if weight is not None else None
    b = bias.float() if bias is not None else None
    xn = torch.nn.functional.layer_norm(xf, (x.shape[-1],), w, b, eps)
    return xn * (1 + scale.float()[:, None]) + shift.float()[:, None]


def make_module(hidden, elementwise_affine, eps, device, dtype, nondefault_affine=False):
    m = AdaLayerNorm(hidden, elementwise_affine=elementwise_affine, eps=eps)
    if elementwise_affine:
        m = m.to(dtype)
        if nondefault_affine:
            with torch.no_grad():
                g = torch.Generator(device="cpu").manual_seed(42)
                m.layernorm.weight.copy_(0.5 + torch.rand(hidden, generator=g))
                m.layernorm.bias.copy_(0.2 * torch.randn(hidden, generator=g))
    return m.to(device)


def make_inputs(bs, seq, hidden, dtype, device, seed=0, mod_shape=None):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(bs, seq, hidden, generator=g, device=device, dtype=dtype)
    ms = mod_shape or (1, hidden)
    scale = torch.randn(ms, generator=g, device=device, dtype=dtype)
    shift = torch.randn(ms, generator=g, device=device, dtype=dtype)
    return x, scale, shift


def assert_close(a, b, dtype, loose=False):
    atol, rtol = (TOL_LOOSE if loose else TOL_STRICT)[tol_key(dtype)]
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("chunks", [2, 6])
@pytest.mark.parametrize("bs,frames,spatial", [(1, 21, 880), (2, 3, 5), (2, 1, 5), (1, 3, 5)])
def test_matrix_framewise_modulation(dtype, chunks, bs, frames, spatial):
    hidden = 2240
    m = make_module(hidden, False, 1e-6, "cuda", dtype)
    g = torch.Generator(device="cuda").manual_seed(61)
    x = torch.randn(bs, frames, spatial, hidden, generator=g, device="cuda", dtype=dtype)
    modulation = torch.randn(bs, frames, chunks, hidden, generator=g, device="cuda", dtype=dtype)
    scale, shift = modulation.chunk(chunks, dim=2)[:2]
    # Sana 的 block/final modulation 是跨帧的 chunk view。
    assert scale.shape == (bs, frames, 1, hidden)
    assert scale.stride(1) == chunks * hidden
    fused = _adaln_fused_forward(m, x, scale, shift)
    assert fused is not None, "framewise chunk views must take the fused path"
    assert fused.shape == x.shape and fused.dtype == dtype
    assert_close(fused, m.forward_native(x, scale, shift), dtype)
    ref = torch.nn.functional.layer_norm(x.float(), (hidden,), eps=m.eps)
    ref = ref * (1 + scale.float()) + shift.float()
    assert_close(fused, ref, dtype, loose=True)


@pytest.mark.parametrize("shared_scale,shared_shift", [(False, True), (True, False), (True, True)])
def test_matrix_framewise_shared_modulation(shared_scale, shared_shift):
    dtype = torch.bfloat16
    hidden = 2240
    m = make_module(hidden, True, 1e-6, "cuda", dtype, nondefault_affine=True)
    x = torch.randn(2, 3, 5, hidden, device="cuda", dtype=dtype)
    scale = torch.randn((hidden,) if shared_scale else (2, 3, 1, hidden), device="cuda", dtype=dtype)
    shift = torch.randn((1, 1, 1, hidden) if shared_shift else (2, 3, 1, hidden), device="cuda", dtype=dtype)
    fused = _adaln_fused_forward(m, x, scale, shift)
    assert fused is not None
    assert_close(fused, m.forward_native(x, scale, shift), dtype)


def test_matrix_framewise_irregular_group_stride_falls_back():
    dtype = torch.bfloat16
    hidden = 2240
    m = make_module(hidden, False, 1e-6, "cuda", dtype)
    x = torch.randn(2, 3, 5, hidden, device="cuda", dtype=dtype)
    modulation = torch.randn(3, 2, 2, hidden, device="cuda", dtype=dtype).transpose(0, 1)
    scale, shift = modulation.chunk(2, dim=2)
    assert scale.stride(0) != x.shape[1] * scale.stride(1)
    assert _adaln_fused_forward(m, x, scale, shift) is None
    assert_close(m.forward_cuda(x, scale, shift), m.forward_native(x, scale, shift), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_sana_patch_embed_reaches_framewise_fused_path(dtype):
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmPatchEmbedMS3D

    hidden = 2240
    patch = SanaWmPatchEmbedMS3D((1, 1, 1), 8, hidden).to(device="cuda", dtype=dtype)
    latents = torch.randn(2, 8, 3, 2, 3, device="cuda", dtype=dtype)
    with torch.no_grad():
        projected = patch.proj(latents)
        tokens, (frames, height, width) = patch.project_with_shape(latents)
    torch.testing.assert_close(tokens, projected.flatten(2).transpose(1, 2), atol=0, rtol=0)
    assert tokens.is_contiguous()
    x = tokens.reshape(2, frames, height * width, hidden)
    modulation = torch.randn(2, frames, 6, hidden, device="cuda", dtype=dtype)
    scale, shift = modulation.chunk(6, dim=2)[:2]
    m = make_module(hidden, False, 1e-6, "cuda", dtype)
    fused = _adaln_fused_forward(m, x, scale, shift)
    assert fused is not None, "real Sana patch features must reach the fused path"
    assert_close(fused, m.forward_native(x, scale, shift), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize(
    "bs,seq,hidden", [(1, 4096, 3072), (2, 1024, 1536), (3, 128, 4096), (1, 1, 3072), (1, 8192, 1536)]
)
def test_matrix_main(dtype, affine, bs, seq, hidden):
    device = "cuda"
    m = make_module(hidden, affine, 1e-6, device, dtype)
    x, scale, shift = make_inputs(bs, seq, hidden, dtype, device)
    out_cuda = _adaln_fused_forward(m, x, scale, shift)
    assert out_cuda is not None, "supported matrix inputs must take the fused path"
    out_native = m.forward_native(x, scale, shift)
    assert out_cuda.shape == x.shape and out_cuda.dtype == dtype and out_cuda.device == x.device
    assert_close(out_cuda, out_native, dtype, loose=False)
    w = m.layernorm.weight if affine else None
    b = m.layernorm.bias if affine else None
    assert_close(out_cuda, fp32_reference(x, scale, shift, 1e-6, w, b), dtype, loose=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_weight_bias_nondefault(dtype):
    # weight/bias semantics with non-identity initialization:
    # out = (w * ln(x) + b) * (1 + scale) + shift
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, True, 1e-6, device, dtype, nondefault_affine=True)
    x, scale, shift = make_inputs(1, 1024, hidden, dtype, device, seed=13)
    out_cuda = _adaln_fused_forward(m, x, scale, shift)
    assert out_cuda is not None, "non-default affine parameters must take the fused path"
    out_native = m.forward_native(x, scale, shift)
    assert_close(out_cuda, out_native, dtype)
    ref = fp32_reference(x, scale, shift, 1e-6, m.layernorm.weight, m.layernorm.bias)
    assert_close(out_cuda, ref, dtype, loose=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("eps", [1e-5, 1e-3, 1e-2])
def test_matrix_eps_variants(dtype, eps):
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, eps, device, dtype)
    x, scale, shift = make_inputs(1, 512, hidden, dtype, device, seed=3)
    fused = _adaln_fused_forward(m, x, scale, shift)
    assert fused is not None, "supported eps variants must take the fused path"
    assert_close(fused, m.forward_native(x, scale, shift), dtype)
    assert_close(
        fused,
        fp32_reference(x, scale, shift, eps),
        dtype,
        loose=True,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mod_shape", ["C", "1x1xC", "1xC"])
def test_matrix_broadcast_scale_shift(dtype, mod_shape):
    # (C,), (1, 1, C) and (1, C) are all legal per-channel broadcasts against x (B, L, C)
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, _, _ = make_inputs(2, 512, hidden, dtype, device, seed=5)
    g = torch.Generator(device=device).manual_seed(6)
    if mod_shape == "C":
        scale = torch.randn(hidden, generator=g, device=device, dtype=dtype)
        shift = torch.randn(hidden, generator=g, device=device, dtype=dtype)
    elif mod_shape == "1x1xC":
        scale = torch.randn(1, 1, hidden, generator=g, device=device, dtype=dtype)
        shift = torch.randn(1, 1, hidden, generator=g, device=device, dtype=dtype)
    else:
        scale = torch.randn(1, hidden, generator=g, device=device, dtype=dtype)
        shift = torch.randn(1, hidden, generator=g, device=device, dtype=dtype)
    out_cuda = _adaln_fused_forward(m, x, scale, shift)
    assert out_cuda is not None, f"shared modulation layout {mod_shape} must take the fused path"
    out_native = m.forward_native(x, scale, shift)
    assert_close(out_cuda, out_native, dtype)
    s = scale.float().reshape(1, 1, hidden) if scale.ndim == 1 else scale.float()
    sh = shift.float().reshape(1, 1, hidden) if shift.ndim == 1 else shift.float()
    ref = fp32_reference(x, s.reshape(-1, hidden)[0:1], sh.reshape(-1, hidden)[0:1], 1e-6)
    assert_close(out_cuda, ref, dtype, loose=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_per_sample_modulation(dtype):
    # Qwen-Image's _modulate produces (B, 1, C): one modulation row per sample.
    # B > 1 must take the fused path (not fall back to native) and match native.
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, _, _ = make_inputs(4, 512, hidden, dtype, device, seed=7, mod_shape=(4, hidden))
    g = torch.Generator(device=device).manual_seed(8)
    scale = torch.randn(4, 1, hidden, generator=g, device=device, dtype=dtype)
    shift = torch.randn(4, 1, hidden, generator=g, device=device, dtype=dtype)
    fused = _adaln_fused_forward(m, x, scale, shift)
    assert fused is not None, "(B, 1, C) per-sample modulation must take the fused path"
    out_cuda = fused
    out_native = m.forward_native(x, scale, shift)
    assert_close(out_cuda, out_native, dtype)
    # fp32_reference's [:, None] expects a 2D (B, C) modulation row; the
    # (B, 1, C) tensor reshapes to it losslessly.
    ref = fp32_reference(x, scale.reshape(x.shape[0], hidden), shift.reshape(x.shape[0], hidden), 1e-6)
    assert_close(out_cuda, ref, dtype, loose=True)
    # Per-sample: different rows must receive different modulation.
    row_diff = (out_cuda[0] - out_cuda[1]).abs().max().item()
    assert row_diff > 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_chunk_view_modulation(dtype):
    # Wan2.2-style producer: the modulation projection is (B, 6, C) and the
    # consumer chunks it along dim=1. The resulting (B, 1, C) views are
    # NON-CONTIGUOUS (row stride 6*C) - the fused path must consume them
    # directly via the explicit row stride, without a contiguous copy.
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, _, _ = make_inputs(4, 512, hidden, dtype, device, seed=27)
    g = torch.Generator(device=device).manual_seed(28)
    source = torch.randn(4, 6, hidden, generator=g, device=device, dtype=dtype) * 0.1
    chunks = source.chunk(6, dim=1)
    scale, shift = chunks[1], chunks[2]
    assert scale.shape == (4, 1, hidden) and shift.shape == (4, 1, hidden)
    assert not scale.is_contiguous()
    assert scale.stride(-1) == 1
    fused = _adaln_fused_forward(m, x, scale, shift)
    assert fused is not None, "chunk-view modulation must take the fused path"
    out_native = m.forward_native(x, scale, shift)
    assert_close(fused, out_native, dtype)
    ref = fp32_reference(x, scale.reshape(4, hidden), shift.reshape(4, hidden), 1e-6)
    assert_close(fused, ref, dtype, loose=True)


def test_matrix_3d_modulation_fallback():
    # Wan2.2 TI2V-style per-token modulation: scale/shift (B, L, C) full 3D.
    # This is intentionally outside the fused kernel contract - the fast path
    # must be rejected (None) and forward_cuda must match native.
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, torch.bfloat16)
    x, scale, shift = make_inputs(2, 128, hidden, torch.bfloat16, device, seed=29, mod_shape=(2, 128, hidden))
    fused = _adaln_fused_forward(m, x, scale, shift)
    assert fused is None, "(B, L, C) per-token modulation must be rejected"
    out = m.forward_cuda(x, scale, shift)
    native = m.forward_native(x, scale, shift)
    torch.testing.assert_close(out.float(), native.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_matrix_production_widths(dtype):
    # Real consumer widths: Sana-WM 2240 (BLOCK_C 4096, 45.3% mask) and
    # Wan2.2 A14B 5120 (BLOCK_C 8192, the max supported block, 16-warp
    # compiled variant). BF16 + elementwise_affine=False.
    device = "cuda"
    for hidden in (2240, 5120):
        m = make_module(hidden, False, 1e-6, device, dtype)
        x, scale, shift = make_inputs(1, 512, hidden, dtype, device, seed=31)
        fused = _adaln_fused_forward(m, x, scale, shift)
        assert fused is not None, f"hidden={hidden} must take the fused path"
        out_native = m.forward_native(x, scale, shift)
        assert_close(fused, out_native, dtype)
        assert_close(fused, fp32_reference(x, scale, shift, 1e-6), dtype, loose=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_noncontiguous_fallback(dtype):
    # Non-contiguous inputs must fall back to native and stay correct
    # (the fused kernel only serves the contiguous fast path).
    device = "cuda"
    hidden = 1536
    m = make_module(hidden, False, 1e-6, device, dtype)
    x_big, scale, shift = make_inputs(1, 1024, hidden, dtype, device, seed=7)
    x = x_big[:, ::2, :]
    assert not x.is_contiguous()
    out_cuda = m.forward_cuda(x, scale, shift)
    out_native = m.forward_native(x, scale, shift)
    assert_close(out_cuda, out_native, dtype)
    assert_close(out_cuda, fp32_reference(x.contiguous(), scale, shift, 1e-6), dtype, loose=True)


def test_matrix_fallback_fp64():
    # fp64 is not in the kernel's supported dtype list -> must fall back to
    # native and stay correct (fallback completeness).
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, torch.float64)
    x, scale, shift = make_inputs(1, 256, hidden, torch.float64, device, seed=9)
    out_cuda = m.forward_cuda(x, scale, shift)
    out_native = m.forward_native(x, scale, shift)
    torch.testing.assert_close(out_cuda, out_native)


def test_matrix_fallback_oversized_hidden():
    # hidden sizes whose next_power_of_2 exceeds the kernel's supported block
    # bound must fall back to native and stay correct.
    device = "cuda"
    hidden = 10000  # next_power_of_2 = 16384 > MAX_BLOCK_C
    m = make_module(hidden, False, 1e-6, device, torch.bfloat16)
    x, scale, shift = make_inputs(1, 128, hidden, torch.bfloat16, device, seed=10)
    out_cuda = m.forward_cuda(x, scale, shift)
    out_native = m.forward_native(x, scale, shift)
    torch.testing.assert_close(out_cuda, out_native)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_zero_scale_shift_identity(dtype):
    # With scale = shift = 0 the output must equal a pure LayerNorm.
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, _, _ = make_inputs(1, 256, hidden, dtype, device, seed=19)
    scale = torch.zeros(1, hidden, device=device, dtype=dtype)
    shift = torch.zeros(1, hidden, device=device, dtype=dtype)
    out_cuda = _adaln_fused_forward(m, x, scale, shift)
    assert out_cuda is not None, "zero modulation must take the fused path"
    assert_close(out_cuda, m.forward_native(x, scale, shift), dtype)
    assert_close(out_cuda, m.layernorm(x), dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_determinism(dtype):
    # Two calls on the same input must agree bitwise (a hard constraint for
    # any cached/direct-launch scheme).
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, scale, shift = make_inputs(1, 4096, hidden, dtype, device, seed=11)
    out1 = _adaln_fused_forward(m, x, scale, shift)
    out2 = _adaln_fused_forward(m, x, scale, shift)
    assert out1 is not None and out2 is not None, "determinism must exercise the fused path"
    assert torch.equal(out1, out2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_multi_batch_shared_modulation(dtype):
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, scale, shift = make_inputs(4, 512, hidden, dtype, device, seed=15)
    out = _adaln_fused_forward(m, x, scale, shift)
    assert out is not None, "multi-batch shared modulation must take the fused path"
    assert_close(out, m.forward_native(x, scale, shift), dtype)
    for b in range(4):
        single = _adaln_fused_forward(m, x[b : b + 1], scale, shift)
        assert single is not None
        assert_close(out[b : b + 1], single, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_bc_modulation_preserves_native_error(dtype):
    # Frozen native contract: (B, C) modulation with B > 1 raises
    # RuntimeError (torch broadcasts B against L). The fused path preserves
    # this behavior - it neither supports nor silently changes it.
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, scale, shift = make_inputs(4, 64, hidden, dtype, device, seed=17, mod_shape=(4, hidden))
    with pytest.raises(RuntimeError):
        m.forward_native(x, scale, shift)
    with pytest.raises(RuntimeError):
        m.forward_cuda(x, scale, shift)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matrix_cross_device_fallback(dtype):
    # Raw pointers go to one Triton kernel, which does no cross-device
    # checking: the fast-path guard must reject cross-device parameters so
    # they fall back to native, which raises torch's own cross-device error
    # instead of crashing in the kernel.
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, scale, shift = make_inputs(1, 512, hidden, dtype, device, seed=21)
    cpu_scale = scale.to("cpu")
    with pytest.raises(RuntimeError):
        m.forward_cuda(x, cpu_scale, shift)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_matrix_large_offset_small_variance(dtype):
    """A large constant offset with a small variance must stay stable.

    Mirrors the fused_adaptive_group_norm_silu case: x ~ 10000 +/- 0.1 has
    mean ~1e4 and variance ~1e-2, so a plain fp32 sum of large-offset values
    rounds away the low-order bits that the variance is built from. The fused
    kernel centers the row on its first element before the tree sum
    (shift-invariant two-pass), keeping mean/variance accurate.

    For fp32 the meaningful reference is an fp64 computation, NOT the native
    eager chain: at mean/std = 1e5 the native chain itself carries a
    ~0.07 absolute deviation from the fp64 truth (its own fp32 dynamic-range
    floor), while the fused kernel stays within ~2e-6. For fp16/bf16 inputs
    the +/-0.1 perturbation quantizes away at magnitude 1e4, so the
    meaningful assertion is that outputs stay finite and match native.
    """
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, scale, shift = make_inputs(1, 4096, hidden, dtype, device, seed=23)
    # Apply the large constant offset AFTER generation: x ~ 10000 +/- 0.1 is
    # the cancellation case that motivates the shift-invariant two-pass.
    x = x * 0.1 + 10000
    out_cuda = _adaln_fused_forward(m, x, scale, shift)
    assert out_cuda is not None, "large-offset stability must exercise the fused path"
    out_native = m.forward_native(x, scale, shift)
    assert torch.isfinite(out_cuda).all()

    # fp64 reference computed from the SAME fp32 input the implementations see
    # (the kernel cannot recover bits that fp32 quantization never stored).
    x64 = x.double()
    mean64 = x64.mean(dim=-1, keepdim=True)
    var64 = x64.var(dim=-1, unbiased=False, keepdim=True)
    ref64 = (x64 - mean64) * torch.rsqrt(var64 + 1e-6) * (1 + scale.double()[:, None, :]) + shift.double()[:, None, :]

    if dtype is torch.float32:
        # fp32 fused output vs the fp64 truth of the same input: within the
        # fp32 representation floor (~2e-6 measured). The native eager chain
        # deviates ~0.07 on the same input (its own dynamic-range floor) -
        # documented, not asserted.
        torch.testing.assert_close(out_cuda.double(), ref64, atol=1e-4, rtol=1e-4)
    else:
        # bf16/fp16 quantize the perturbation away: no NaN, matches native.
        assert_close(out_cuda, out_native, dtype, loose=True)
