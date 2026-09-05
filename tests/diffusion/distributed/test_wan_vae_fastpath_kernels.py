# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU bit-exactness tests for the Wan VAE decoder fast path kernels and forwards."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    CACHE_T,
    WanCausalConv3d,
    WanResample,
    WanResidualBlock,
    WanRMS_norm,
    WanUpsample,
)
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fp
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    install_wan_vae_fastpath,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import triton_data_movement as dm
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import triton_rms_norm as rn
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import triton_rms_norm_cl as cl
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import triton_upsample as up

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

LOW_PRECISION = [torch.bfloat16, torch.float16]
ALL_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# Cosmos3 720p decoder stages, spatially scaled down so the suite stays fast.
NORM_SHAPES = [
    (1, 1024, 1, 45, 80),
    (1, 1024, 2, 45, 80),
    (1, 512, 4, 45, 80),
    (1, 256, 4, 90, 160),
]


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    int_dtype = torch.int16 if a.element_size() == 2 else torch.int32
    return a.shape == b.shape and bool((a.view(int_dtype) == b.view(int_dtype)).all())


def _same_dense_strides(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Stride equality ignoring size-1 dims, whose stride carries no information."""
    return a.shape == b.shape and all(
        sa == sb for sa, sb, n in zip(a.stride(), b.stride(), a.shape, strict=True) if n > 1
    )


def _with_negative_zeros(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    flat[:: max(1, flat.numel() // 97)] = -0.0
    return x


def _reference_rms_norm(norm: WanRMS_norm, x: torch.Tensor) -> torch.Tensor:
    return WanRMS_norm.forward(norm, x)


def _make_norm(channels: int, dtype: torch.dtype, images: bool = False) -> WanRMS_norm:
    norm = WanRMS_norm(channels, images=images).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.gamma.copy_(torch.randn_like(norm.gamma) * 0.5 + 1.0)
    return norm


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #


@torch.no_grad()
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", NORM_SHAPES)
def test_vector_norm_from_low_precision_matches_fp32_normalize_denominator(dtype: torch.dtype, shape) -> None:
    """The fast path relies on ATen running the same reduction for both inputs."""
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda").to(dtype)
    expected = x.float().norm(2, dim=1, keepdim=True)
    actual = torch.linalg.vector_norm(x, dim=1, keepdim=True, dtype=torch.float32)
    assert torch.equal(actual, expected)
    frame_major = x.permute(0, 2, 1, 3, 4).contiguous().permute(0, 2, 1, 3, 4)
    expected = frame_major.float().norm(2, dim=1, keepdim=True)
    actual = torch.linalg.vector_norm(frame_major, dim=1, keepdim=True, dtype=torch.float32)
    assert torch.equal(actual, expected)


@torch.no_grad()
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("layout", ["contiguous", "frame_major"])
@pytest.mark.parametrize("autocast", [False, True])
def test_rms_norm_fastpath_is_bitwise_exact(dtype: torch.dtype, shape, layout: str, autocast: bool) -> None:
    torch.manual_seed(0)
    norm = _make_norm(shape[1], dtype)
    x = _with_negative_zeros(torch.randn(shape, device="cuda").to(dtype))
    if layout == "frame_major":
        x = x.permute(0, 2, 1, 3, 4).contiguous().permute(0, 2, 1, 3, 4)
    with torch.autocast("cuda", dtype=dtype if dtype is not torch.float32 else torch.bfloat16, enabled=autocast):
        expected = _reference_rms_norm(norm, x)
        actual = fp.rms_norm_fastpath(norm, x)
    assert actual is not None
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert _bits_equal(actual, expected), "RMSNorm fast path is not bit-exact (signed zeros included)"


@torch.no_grad()
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_rms_norm_fastpath_handles_attention_block_4d_input(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    norm = _make_norm(1024, dtype, images=True)
    x = torch.randn(2, 1024, 45, 80, device="cuda").to(dtype)
    expected = _reference_rms_norm(norm, x)
    actual = fp.rms_norm_fastpath(norm, x)
    assert actual is not None and _bits_equal(actual, expected)


@torch.no_grad()
@pytest.mark.parametrize("dtype", LOW_PRECISION)
def test_fused_silu_epilogue_matches_silu_of_reference(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    if not rn.silu_epilogue_is_exact(device, dtype):
        pytest.skip(f"fused SiLU epilogue is not bit-exact for {dtype} on this toolkit; fusion stays disabled")
    torch.manual_seed(0)
    norm = _make_norm(256, dtype)
    x = _with_negative_zeros(torch.randn(1, 256, 4, 90, 160, device="cuda").to(dtype))
    expected = F.silu(_reference_rms_norm(norm, x))
    actual = fp.rms_norm_fastpath(norm, x, silu=True)
    assert actual is not None and _bits_equal(actual, expected)


@torch.no_grad()
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", [(1, 1024, 2, 45, 80), (1, 256, 4, 90, 160), (2, 512, 45, 80)])
@pytest.mark.parametrize("silu", [False, True])
def test_channels_last_rms_norm_is_close_and_keeps_layout(dtype: torch.dtype, shape, silu: bool) -> None:
    torch.manual_seed(0)
    norm = _make_norm(shape[1], dtype, images=len(shape) == 4)
    memory_format = torch.channels_last_3d if len(shape) == 5 else torch.channels_last
    x = torch.randn(shape, device="cuda").to(dtype).contiguous(memory_format=memory_format)
    expected = _reference_rms_norm(norm, x)
    if silu:
        expected = F.silu(expected)
    actual = cl.rms_norm_channels_last(x, norm.gamma, norm.scale, silu=silu)
    assert actual is not None
    assert actual.is_contiguous(memory_format=memory_format)
    assert actual.dtype == expected.dtype
    # Fast-math kernel: approximate divide/exp and a single final rounding.
    if dtype is torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    # Declines channels-first input and too many channels.
    assert cl.rms_norm_channels_last(x.contiguous(), norm.gamma, norm.scale) is None
    wide = torch.randn(1, 2048, 1, 4, 4, device="cuda", dtype=dtype).contiguous(memory_format=torch.channels_last_3d)
    assert cl.rms_norm_channels_last(wide, torch.ones(2048, device="cuda", dtype=dtype), 1.0) is None


@torch.no_grad()
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("silu", [False, True])
def test_channels_last_rms_norm_absorbs_conv_bias(dtype: torch.dtype, silu: bool) -> None:
    torch.manual_seed(0)
    shape = (1, 256, 4, 90, 160)
    norm = _make_norm(shape[1], dtype)
    x = torch.randn(shape, device="cuda").to(dtype).contiguous(memory_format=torch.channels_last_3d)
    bias = (torch.randn(shape[1], device="cuda") * 0.1).to(dtype)
    # Reference: ATen adds the conv bias in place (one rounding) before the norm.
    expected = _reference_rms_norm(norm, x + bias.view(1, -1, 1, 1, 1))
    if silu:
        expected = F.silu(expected)
    actual = cl.rms_norm_channels_last(x, norm.gamma, norm.scale, silu=silu, bias=bias)
    assert actual is not None
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    if dtype is torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    assert cl.rms_norm_channels_last(x, norm.gamma, norm.scale, bias=bias[:-1]) is None
    assert cl.rms_norm_channels_last(x, norm.gamma, norm.scale, bias=bias.view(1, -1)) is None
    assert cl.rms_norm_channels_last(x, norm.gamma, norm.scale, bias=bias.cpu()) is None


@torch.no_grad()
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_channels_last_residual_block_folds_conv1_bias(dtype: torch.dtype, monkeypatch) -> None:
    """The conv1 bias is left un-added and consumed by norm2's kernel (rounded like ATen); output stays close."""
    torch.manual_seed(0)
    block = WanResidualBlock(64, 64, dropout=0.0).eval().to(device="cuda", dtype=dtype)
    with torch.no_grad():
        for conv in (block.conv1, block.conv2):
            conv.bias.copy_(torch.randn_like(conv.bias) * 0.1)
    x = torch.randn(1, 64, 4, 24, 40, device="cuda", dtype=dtype)
    expected = WanResidualBlock.forward(block, x, feat_cache=[None, None], feat_idx=[0])

    for conv in (block.conv1, block.conv2):  # as the installer does: conv weights only
        conv.to(memory_format=torch.channels_last_3d)
    cfg = fp.FastPathConfig(channels_last=True)
    for module in (block, block.norm1, block.norm2):
        setattr(module, fp.CFG_ATTR, cfg)
    seen: list[torch.Tensor | None] = []
    original = fp.rms_norm_fastpath

    def spy(norm, x, *, silu=False, bias=None):
        seen.append(bias)
        return original(norm, x, silu=silu, bias=bias)

    monkeypatch.setattr(fp, "rms_norm_fastpath", spy)
    x_cl = x.contiguous(memory_format=torch.channels_last_3d)
    actual = fp.residual_block_forward(block, x_cl, feat_cache=[None, None], feat_idx=[0])
    assert seen[0] is None and seen[1] is block.conv1.bias, "norm2 should receive conv1's un-added bias"
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    if dtype is torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)  # channels-last convs may run TF32
    else:
        # Fast-math norm plus channels-last cuDNN accumulation order: a few outputs differ by one bf16 ulp.
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_rms_norm_scale_declines_bad_operands() -> None:
    x = torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16)
    denom = torch.ones(1, 1, 16, device="cuda")
    gamma = torch.ones(8, device="cuda", dtype=torch.bfloat16)
    assert rn.rms_norm_scale(x, denom, gamma, 1.0) is not None
    assert rn.rms_norm_scale(x.cpu(), denom.cpu(), gamma.cpu(), 1.0) is None
    assert rn.rms_norm_scale(x, denom.to(torch.bfloat16), gamma, 1.0) is None
    assert rn.rms_norm_scale(x, denom, gamma.float(), 1.0) is None
    assert rn.rms_norm_scale(x.transpose(1, 2), denom, gamma, 1.0) is None
    assert rn.rms_norm_scale(x, torch.ones(1, 1, 8, device="cuda"), gamma, 1.0) is None


# --------------------------------------------------------------------------- #
# Data movement
# --------------------------------------------------------------------------- #


def _dense_5d(shape, dtype: torch.dtype, channels_last: bool) -> torch.Tensor:
    tensor = torch.randn(shape, device="cuda", dtype=dtype)
    if channels_last:
        return tensor.contiguous(memory_format=torch.channels_last_3d)
    return tensor.contiguous()


def _reference_cat_pad(x: torch.Tensor, cache: torch.Tensor | None, padding) -> torch.Tensor:
    reference_padding = list(padding)
    if cache is not None:
        x = torch.cat([cache, x], dim=2)
        reference_padding[4] -= cache.shape[2]
    return F.pad(x, reference_padding)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "frame_major"])
@pytest.mark.parametrize(
    "channels,frames,height,width,cache_frames,padding",
    [
        (96, 1, 10, 14, 0, (1, 1, 1, 1, 2, 0)),
        (96, 1, 10, 14, 1, (1, 1, 1, 1, 2, 0)),
        (96, 1, 10, 14, 2, (1, 1, 1, 1, 2, 0)),
        (64, 1, 10, 14, 2, (0, 0, 0, 0, 2, 0)),
        (48, 4, 10, 14, 2, (1, 1, 1, 1, 2, 0)),
        # Non-power-of-two channels and a ragged number of rows per program.
        (12, 3, 7, 9, 1, (1, 1, 1, 1, 2, 0)),
        # Width wider than one block so the in-kernel width loop runs.
        (8, 2, 3, 1300, 2, (1, 1, 1, 1, 2, 0)),
        # Cosmos3 stage-3 shape, scaled down in height.
        (256, 4, 8, 640, 2, (1, 1, 1, 1, 2, 0)),
    ],
)
def test_cat_pad_5d_is_bitwise_and_layout_exact(
    dtype, layout, channels, frames, height, width, cache_frames, padding
) -> None:
    torch.manual_seed(0)
    channels_last = layout == "channels_last"
    if layout == "frame_major":
        x = torch.randn((1, frames, channels, height, width), device="cuda", dtype=dtype).permute(0, 2, 1, 3, 4)
    else:
        x = _dense_5d((1, channels, frames, height, width), dtype, channels_last)
    cache = None
    if cache_frames:
        pad_height, pad_width = padding[2], padding[0]
        buffer = _dense_5d(
            (1, channels, cache_frames, height + 2 * pad_height, width + 2 * pad_width), dtype, channels_last
        )
        cache = buffer[:, :, :, pad_height : pad_height + height, pad_width : pad_width + width]

    reference = _reference_cat_pad(x, cache, padding)
    output = dm.cat_pad_5d(x, cache, padding)
    assert output is not None
    assert output.stride() == reference.stride()
    assert torch.equal(output, reference)

    pair = dm.cat_pad_5d(x, cache, padding, keep_cache_frames=CACHE_T)
    assert pair is not None
    output_with_cache, next_cache = pair
    assert torch.equal(output_with_cache, reference)
    pad_height, pad_width = padding[2], padding[0]
    keep_frames = min(CACHE_T, reference.shape[2])
    expected_cache = reference[:, :, -keep_frames:, pad_height : pad_height + height, pad_width : pad_width + width]
    assert torch.equal(next_cache, expected_cache)
    assert next_cache.is_contiguous(memory_format=torch.channels_last_3d) is channels_last


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "frame_major"])
@pytest.mark.parametrize("cache_layout", ["contiguous", "channels_last"])
@pytest.mark.parametrize(
    "channels,frames,height,width,cache_frames",
    [
        (96, 1, 10, 14, 0),
        (96, 1, 10, 14, 2),
        (12, 3, 7, 9, 1),
        (8, 2, 3, 1300, 2),
        (256, 4, 8, 640, 2),
        (64, 4, 70, 70, 2),
    ],
)
def test_cat_pad_5d_assembles_channels_last_output_from_channels_first_input(
    dtype, layout, cache_layout, channels, frames, height, width, cache_frames
) -> None:
    """The conv_out path: channels-first activations, channels-last padded input, cache in the input layout."""
    torch.manual_seed(0)
    padding = (1, 1, 1, 1, 2, 0)
    if layout == "frame_major":
        x = torch.randn((1, frames, channels, height, width), device="cuda", dtype=dtype).permute(0, 2, 1, 3, 4)
    else:
        x = torch.randn((1, channels, frames, height, width), device="cuda", dtype=dtype)
    cache = None
    if cache_frames:
        cache = _dense_5d((1, channels, cache_frames, height, width), dtype, cache_layout == "channels_last")
    reference = _reference_cat_pad(x, cache, padding)

    output = dm.cat_pad_5d(x, cache, padding, channels_last_output=True)
    assert output is not None
    assert output.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(output, reference)

    pair = dm.cat_pad_5d(x, cache, padding, keep_cache_frames=CACHE_T, channels_last_output=True)
    assert pair is not None
    output_with_cache, next_cache = pair
    assert output_with_cache.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(output_with_cache, reference)
    keep_frames = min(CACHE_T, reference.shape[2])
    assert next_cache.is_contiguous()  # follows x's layout, not the output's
    assert torch.equal(next_cache, reference[:, :, -keep_frames:, 1 : 1 + height, 1 : 1 + width])


@torch.no_grad()
@pytest.mark.parametrize("layout", ["contiguous", "frame_major"])
def test_conv_out_channels_last_verdict_is_recorded_and_exact(layout: str) -> None:
    torch.manual_seed(0)
    conv = WanCausalConv3d(64, 12, 3, padding=1).to(device="cuda", dtype=torch.bfloat16)
    if layout == "frame_major":
        chunk = torch.randn(1, 4, 64, 24, 40, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3, 4)
    else:
        chunk = torch.randn(1, 64, 4, 24, 40, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(1, 64, 2, 24, 40, device="cuda", dtype=torch.bfloat16)
    reference = WanCausalConv3d.forward(conv, chunk, cache)
    fast_cache = [cache.clone()]
    out = fp._run_conv_out_channels_last(conv, chunk, fast_cache, 0)
    assert out is not None  # the verdict decides which formulation's result is returned, both are exact
    assert out.is_contiguous()
    assert _bits_equal(out, reference)
    verdicts = fp._CONV_OUT_LAYOUT_VERDICTS[conv]
    assert len(verdicts) == 1
    assert torch.equal(fast_cache[0], torch.cat([cache, chunk], dim=2)[:, :, -CACHE_T:])
    assert fast_cache[0].is_contiguous()
    again = fp._run_conv_out_channels_last(conv, chunk, [cache.clone()], 0)
    if next(iter(verdicts.values())):
        assert again is not None and _bits_equal(again, reference)
    else:
        assert again is None  # routed back to the standard path
    # Channels-last activations are not this path's business.
    assert (
        fp._run_conv_out_channels_last(conv, chunk.contiguous(memory_format=torch.channels_last_3d), [None], 0) is None
    )


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "frame_major"])
@pytest.mark.parametrize(
    "channels,frames,height,width,cache_frames,pad_front",
    [(96, 1, 10, 14, 0, 2), (96, 1, 10, 14, 1, 2), (96, 4, 10, 14, 2, 2), (12, 3, 7, 9, 2, 2), (256, 4, 8, 640, 2, 2)],
)
def test_cat_time_5d_is_bitwise_and_layout_exact(
    dtype, layout, channels, frames, height, width, cache_frames, pad_front
):
    torch.manual_seed(0)
    channels_last = layout == "channels_last"
    if layout == "frame_major":
        x = torch.randn((1, frames, channels, height, width), device="cuda", dtype=dtype).permute(0, 2, 1, 3, 4)
    else:
        x = _dense_5d((1, channels, frames, height, width), dtype, channels_last)
    cache = _dense_5d((1, channels, cache_frames, height, width), dtype, channels_last) if cache_frames else None

    parts = [torch.zeros_like(x[:, :, :1]).expand(-1, -1, pad_front - cache_frames, -1, -1)]
    if cache is not None:
        parts.append(cache)
    parts.append(x)
    reference = torch.cat(parts, dim=2)
    if channels_last:
        reference = reference.contiguous(memory_format=torch.channels_last_3d)
    else:
        reference = reference.contiguous()

    pair = dm.cat_time_5d(x, cache, pad_front, keep_cache_frames=CACHE_T)
    assert pair is not None
    assembled, next_cache = pair
    assert assembled.stride() == reference.stride()
    assert torch.equal(assembled, reference)
    assert torch.equal(next_cache, reference[:, :, -CACHE_T:])
    assert next_cache.is_contiguous(memory_format=torch.channels_last_3d) is channels_last
    assert torch.equal(dm.cat_time_5d(x, cache, pad_front), reference)


@torch.no_grad()
def test_cached_conv_spatial_padding_verdict_is_recorded_and_exact() -> None:
    torch.manual_seed(0)
    conv = WanCausalConv3d(64, 64, 3, padding=1).to(device="cuda", dtype=torch.bfloat16)
    for layout in (torch.contiguous_format, torch.channels_last_3d):
        if layout is torch.channels_last_3d:
            conv.to(memory_format=torch.channels_last_3d)
        chunk = torch.randn(1, 64, 4, 24, 40, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=layout)
        cache = torch.randn(1, 64, 2, 24, 40, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=layout)
        reference_cache = [cache.clone()]
        reference = WanCausalConv3d.forward(conv, chunk, cache)  # upstream pre-padded path
        fast_cache = [cache.clone()]
        out = fp._run_cached_causal_conv(conv, chunk, fast_cache, 0)
        verdicts = fp._SPATIAL_PAD_VERDICTS[conv]
        assert len(verdicts) >= 1
        assert _bits_equal(out, reference)  # exact whichever path the verdict selected
        again = fp._run_cached_causal_conv(conv, chunk, [cache.clone()], 0)
        assert _bits_equal(again, reference)
        assert torch.equal(fast_cache[0], torch.cat([cache, chunk], dim=2)[:, :, -CACHE_T:])
        del reference_cache


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("main_layout", ["contiguous", "channels_last", "frame_major"])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize(
    "in_channels,out_channels,frames,height,width,factor_temporal,factor_spatial,drop_first",
    [
        # Cosmos3 / Wan2.2 block shapes: repeats 8 (same channels), 4 (halved), 2 (spatial only, halved).
        (128, 64, 1, 10, 14, 2, 2, False),
        (128, 64, 1, 10, 14, 2, 2, True),
        (64, 32, 2, 10, 14, 1, 2, False),
        (64, 64, 4, 10, 14, 1, 2, False),
        # More output than input channels (repeats 8 with a 2x channel increase) and a 4x spatial factor.
        (32, 64, 2, 10, 14, 1, 2, False),
        (16, 16, 1, 5, 6, 1, 4, False),
        # Odd sizes and a row wider than one column block.
        (12, 12, 3, 7, 9, 2, 2, True),
        (8, 8, 1, 3, 700, 1, 2, False),
    ],
)
def test_dup_up3d_add_is_bitwise_and_layout_exact(
    dtype,
    main_layout,
    with_bias,
    in_channels,
    out_channels,
    frames,
    height,
    width,
    factor_temporal,
    factor_spatial,
    drop_first,
) -> None:
    torch.manual_seed(0)
    repeats = out_channels * factor_temporal * factor_spatial * factor_spatial // in_channels
    source = _dense_5d((1, in_channels, frames, height, width), dtype, main_layout == "channels_last")
    if main_layout == "frame_major":
        # The previous block's output is frame-major too in the lossless level.
        source = torch.randn((1, frames, in_channels, height, width), device="cuda", dtype=dtype)
        source = source.permute(0, 2, 1, 3, 4)
    out_frames = frames * factor_temporal - (factor_temporal - 1 if drop_first else 0)
    shape = (1, out_channels, out_frames, height * factor_spatial, width * factor_spatial)
    if main_layout == "channels_last":
        main = _dense_5d(shape, dtype, True)
    elif main_layout == "frame_major":
        main = torch.randn((shape[0], shape[2], shape[1], shape[3], shape[4]), device="cuda", dtype=dtype)
        main = main.permute(0, 2, 1, 3, 4)
    else:
        main = _dense_5d(shape, dtype, False)

    shortcut = source.repeat_interleave(repeats, dim=1)
    shortcut = shortcut.view(1, out_channels, factor_temporal, factor_spatial, factor_spatial, frames, height, width)
    shortcut = shortcut.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    shortcut = shortcut.view(shape[0], shape[1], frames * factor_temporal, shape[3], shape[4])
    if drop_first:
        shortcut = shortcut[:, :, factor_temporal - 1 :]
    bias = None
    biased_main = main
    if with_bias:
        bias = (torch.randn(out_channels, device="cuda") * 0.1).to(dtype)
        biased_main = main.clone()
        biased_main.add_(bias.view(1, -1, 1, 1, 1))  # ATen's conv bias add
    reference = biased_main + shortcut

    output = dm.dup_up3d_add(main, source, factor_temporal, factor_spatial, repeats, drop_first, main_bias=bias)
    assert output is not None
    assert _same_dense_strides(output, reference)
    assert torch.equal(output, reference)


def test_dup_up3d_add_declines_unsupported_cases() -> None:
    main = torch.randn(1, 8, 2, 10, 14, device="cuda", dtype=torch.bfloat16)
    source = torch.randn(1, 32, 2, 5, 7, device="cuda", dtype=torch.bfloat16)
    # repeats (1) below the spatial factor: the source channel would change along a row.
    assert dm.dup_up3d_add(main, source, 1, 2, 1, False) is None
    source = torch.randn(1, 8, 2, 5, 7, device="cuda", dtype=torch.bfloat16)
    assert dm.dup_up3d_add(main, source, 1, 2, 4, False) is not None
    # Mixed layouts, wrong bias shape, wrong main shape.
    assert dm.dup_up3d_add(main.contiguous(memory_format=torch.channels_last_3d), source, 1, 2, 4, False) is None
    assert dm.dup_up3d_add(main, source, 1, 2, 4, False, main_bias=torch.ones(7, device="cuda")) is None
    assert dm.dup_up3d_add(main[:, :, :1], source, 1, 2, 4, False) is None


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "frame_major"])
@pytest.mark.parametrize(
    "frames,channels,height,width",
    [
        (1, 96, 10, 14),
        (4, 64, 10, 14),
        # Non-power-of-two channels and ragged rows/columns for every block size.
        (3, 12, 7, 9),
        # Width wider than one column block so the in-kernel loop runs.
        (2, 8, 3, 1300),
        # Cosmos3 720p stage-3 input, scaled down in height.
        (4, 256, 8, 640),
        # Channels equal to and above the channels-last block: in-kernel channel loop.
        (2, 1024, 5, 160),
        (1, 1100, 3, 20),
    ],
)
def test_upsample_nearest_2x_is_bitwise_and_layout_exact(dtype, layout, frames, channels, height, width) -> None:
    torch.manual_seed(0)
    if layout == "frame_major":
        # The strided ``(t, c, h, w)`` view ``WanResample`` hands the upsampler in the lossless level.
        x = fp._merge_batch_and_frames(torch.randn((1, channels, frames, height, width), device="cuda", dtype=dtype))
        assert frames == 1 or not x.is_contiguous()
    elif layout == "channels_last":
        x = torch.randn((frames, channels, height, width), device="cuda", dtype=dtype)
        x = x.contiguous(memory_format=torch.channels_last)
    else:
        x = torch.randn((frames, channels, height, width), device="cuda", dtype=dtype)
    for mode in ("nearest-exact", "nearest"):
        reference = F.interpolate(x, scale_factor=(2.0, 2.0), mode=mode)
        output = up.upsample_nearest_2x(x)
        assert output is not None
        assert output.stride() == reference.stride()
        assert torch.equal(output, reference)


def test_upsample_nearest_2x_declines_unsupported_inputs() -> None:
    x = torch.randn(2, 8, 6, 10, device="cuda", dtype=torch.bfloat16)
    assert up.upsample_nearest_2x(x) is not None
    assert up.upsample_nearest_2x(x.cpu()) is None
    assert up.upsample_nearest_2x(x.unsqueeze(2)) is None
    assert up.upsample_nearest_2x(x.transpose(2, 3)) is None
    assert up.upsample_nearest_2x(x[..., ::2]) is None
    assert up.upsample_nearest_2x(x.double()) is None


@torch.no_grad()
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_upsample_forward_matches_wan_upsample(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 64, 10, 14, device="cuda", dtype=dtype)
    for layout in (torch.contiguous_format, torch.channels_last):
        x = x.contiguous(memory_format=layout)
        module = WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact")
        expected = WanUpsample.forward(module, x)
        actual = fp.upsample_forward(module, x)
        assert actual.stride() == expected.stride()
        assert torch.equal(actual, expected)
        # Anything but a nearest 2x upsample takes the reference path.
        other = WanUpsample(scale_factor=(3.0, 3.0), mode="nearest-exact")
        assert torch.equal(fp.upsample_forward(other, x), WanUpsample.forward(other, x))


@torch.no_grad()
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("mode", ["upsample2d", "upsample3d"])
def test_resample_forward_can_leave_conv_bias_pending(dtype: torch.dtype, mode: str) -> None:
    torch.manual_seed(0)
    resample = WanResample(32, mode=mode).eval().to(device="cuda", dtype=dtype)
    setattr(resample, fp.CFG_ATTR, fp.FastPathConfig())
    x = torch.randn(1, 32, 2, 10, 14, device="cuda", dtype=dtype)
    cache = [torch.randn(1, 32, 2, 10, 14, device="cuda", dtype=dtype)] if mode == "upsample3d" else [None]
    expected = WanResample.forward(
        resample, x, feat_cache=[c.clone() if c is not None else None for c in cache], feat_idx=[0]
    )
    out, bias = fp.resample_forward(
        resample, x, [c.clone() if c is not None else None for c in cache], [0], return_bias=True
    )
    assert bias is resample.resample[1].bias
    assert out.stride() == expected.stride()
    assert _bits_equal(out + bias.view(1, -1, 1, 1, 1), expected)
    plain = fp.resample_forward(resample, x, [c.clone() if c is not None else None for c in cache], [0])
    assert _bits_equal(plain, expected)


@torch.no_grad()
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("with_bias_x", [True, False])
@pytest.mark.parametrize("with_bias_h", [True, False])
def test_add_bias_residual_matches_aten_rounding(dtype: torch.dtype, with_bias_x: bool, with_bias_h: bool) -> None:
    torch.manual_seed(0)
    shape = (1, 256, 4, 45, 80)
    x = torch.randn(shape, device="cuda").to(dtype)
    h = torch.randn(shape, device="cuda").to(dtype)
    bias_x = torch.randn(256, device="cuda").to(dtype) if with_bias_x else None
    bias_h = torch.randn(256, device="cuda").to(dtype) if with_bias_h else None
    expected_x = x.clone()
    expected_h = h.clone()
    if bias_x is not None:
        expected_x.add_(bias_x.view(1, -1, 1, 1, 1))
    if bias_h is not None:
        expected_h.add_(bias_h.view(1, -1, 1, 1, 1))
    expected = expected_x + expected_h
    actual = dm.add_bias_residual(x, bias_x, h, bias_h)
    assert actual is not None
    assert actual.stride() == expected.stride()
    assert _bits_equal(actual, expected)
    x_cl = x.contiguous(memory_format=torch.channels_last_3d)
    h_cl = h.contiguous(memory_format=torch.channels_last_3d)
    actual_cl = dm.add_bias_residual(x_cl, bias_x, h_cl, bias_h)
    assert actual_cl is not None and actual_cl.is_contiguous(memory_format=torch.channels_last_3d)
    assert _bits_equal(actual_cl.contiguous(), expected)
    assert dm.add_bias_residual(x_cl, bias_x, h, bias_h) is None
    assert dm.add_bias_residual(x.permute(0, 2, 1, 3, 4).contiguous().permute(0, 2, 1, 3, 4), bias_x, h, bias_h) is None


@torch.no_grad()
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("temporal_only", [False, True])
@pytest.mark.parametrize("start", [None, "Rep"])
def test_cached_conv_chunk_loop_matches_reference_including_mixed_paths(
    monkeypatch, channels_last: bool, temporal_only: bool, start
) -> None:
    torch.manual_seed(0)
    channels = 64
    if temporal_only:
        conv = WanCausalConv3d(channels, 2 * channels, (3, 1, 1), padding=(1, 0, 0))
    else:
        conv = WanCausalConv3d(channels, channels, 3, padding=1)
    conv = conv.to(device="cuda", dtype=torch.float32)
    chunks = [_dense_5d((1, channels, 1, 10, 14), torch.float32, channels_last) for _ in range(4)]

    decline = lambda *a, **k: None  # noqa: E731
    original_time, original_pad = fp.dm.cat_time_5d, fp.dm.cat_pad_5d

    def run(mode: list[str]):
        cache = [start]
        outputs = []
        for chunk, m in zip(chunks, mode, strict=True):
            monkeypatch.setattr(fp.dm, "cat_time_5d", decline if m in ("reference", "padded") else original_time)
            monkeypatch.setattr(fp.dm, "cat_pad_5d", decline if m == "reference" else original_pad)
            outputs.append(fp._run_cached_causal_conv(conv, chunk, cache, 0))
        monkeypatch.setattr(fp.dm, "cat_time_5d", original_time)
        monkeypatch.setattr(fp.dm, "cat_pad_5d", original_pad)
        return outputs

    reference = run(["reference"] * 4)
    for pattern in (
        ["time"] * 4,
        ["padded"] * 4,
        ["time", "reference", "padded", "time"],
        ["reference", "padded", "time", "reference"],
    ):
        fused = run(pattern)
        for out, ref in zip(fused, reference, strict=True):
            assert torch.equal(out, ref), pattern


# --------------------------------------------------------------------------- #
# Whole decoder
# --------------------------------------------------------------------------- #

TINY_RESIDUAL = dict(
    base_dim=8,
    decoder_base_dim=16,
    z_dim=8,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    temperal_downsample=[False, True, True],
    is_residual=True,
    patch_size=2,
    in_channels=12,
    out_channels=12,
)
TINY_WAN21 = dict(base_dim=8, z_dim=4, dim_mult=[1, 2], num_res_blocks=1, temperal_downsample=[False, True])


def _build_pair(config: dict, dtype: torch.dtype):
    torch.manual_seed(0)
    reference = AutoencoderKLWan(**config).eval().to(device="cuda", dtype=dtype)
    candidate = AutoencoderKLWan(**config).eval().to(device="cuda", dtype=dtype)
    candidate.load_state_dict(reference.state_dict())
    return reference, candidate


@torch.no_grad()
@pytest.mark.parametrize("config", [TINY_RESIDUAL, TINY_WAN21], ids=["residual_patch2", "wan21"])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("frames", [1, 2, 5])
def test_lossless_decoder_is_bitwise_exact_on_cuda(config: dict, dtype: torch.dtype, frames: int) -> None:
    reference, candidate = _build_pair(config, dtype)
    report = install_wan_vae_fastpath(candidate, level="lossless")
    assert report.installed, report
    torch.manual_seed(1)
    latents = torch.randn(1, config["z_dim"], frames, 6, 10, device="cuda").to(dtype)
    autocast = dtype is not torch.float32
    with torch.autocast("cuda", dtype=dtype, enabled=autocast):
        expected = reference.decode(latents, return_dict=False)[0]
        actual = candidate.decode(latents, return_dict=False)[0]
    assert actual.stride() == expected.stride()
    assert _bits_equal(actual, expected)


@torch.no_grad()
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_omni_wan_vae_decode_is_bitwise_exact_on_cuda(dtype: torch.dtype) -> None:
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan

    torch.manual_seed(0)
    reference = AutoencoderKLWan(**TINY_RESIDUAL).eval().to(device="cuda", dtype=dtype)
    candidate = OmniAutoencoderKLWan(**TINY_RESIDUAL).eval().to(device="cuda", dtype=dtype)
    candidate.load_state_dict(reference.state_dict())
    install_wan_vae_fastpath(candidate)
    latents = torch.randn(1, TINY_RESIDUAL["z_dim"], 5, 6, 10, device="cuda").to(dtype)
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not torch.float32):
        expected = reference.decode(latents, return_dict=False)[0]
    actual = candidate.decode(latents, return_dict=False)[0]  # applies its own autocast context
    assert actual.stride() == expected.stride()
    assert _bits_equal(actual, expected)


@torch.no_grad()
@pytest.mark.parametrize(
    ("dtype", "allow_tf32", "tolerance"),
    [
        pytest.param(torch.bfloat16, True, 3e-2, id="bf16"),
        # With TF32 disabled the only difference is cuDNN's channels-last algorithm
        # choice and the norm kernel's reduction order.
        pytest.param(torch.float32, False, 2e-4, id="fp32"),
        # PyTorch's default allows cuDNN to run fp32 convolutions in TF32; the
        # channels-last algorithms use it while the NCDHW reference may not, so
        # the deviation is dominated by TF32's 10-bit mantissa.
        pytest.param(torch.float32, True, 1e-2, id="fp32-tf32"),
    ],
)
def test_channels_last_decoder_keeps_layout_and_stays_close(
    dtype: torch.dtype, allow_tf32: bool, tolerance: float
) -> None:
    reference, candidate = _build_pair(TINY_RESIDUAL, dtype)
    report = install_wan_vae_fastpath(candidate, level="channels_last")
    assert report.installed and report.channels_last

    violations: list[str] = []

    def audit(name: str):
        def hook(module: nn.Module, inputs) -> None:
            x = inputs[0]
            if x.dim() == 5 and x.shape[1] > 1 and not x.is_contiguous(memory_format=torch.channels_last_3d):
                violations.append(f"{name}: {tuple(x.shape)} strides={x.stride()}")
            if x.dim() == 4 and x.shape[1] > 1 and not x.is_contiguous(memory_format=torch.channels_last):
                violations.append(f"{name}: {tuple(x.shape)} strides={x.stride()}")

        return hook

    handles = [
        module.register_forward_pre_hook(audit(name))
        for name, module in candidate.decoder.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Conv3d))
    ]
    torch.manual_seed(1)
    latents = torch.randn(1, TINY_RESIDUAL["z_dim"], 3, 6, 10, device="cuda").to(dtype)
    with (
        torch.backends.cudnn.flags(enabled=True, allow_tf32=allow_tf32),
        torch.autocast("cuda", dtype=dtype, enabled=dtype is not torch.float32),
    ):
        expected = reference.decode(latents, return_dict=False)[0]
        actual = candidate.decode(latents, return_dict=False)[0]
    for handle in handles:
        handle.remove()
    assert not violations, "\n".join(violations)
    assert actual.shape == expected.shape
    difference = (actual.float() - expected.float()).abs()
    error = difference.max().item()
    mse = (difference**2).mean().item()
    psnr = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()
    assert error <= tolerance, f"max_abs_diff={error:.3e} psnr={psnr:.2f} dB tolerance={tolerance}"
