# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA encoder parity/layout tests; run on both H100 and GB200."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D, WanResample

from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    encode_frames,
    install_wan_vae_encoder_fastpath,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import encoder_forwards as ef
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fp
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import triton_downsample as down
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import triton_norm_cache as nc
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath._utils import encoder_nrmse_limit

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
CONFIG = dict(
    base_dim=20,
    decoder_base_dim=32,
    z_dim=48,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    temperal_downsample=[False, True, True],
    is_residual=True,
    patch_size=2,
    in_channels=12,
    out_channels=12,
    scale_factor_temporal=4,
    scale_factor_spatial=16,
)


def bits_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    integer = torch.int16 if a.element_size() == 2 else torch.int32
    assert torch.equal(a.contiguous().view(integer), b.contiguous().view(integer))


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "frame_major", "strided", "channels_last_strided"])
@pytest.mark.parametrize("batch,frames", [(1, 1), (1, 4), (2, 4)])
def test_spatial_pad_matches_reference(dtype, layout, batch, frames):
    x = torch.randn(batch, 160, frames, 12, 20, device="cuda", dtype=dtype)
    if layout in ("channels_last", "channels_last_strided"):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        if layout == "channels_last_strided":
            x = x[:, :, :, 1::2, ::2]
    elif layout == "frame_major":
        x = x.permute(0, 2, 1, 3, 4).contiguous().permute(0, 2, 1, 3, 4)
    elif layout == "strided":
        x = x[:, :, :, 1::2, ::2]
    expected = F.pad(fp._merge_batch_and_frames(x), (0, 1, 0, 1))
    actual = down.spatial_downsample_input(x)
    assert actual is not None
    bits_equal(actual, expected)
    if layout.startswith("channels_last"):
        assert actual.is_contiguous(memory_format=torch.channels_last)
    else:
        assert actual.stride() == expected.stride()


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("frames", [1, 4])
@pytest.mark.parametrize("cin,cout,ft,fs", [(160, 160, 1, 2), (160, 320, 2, 2), (320, 640, 2, 2), (640, 640, 1, 1)])
def test_average_shortcuts_match_cosmos3_grouping(dtype, channels_last, frames, cin, cout, ft, fs):
    shortcut = AvgDown3D(cin, cout, ft, fs)
    source = torch.randn(2, cin, frames, 8, 12, device="cuda", dtype=dtype)
    if channels_last:
        source = source.contiguous(memory_format=torch.channels_last_3d)
    shape = (2, cout, (frames + ft - 1) // ft, 8 // fs, 12 // fs)
    main = torch.randn(shape, device="cuda", dtype=dtype)
    if channels_last:
        main = main.contiguous(memory_format=torch.channels_last_3d)
    expected = main + shortcut(source)
    actual = down.avg_down3d_add(main, source, ft, fs, shortcut.group_size)
    assert actual is not None
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d if channels_last else torch.contiguous_format)


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
def test_average_shortcut_edge_values_and_strides(dtype):
    # Zeros, tiny values, cancellation and temporal front padding. Values stay
    # finite so this also checks that masked temporal loads cannot leak NaNs.
    source = torch.zeros(1, 160, 1, 8, 12, device="cuda", dtype=dtype)
    source[:, ::4] = 1
    source[:, 1::4] = -1
    source[:, 2::4] = torch.finfo(dtype).tiny
    source = source[:, :, :, ::2, ::2]
    shortcut = AvgDown3D(160, 320, 2, 2)
    main = torch.zeros(1, 320, 1, 2, 3, device="cuda", dtype=dtype)
    actual = down.avg_down3d_add(main, source, 2, 2, 4)
    assert actual is not None and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, main + shortcut(source))
    assert down.avg_down3d_add(main, source, 2, 2, 3) is None


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels_last", [False, True])
def test_temporal_downsample_cache_across_fused_and_fallback_calls(monkeypatch, dtype, channels_last):
    module = WanResample(160, "downsample3d").eval().to(device="cuda", dtype=dtype)
    setattr(module, fp.CFG_ATTR, fp.FastPathConfig(channels_last=channels_last))
    if channels_last:
        module.resample[1].to(memory_format=torch.channels_last)
        module.time_conv.to(memory_format=torch.channels_last_3d)
    chunks = [torch.randn(1, 160, t, 12, 20, device="cuda", dtype=dtype) for t in (1, 4, 4, 4)]
    if channels_last:
        chunks = [x.contiguous(memory_format=torch.channels_last_3d) for x in chunks]
    reference_cache, cache = [None], [None]
    original = ef.dm.cat_time_5d
    for i, chunk in enumerate(chunks):
        monkeypatch.setattr(ef.dm, "cat_time_5d", (lambda *a, **k: None) if i == 2 else original)
        expected = WanResample.forward(module, chunk, reference_cache, [0])
        actual = ef.downsample_forward(module, chunk, cache, [0])
        bits_equal(actual, expected)
        bits_equal(cache[0], reference_cache[0])


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("frames", [1, 5, 9])
def test_lossless_encoder_posterior_and_output_assembly(dtype, frames):
    torch.manual_seed(0)
    ref = AutoencoderKLWan(**CONFIG).eval().to(device="cuda", dtype=dtype)
    fast = AutoencoderKLWan(**CONFIG).eval().to(device="cuda", dtype=dtype)
    fast.load_state_dict(ref.state_dict())
    assert install_wan_vae_encoder_fastpath(fast).installed
    x = torch.rand(1, 3, frames, 64, 96, device="cuda", dtype=dtype) * 2 - 1
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        expected = ref.encode(x).latent_dist
        actual = fast.encode(x).latent_dist
        bits_equal(actual.parameters, expected.parameters)
        bits_equal(actual.logvar, expected.logvar)
        bits_equal(actual.mode(), expected.mode())
        bits_equal(encode_frames(fast, x), expected.parameters)


@torch.no_grad()
@pytest.mark.parametrize(
    "dtype,tf32", [(torch.bfloat16, True), (torch.float16, True), (torch.float32, False), (torch.float32, True)]
)
def test_channels_last_encoder_posterior_and_layout(dtype, tf32):
    torch.manual_seed(1)
    ref = AutoencoderKLWan(**CONFIG).eval().to(device="cuda", dtype=dtype)
    fast = AutoencoderKLWan(**CONFIG).eval().to(device="cuda", dtype=dtype)
    fast.load_state_dict(ref.state_dict())
    assert install_wan_vae_encoder_fastpath(fast, level="channels_last").installed
    # Audit kernels directly; functional convolution calls bypass module hooks.
    source = torch.randn(1, 640, 4, 8, 12, device="cuda", dtype=dtype).contiguous(memory_format=torch.channels_last_3d)
    assert down.spatial_downsample_input(source).is_contiguous(memory_format=torch.channels_last)
    x = torch.rand(1, 3, 9, 64, 96, device="cuda", dtype=dtype) * 2 - 1
    with (
        torch.backends.cudnn.flags(allow_tf32=tf32),
        torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
    ):
        expected = ref.encode(x).latent_dist.parameters
        actual = encode_frames(fast, x)
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-8)
    nrmse = (error / scale).item()
    limit = encoder_nrmse_limit(dtype)
    if nrmse > limit:
        # Run ablations only on failure, with the same weights/input/backend flags.
        # Weight layout alone can change cuDNN arithmetic; distinguish that from
        # errors introduced by the replacement forwards and approximate kernels.
        layout_only = deepcopy(ref)
        for module in (*layout_only.encoder.modules(), layout_only.quant_conv):
            if isinstance(module, torch.nn.Conv3d):
                module.to(memory_format=torch.channels_last_3d)
            elif isinstance(module, torch.nn.Conv2d):
                module.to(memory_format=torch.channels_last)
        with (
            torch.backends.cudnn.flags(allow_tf32=tf32),
            torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
        ):
            layout_expected = layout_only.encode(x).latent_dist.parameters
            assert install_wan_vae_encoder_fastpath(layout_only, level="lossless").installed
            layout_lossless = encode_frames(layout_only, x)

        def relative_error(value, reference):
            rms = reference.float().square().mean().sqrt().clamp_min(1e-8)
            return ((value.float() - reference.float()).square().mean().sqrt() / rms).item()

        pytest.fail(
            f"channels_last NRMSE={nrmse:.8f} exceeds {limit:g}; "
            f"layout_only_vs_reference={relative_error(layout_expected, expected):.8f}; "
            f"lossless_with_cl_weights_vs_layout_only={relative_error(layout_lossless, layout_expected):.8f}; "
            f"fast_vs_layout_only={relative_error(actual, layout_expected):.8f}; "
            f"GPU={torch.cuda.get_device_name()}, torch={torch.__version__}, "
            f"CUDA={torch.version.cuda}, cuDNN={torch.backends.cudnn.version()}, dtype={dtype}, tf32={tf32}"
        )
    assert nrmse <= limit


@torch.no_grad()
@pytest.mark.parametrize("silu", [False, True])
def test_channels_last_bf16_preserves_normalization_rounding(silu):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    # Ones make the reduction exact regardless of its order. The non-power-of-two
    # width exposes rounding between normalization, scale, gamma, and SiLU.
    x = torch.ones(1, 160, 1, 3, 11, device="cuda", dtype=torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    norm = WanRMS_norm(160, images=False).to(device="cuda", dtype=x.dtype)
    norm.gamma.copy_(torch.linspace(-2, 2, 160, device="cuda", dtype=x.dtype).view_as(norm.gamma))
    setattr(norm, fp.CFG_ATTR, fp.FastPathConfig(channels_last=True))
    expected = F.silu(norm(x)) if silu else norm(x)
    bits_equal(fp.rms_norm_fastpath(norm, x, silu=silu), expected)
    actual, cache = nc.norm_act_cat_time(x, norm.gamma, norm.scale, None, 2, channels_last=True, silu=silu)
    bits_equal(actual, F.pad(expected, (0, 0, 0, 0, 2, 0)))
    bits_equal(cache, actual[:, :, -2:])


@torch.no_grad()
@pytest.mark.parametrize("channels", [160, 320, 640])
@pytest.mark.parametrize("dtype", DTYPES)
def test_production_width_normalization(channels, dtype):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    norm = WanRMS_norm(channels, images=False).to(device="cuda", dtype=dtype)
    x = torch.randn(1, channels, 4, 8, 12, device="cuda", dtype=dtype)
    setattr(norm, fp.CFG_ATTR, fp.FastPathConfig())
    bits_equal(fp.rms_norm_fastpath(norm, x), norm(x))
    setattr(norm, fp.CFG_ATTR, fp.FastPathConfig(channels_last=True))
    cl = x.contiguous(memory_format=torch.channels_last_3d)
    actual = fp.rms_norm_fastpath(norm, cl, silu=True)
    expected = F.silu(norm(cl))
    tol = 0.04 if dtype == torch.bfloat16 else 0.004 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "frame_major", "channels_last"])
@pytest.mark.parametrize("channels", [160, 320, 640])
@pytest.mark.parametrize("frames,history", [(1, 0), (1, 1), (1, 2), (2, 2), (4, 2)])
@pytest.mark.parametrize("silu", [False, True])
def test_fused_norm_cache_matches_existing_kernels(dtype, layout, channels, frames, history, silu):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    channels_last = layout == "channels_last"
    fmt = torch.channels_last_3d if channels_last else torch.contiguous_format
    # Odd spatial size exercises masked tiles; batch > 1 and sliced history
    # exercise storage offsets and noncanonical cache batch/channel strides.
    x = torch.randn(2, channels, frames, 3, 11, device="cuda", dtype=dtype)
    if channels_last:
        x = x.contiguous(memory_format=fmt)
    elif layout == "frame_major":
        x = x.permute(0, 2, 1, 3, 4).contiguous().permute(0, 2, 1, 3, 4)
    norm = WanRMS_norm(channels, images=False).to(device="cuda", dtype=dtype)
    norm.gamma.uniform_(-2, 2)
    setattr(norm, fp.CFG_ATTR, fp.FastPathConfig(channels_last=channels_last))
    bias = torch.randn(channels, device="cuda", dtype=dtype) if channels_last else None
    cache = None
    if history:
        cache = torch.randn(2, channels, history + 2, 3, 11, device="cuda", dtype=dtype).contiguous(memory_format=fmt)
        cache[:, ::3] = -0.0
        cache = cache[:, :, 1 : history + 1]
    previous = None if cache is None else cache.clone()
    normalized = fp.rms_norm_fastpath(norm, x, silu=silu, bias=bias)
    expected, expected_cache = fp.dm.cat_time_5d(normalized, cache, 2, keep_cache_frames=2)
    actual, actual_cache = nc.norm_act_cat_time(
        x, norm.gamma, norm.scale, cache, 2, channels_last=channels_last, silu=silu, bias=bias
    )
    bits_equal(actual, expected)
    bits_equal(actual_cache, expected_cache)
    if cache is not None:
        bits_equal(cache, previous)
    assert actual.is_contiguous(memory_format=fmt) and actual_cache.is_contiguous(memory_format=fmt)
    assert actual.untyped_storage().data_ptr() != actual_cache.untyped_storage().data_ptr()


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("pad", [0, 2])
def test_fused_norm_cache_edge_values(dtype, channels_last, pad):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    x = torch.zeros(1, 160, 1, 1, 2051, device="cuda", dtype=dtype)
    x[..., 1::4] = -0.0
    x[:, ::2, :, :, 2::4] = torch.finfo(dtype).tiny
    x[:, ::2, :, :, 3::4] = 1
    x[:, 1::2, :, :, 3::4] = -1
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    norm = WanRMS_norm(160, images=False).to(device="cuda", dtype=dtype)
    setattr(norm, fp.CFG_ATTR, fp.FastPathConfig(channels_last=channels_last))
    expected = fp.rms_norm_fastpath(norm, x, silu=True)
    expected, history = fp.dm.cat_time_5d(expected, None, pad, keep_cache_frames=2)
    actual, cache = nc.norm_act_cat_time(x, norm.gamma, norm.scale, None, pad, channels_last=channels_last, silu=True)
    bits_equal(actual, expected)
    bits_equal(cache, history)


@torch.no_grad()
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize(
    "invalid", ["gamma_dtype", "gamma_shape", "layout", "cache_dtype", "cache_length", "grad", "compile"]
)
def test_fused_norm_cache_declines_unsupported_inputs(monkeypatch, channels_last, invalid):
    fmt = torch.channels_last_3d if channels_last else torch.contiguous_format
    x = torch.randn(1, 160, 1, 4, 8, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=fmt)
    gamma = torch.ones(160, device="cuda", dtype=x.dtype)
    cache = torch.randn(1, 160, 2, 4, 8, device="cuda", dtype=x.dtype).contiguous(memory_format=fmt)
    if invalid == "gamma_dtype":
        gamma = gamma.float()
    elif invalid == "gamma_shape":
        gamma = gamma[:80]
    elif invalid == "layout":
        x = x[..., ::2]
    elif invalid == "cache_dtype":
        cache = cache.float()
    elif invalid == "cache_length":
        cache = torch.cat((cache, cache), dim=2)
    elif invalid == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    previous = cache.clone()
    with torch.set_grad_enabled(invalid == "grad"):
        assert nc.norm_act_cat_time(x, gamma, 160**0.5, cache, 2, channels_last=channels_last, silu=True) is None
    bits_equal(cache, previous)


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("level", ["lossless", "channels_last"])
@pytest.mark.parametrize("frames", [1, 9])
def test_encoder_norm_cache_fusion_adds_no_posterior_drift(monkeypatch, dtype, level, frames):
    torch.manual_seed(42)
    vae = AutoencoderKLWan(**CONFIG).eval().to(device="cuda", dtype=dtype)
    assert install_wan_vae_encoder_fastpath(vae, level=level).installed
    configs = [(m, getattr(m, fp.CFG_ATTR)) for m in vae.encoder.modules() if hasattr(m, fp.CFG_ATTR)]
    x = torch.rand(1, 3, frames, 64, 96, device="cuda", dtype=dtype) * 2 - 1
    for module, cfg in configs:
        setattr(module, fp.CFG_ATTR, replace(cfg, fuse_norm_cache=False))
    expected = encode_frames(vae, x)
    for module, cfg in configs:
        setattr(module, fp.CFG_ATTR, cfg)
    fused = nc.norm_act_cat_time
    hits = []

    def record(*args, **kwargs):
        pair = fused(*args, **kwargs)
        if pair is not None:
            hits.append(args[0].shape)
        return pair

    monkeypatch.setattr(nc, "norm_act_cat_time", record)
    actual = encode_frames(vae, x)
    bits_equal(actual, expected)
    if level == "channels_last" or dtype in configs[0][1].fused_silu_dtypes:
        assert hits, "parity must exercise the fused kernel, not only its fallback"


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("channels_last", [False, True])
def test_norm_cache_history_across_fusion_and_fallback(monkeypatch, dtype, channels_last):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanRMS_norm

    norm = WanRMS_norm(160, images=False).to(device="cuda", dtype=dtype)
    conv = WanCausalConv3d(160, 160, 3, padding=1).eval().to(device="cuda", dtype=dtype)
    if channels_last:
        conv.to(memory_format=torch.channels_last_3d)
    cfg = fp.FastPathConfig(fused_silu_dtypes=frozenset({dtype}), channels_last=channels_last, fuse_norm_cache=True)
    setattr(norm, fp.CFG_ATTR, cfg)
    cache, reference_cache = [None], [None]
    fused = nc.norm_act_cat_time
    for i, frames in enumerate((1, 1, 4, 1, 2)):
        x = torch.randn(1, 160, frames, 6, 10, device="cuda", dtype=dtype)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        bias = torch.randn(160, device="cuda", dtype=dtype) if channels_last else None
        normalized = fp._norm_act(norm, torch.nn.SiLU(), x, pending_bias=bias)
        expected = fp._run_cached_causal_conv(conv, normalized, reference_cache, 0)
        monkeypatch.setattr(nc, "norm_act_cat_time", (lambda *a, **kw: None) if i == 2 else fused)
        actual = fp._run_norm_act_cached_conv(norm, torch.nn.SiLU(), conv, x, cache, 0, pending_bias=bias)
        bits_equal(actual, expected)
        bits_equal(cache[0], reference_cache[0])
