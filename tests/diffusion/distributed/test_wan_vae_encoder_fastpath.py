# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference, cache and installation tests for the Cosmos3 Wan encoder."""

from __future__ import annotations

import importlib
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanResample, WanRMS_norm
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    ENCODER_REPORT_ATTR,
    can_encode_frames,
    encode_frames,
    install_wan_vae_encoder_fastpath,
    install_wan_vae_fastpath,
    is_encoder_installed,
    is_installed,
    uninstall_wan_vae_encoder_fastpath,
    uninstall_wan_vae_fastpath,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import encoder_forwards as ef
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fp
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import install as installer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# Preserve all four stages and both temporal downsamplers, unlike the decoder's
# two-stage fixtures. Channel counts are reduced only to keep CPU tests small.
CONFIG = dict(
    base_dim=4,
    decoder_base_dim=4,
    z_dim=4,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=1,
    temperal_downsample=[False, True, True],
    is_residual=True,
    patch_size=2,
    in_channels=12,
    out_channels=12,
    scale_factor_temporal=4,
    scale_factor_spatial=16,
)


def pair(dtype=torch.float32):
    torch.manual_seed(0)
    ref = AutoencoderKLWan(**CONFIG).eval().to(dtype)
    fast = AutoencoderKLWan(**CONFIG).eval().to(dtype)
    fast.load_state_dict(ref.state_dict())
    return ref, fast


def bits_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    integer = torch.int16 if a.element_size() == 2 else torch.int32
    assert torch.equal(a.contiguous().view(integer), b.contiguous().view(integer))


def test_wan22_import_preserves_diffusers_encoder_norm():
    from diffusers.models.autoencoders import autoencoder_kl_wan

    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.is_npu():
        pytest.skip("Wan 2.2 intentionally replaces WanRMS_norm on NPU")

    importlib.import_module("vllm_omni.diffusion.models.wan2_2")
    assert autoencoder_kl_wan.WanRMS_norm is WanRMS_norm
    assert WanRMS_norm.__module__ == "diffusers.models.autoencoders.autoencoder_kl_wan"
    assert fp.is_diffusers_rms_norm(WanRMS_norm(8, images=False))


@torch.no_grad()
@pytest.mark.parametrize("frames", [1, 5, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_encoder_and_chunk_assembly_match_reference(frames, dtype):
    ref, fast = pair(dtype)
    assert install_wan_vae_encoder_fastpath(fast).installed
    x = torch.randn(1, 3, frames, 16, 32).to(dtype)
    expected = ref.encode(x).latent_dist
    actual = fast.encode(x).latent_dist
    bits_equal(actual.parameters, expected.parameters)
    bits_equal(actual.mode(), expected.mode())
    bits_equal(actual.logvar, expected.logvar)
    assert can_encode_frames(fast, x)
    assembled = encode_frames(fast, x)
    bits_equal(assembled, expected.parameters)
    assert assembled.stride() == expected.parameters.stride()
    bits_equal(encode_frames(fast, x), assembled)
    assert all(cache is None for cache in fast._enc_feat_map)


@torch.no_grad()
@pytest.mark.parametrize("autocast", [False, True])
def test_bf16_channels_last_encoder_uses_replacement_forwards(monkeypatch, autocast):
    dtype = torch.float32 if autocast else torch.bfloat16
    ref, fast = pair(dtype)
    # Compare against the same layout with unmodified reference forwards.
    for module in (*ref.encoder.modules(), ref.quant_conv):
        if isinstance(module, nn.Conv3d):
            module.to(memory_format=torch.channels_last_3d)
        elif isinstance(module, nn.Conv2d):
            module.to(memory_format=torch.channels_last)

    calls = []
    original = ef.encoder_forward

    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(ef, "encoder_forward", record)
    assert install_wan_vae_encoder_fastpath(fast, level="channels_last").installed
    x = torch.randn(1, 3, 5, 16, 32).to(dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        bits_equal(encode_frames(fast, x), ref.encode(x).latent_dist.parameters)
    assert calls
    assert fast.quant_conv.forward.__func__ is fp.causal_conv_forward
    uninstall_wan_vae_encoder_fastpath(fast)
    assert "forward" not in fast.encoder.__dict__
    assert "forward" not in fast.quant_conv.__dict__


@torch.no_grad()
def test_strided_input_and_batch_slicing():
    ref, fast = pair()
    install_wan_vae_encoder_fastpath(fast)
    x = torch.randn(2, 3, 9, 32, 64)[:, :, :, ::2, ::2]
    bits_equal(encode_frames(fast, x), ref.encode(x).latent_dist.parameters)
    ref.enable_slicing()
    fast.enable_slicing()
    expected = ref.encode(x, return_dict=False)[0]
    actual = fast.encode(x, return_dict=False)[0]
    bits_equal(actual.parameters, expected.parameters)
    assert not can_encode_frames(fast, x[:, :, :6])
    assert not can_encode_frames(fast, x[:, :, :, :15])


@torch.no_grad()
def test_tiled_encoder_preserves_reference_blending():
    ref, fast = pair()
    install_wan_vae_encoder_fastpath(fast)
    for vae in (ref, fast):
        vae.enable_tiling(
            tile_sample_min_height=32,
            tile_sample_min_width=32,
            tile_sample_stride_height=16,
            tile_sample_stride_width=16,
        )
    x = torch.randn(1, 3, 5, 48, 64)
    bits_equal(fast.encode(x).latent_dist.parameters, ref.encode(x).latent_dist.parameters)


@torch.no_grad()
def test_downsample_first_chunk_and_history(monkeypatch):
    module = WanResample(8, "downsample3d").eval()
    calls = []
    handle = module.time_conv.register_forward_pre_hook(lambda _m, args: calls.append(args[0].clone()))
    # Exercise the new call-site logic while the actual Triton wrappers decline CPU.
    monkeypatch.setattr(fp, "_kernels_allowed", lambda x: True)
    cache = [None]
    reference_cache = [None]
    for frames in (1, 4, 4):
        x = torch.randn(1, 8, frames, 8, 10)
        idx = [0]
        actual = ef.downsample_forward(module, x, cache, idx)
        calls.clear()
        expected = WanResample.forward(module, x, reference_cache, [0])
        bits_equal(actual, expected)
        bits_equal(cache[0], reference_cache[0])
        assert idx == [1]
        assert actual.shape[2] == (1 if frames == 1 else 2)
        assert len(calls) == (0 if frames == 1 else 1)
    handle.remove()


@torch.no_grad()
def test_lossless_downsample_retains_reference_for_channels_last_input(monkeypatch):
    module = WanResample(8, "downsample2d").eval()
    setattr(module, fp.CFG_ATTR, fp.FastPathConfig())
    x = torch.randn(1, 8, 1, 8, 10).contiguous(memory_format=torch.channels_last_3d)
    monkeypatch.setattr(fp, "_kernels_allowed", lambda _x: True)
    monkeypatch.setattr(
        ef.down, "spatial_downsample_input", lambda _x: pytest.fail("lossless must retain reference layout")
    )
    bits_equal(ef.downsample_forward(module, x), WanResample.forward(module, x))


@torch.no_grad()
def test_quant_conv_runs_once_after_all_chunks():
    _, fast = pair()
    install_wan_vae_encoder_fastpath(fast)
    seen = []
    handle = fast.quant_conv.register_forward_pre_hook(lambda _m, args: seen.append(args[0].shape[2]))
    encode_frames(fast, torch.randn(1, 3, 9, 16, 16))
    handle.remove()
    assert seen == [3]


@torch.no_grad()
def test_encode_exception_clears_cache():
    _, fast = pair()
    install_wan_vae_encoder_fastpath(fast)

    def fail(_module, _inputs, _output):
        raise RuntimeError("encoder failure")

    handle = fast.encoder.register_forward_hook(fail)
    with pytest.raises(RuntimeError, match="encoder failure"):
        encode_frames(fast, torch.randn(1, 3, 5, 16, 16))
    assert all(cache is None for cache in fast._enc_feat_map)
    handle.remove()
    encode_frames(fast, torch.randn(1, 3, 1, 16, 16))


@torch.no_grad()
def test_tile_exception_clears_cache():
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
    from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import TileTask

    _, vae = pair()
    install_wan_vae_encoder_fastpath(vae)

    def fail(_module, _inputs, _output):
        raise RuntimeError("tile failure")

    handle = vae.encoder.register_forward_hook(fail)
    task = TileTask(0, (0, 0), [torch.randn(1, 12, 1, 8, 8)])
    with pytest.raises(RuntimeError, match="tile failure"):
        DistributedAutoencoderKLWan.encode_tile_exec(vae, task)
    handle.remove()
    assert all(cache is None for cache in vae._enc_feat_map)


@pytest.mark.parametrize("encode_first", [False, True])
@torch.no_grad()
def test_encoder_decoder_installation_is_independent(encode_first):
    _, vae = pair()
    original_strides = {name: p.stride() for name, p in vae.named_parameters()}
    installers = [install_wan_vae_encoder_fastpath, install_wan_vae_fastpath]
    if not encode_first:
        installers.reverse()
    for install in installers:
        assert install(vae, level="channels_last").installed
    assert is_encoder_installed(vae) and is_installed(vae)
    assert getattr(vae.encoder, fp.CFG_ATTR).fuse_norm_cache
    assert not getattr(vae.decoder, fp.CFG_ATTR).fuse_norm_cache
    decoder_stride = vae.decoder.conv_in.weight.stride()
    vae.encoder.conv_in.weight.add_(1)
    updated = vae.encoder.conv_in.weight.clone()
    uninstall_wan_vae_encoder_fastpath(vae)
    assert not is_encoder_installed(vae) and is_installed(vae)
    assert vae.decoder.conv_in.weight.stride() == decoder_stride
    bits_equal(vae.encoder.conv_in.weight, updated)
    uninstall_wan_vae_fastpath(vae)
    assert original_strides == {name: p.stride() for name, p in vae.named_parameters()}
    assert "forward" not in vae.encoder.__dict__
    assert "forward" not in vae.quant_conv.__dict__


@torch.no_grad()
@pytest.mark.parametrize("encode_first", [False, True])
def test_encoder_installation_with_spatial_shard_decoder(encode_first, monkeypatch):
    from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard

    ref, vae = pair()
    vae.distributed_executor = SimpleNamespace(parallel_mode="spatial_shard_height")
    group = object()  # Installation is local; collectives start only at decode.
    monkeypatch.setattr(wan_spatial_shard, "_rank_world", lambda group: (0, 2))
    if encode_first:
        assert install_wan_vae_encoder_fastpath(vae).installed
    wan_spatial_shard.install_wan_spatial_shard_decode(vae, group)
    if not encode_first:
        assert install_wan_vae_encoder_fastpath(vae).installed
    x = torch.randn(1, 3, 5, 16, 32)
    bits_equal(encode_frames(vae, x), ref.encode(x).latent_dist.parameters)
    decoder_forward = vae.decoder.forward
    uninstall_wan_vae_encoder_fastpath(vae)
    assert vae.decoder.forward is decoder_forward
    assert vae._vllm_omni_wan_spatial_shard_installed
    assert not is_installed(vae)


@pytest.mark.parametrize("target", ["norm", "pad", "shortcut", "causal"])
def test_encoder_rejects_bypassed_hooks(target):
    _, vae = pair()
    block = vae.encoder.down_blocks[0]
    module = {
        "norm": block.resnets[0].norm1,
        "pad": block.downsampler.resample[0],
        "shortcut": block.avg_shortcut,
        "causal": vae.encoder.conv_in,
    }[target]
    handle = module.register_forward_hook(lambda _m, _i, out: out)
    report = install_wan_vae_encoder_fastpath(vae)
    handle.remove()
    assert not report.installed and "hooks" in report.reason
    assert "forward" not in vae.encoder.__dict__


@torch.no_grad()
def test_preserved_mutating_hook_retains_shortcut_clone():
    ref, fast = pair()
    handles = []
    for vae in (ref, fast):

        def mutate(_module, args):
            args[0].add_(0.125)

        handles.append(vae.encoder.down_blocks[0].resnets[0].register_forward_pre_hook(mutate))
    assert install_wan_vae_encoder_fastpath(fast).installed
    assert getattr(fast.encoder.down_blocks[0], fp.CFG_ATTR).clone_encoder_shortcuts
    x = torch.randn(1, 3, 5, 16, 16)
    bits_equal(fast.encode(x).latent_dist.parameters, ref.encode(x).latent_dist.parameters)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("failure_stage", ["binding", "conversion", "report"])
def test_encoder_installation_rollback(monkeypatch, failure_stage):
    _, vae = pair()
    assert install_wan_vae_fastpath(vae).installed
    originals = {name: (p, p.data, p.stride()) for name, p in vae.named_parameters()}

    if failure_stage == "conversion":

        def fail(modules, *, channels_last):
            modules[0].to(memory_format=torch.channels_last_3d)
            raise RuntimeError("injected failure")

        monkeypatch.setattr(installer, "_convert_conv_memory_format", fail)
    elif failure_stage == "binding":
        original_setattr = type(vae.encoder).__setattr__

        def fail_binding(module, name, value):
            original_setattr(module, name, value)
            if module is vae.encoder and name == fp.CFG_ATTR:
                raise RuntimeError("injected failure")

        monkeypatch.setattr(type(vae.encoder), "__setattr__", fail_binding)
    else:
        original_info = installer.logger.info

        def fail_report(message, *args, **kwargs):
            if message.startswith("Wan VAE encoder fast path (%s) installed"):
                assert is_encoder_installed(vae)
                raise RuntimeError("injected failure")
            original_info(message, *args, **kwargs)

        monkeypatch.setattr(installer.logger, "info", fail_report)
    with pytest.raises(RuntimeError, match="injected failure"):
        install_wan_vae_encoder_fastpath(vae, level="channels_last")
    assert not hasattr(vae, ENCODER_REPORT_ATTR)
    assert is_installed(vae)
    for name, parameter in vae.named_parameters():
        original, storage, strides = originals[name]
        assert parameter is original and parameter.data_ptr() == storage.data_ptr()
        assert parameter.stride() == strides
    assert "forward" not in vae.encoder.__dict__


def test_unsupported_architecture_and_custom_forward():
    _, vae = pair()
    assert not install_wan_vae_encoder_fastpath(vae, level="off").installed
    assert not install_wan_vae_encoder_fastpath(nn.Linear(2, 2)).installed
    vae.encoder.forward = lambda x: x
    report = install_wan_vae_encoder_fastpath(vae)
    assert not report.installed and "custom forward" in report.reason
    del vae.encoder.forward
    alias = MethodType(type(vae.encoder).forward, vae.encoder)
    vae.encoder.forward = alias
    assert install_wan_vae_encoder_fastpath(vae).installed
    assert install_wan_vae_encoder_fastpath(vae) is getattr(vae, ENCODER_REPORT_ATTR)
    uninstall_wan_vae_encoder_fastpath(vae)
    assert vae.encoder.forward is alias
    vae.register_to_config(is_residual=False)
    assert not install_wan_vae_encoder_fastpath(vae).installed
    with pytest.raises(ValueError, match="vae_encode_fast_path"):
        install_wan_vae_encoder_fastpath(vae, level="invalid")


def test_encoder_autograd_preserved():
    ref, fast = pair()
    install_wan_vae_encoder_fastpath(fast)
    x = torch.randn(1, 3, 1, 16, 16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    expected = ref.encode(x).latent_dist.mode()
    actual = fast.encode(other).latent_dist.mode()
    assert not can_encode_frames(fast, other)
    bits_equal(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    bits_equal(other.grad, x.grad)


@torch.no_grad()
def test_encode_frames_declines_compile(monkeypatch):
    _, vae = pair()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert not can_encode_frames(vae, torch.randn(1, 3, 5, 16, 16))


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_omni_encode_dispatch_and_autocast(dtype, monkeypatch):
    from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_wan as omni

    torch.manual_seed(4)
    vae = omni.OmniAutoencoderKLWan(**CONFIG).eval().to(dtype)
    pixels = torch.randn(2, 3, 5, 16, 32).to(dtype)
    vae.enable_slicing()
    expected = vae.encode(pixels).latent_dist.parameters
    assert install_wan_vae_encoder_fastpath(vae).installed
    calls = []
    original = omni.encode_frames

    def record(vae, pixels):
        calls.append(pixels.shape)
        return original(vae, pixels)

    monkeypatch.setattr(omni, "encode_frames", record)
    bits_equal(vae.encode(pixels).latent_dist.parameters, expected)
    assert len(calls) == 2 and all(shape[0] == 1 for shape in calls)
    calls.clear()
    # Off uses the original _encode, including its existing noncanonical-frame behavior.
    uninstall_wan_vae_encoder_fastpath(vae)
    bits_equal(vae.encode(pixels, return_dict=False)[0].parameters, expected)
    assert calls == []


@torch.no_grad()
def test_omni_tiling_keeps_parent_dispatch(monkeypatch):
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan

    vae = OmniAutoencoderKLWan(**CONFIG).eval()
    install_wan_vae_encoder_fastpath(vae)
    vae.enable_tiling(tile_sample_min_height=16, tile_sample_min_width=16)
    seen = []

    def tiled(x):
        seen.append(x.shape)
        return x

    monkeypatch.setattr(vae, "tiled_encode", tiled)
    vae._encode(torch.randn(1, 3, 5, 32, 48))
    assert seen == [torch.Size([1, 12, 5, 16, 24])]


@torch.no_grad()
def test_custom_normalization_keeps_its_numerics():
    class AlternateNorm(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.gamma = nn.Parameter(original.gamma.clone())
            self.scale = original.scale

        def forward(self, x):
            return F.normalize(x, dim=1, eps=1e-6) * self.scale * self.gamma

    ref, fast = pair(torch.bfloat16)
    for vae in (ref, fast):
        vae.encoder.norm_out = AlternateNorm(vae.encoder.norm_out)
    original = fast.encoder.norm_out.forward.__func__
    assert install_wan_vae_encoder_fastpath(fast).installed
    assert fast.encoder.norm_out.forward.__func__ is original
    x = torch.randn(1, 3, 5, 16, 16).to(torch.bfloat16)
    bits_equal(fast.encode(x).latent_dist.parameters, ref.encode(x).latent_dist.parameters)


def test_convolution_probes_are_scoped_to_backend_settings():
    conv = nn.Conv3d(8, 8, 3)
    x = torch.randn(1, 8, 1, 4, 4)
    original = fp._conv_verdict_key(conv, x, 1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert fp._conv_verdict_key(conv, x, 1) != original
    with torch.backends.cudnn.flags(allow_tf32=not torch.backends.cudnn.allow_tf32):
        assert fp._conv_verdict_key(conv, x, 1) != original
    assert fp._conv_verdict_key(conv, x, 2) != original
    assert fp._conv_verdict_key(conv, x, 1) == original


@torch.no_grad()
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("return_bias", [False, True])
@pytest.mark.parametrize("history", [None, "Rep", 1, 2])
def test_norm_cache_fusion_dispatch(monkeypatch, channels_last, return_bias, history):
    norm = WanRMS_norm(8, images=False).eval()
    conv = WanCausalConv3d(8, 8, 3, padding=1).eval()
    cfg = fp.FastPathConfig(
        fused_silu_dtypes=frozenset({torch.float32}), channels_last=channels_last, fuse_norm_cache=True
    )
    setattr(norm, fp.CFG_ATTR, cfg)
    x = torch.randn(2, 8, 1, 3, 5)
    bias = torch.randn(8) if channels_last else None
    payload = torch.randn(2, 8, history, 3, 5) if isinstance(history, int) else None
    cache = [payload if payload is not None else history]
    normalized = F.silu(norm(x if bias is None else fp._add_channel_bias(x, bias)))
    assembled = F.pad(
        normalized if payload is None else torch.cat((payload, normalized), dim=2),
        (0, 0, 0, 0, 2 if payload is None else 2 - payload.shape[2], 0),
    )
    next_cache = assembled[:, :, -2:].clone()
    expected = F.conv3d(F.pad(assembled, (1, 1, 1, 1)), conv.weight, None if return_bias else conv.bias)
    seen = []

    def fused(source, gamma, scale, previous, pad, **kwargs):
        assert source is x and gamma is norm.gamma and scale == norm.scale
        assert previous is payload and pad == 2
        assert kwargs == dict(channels_last=channels_last, silu=True, bias=bias)
        seen.append(True)
        return assembled, next_cache

    monkeypatch.setattr(fp, "_kernels_allowed", lambda _x: True)
    monkeypatch.setattr(fp.nc, "norm_act_cat_time", fused)
    monkeypatch.setattr(fp, "_norm_act", lambda *a, **kw: pytest.fail("normalized temporary must not be made"))
    monkeypatch.setattr(fp.dm, "cat_time_5d", lambda *a, **kw: pytest.fail("separate assembly must not run"))
    result = fp._run_norm_act_cached_conv(
        norm, nn.SiLU(), conv, x, cache, 0, pending_bias=bias, return_bias=return_bias, after_norm=nn.Dropout().eval()
    )
    actual, deferred = result if return_bias else (result, None)
    bits_equal(actual, expected)
    assert deferred is (conv.bias if return_bias else None)
    assert seen == [True] and cache[0] is next_cache


@torch.no_grad()
@pytest.mark.parametrize(
    "reason",
    ["disabled", "silu", "norm", "dropout_active", "dropout_hook", "dropout_custom", "padding", "kernel"],
)
def test_norm_cache_fusion_falls_back_in_order(monkeypatch, reason):
    norm = WanRMS_norm(8, images=False).eval()
    conv = WanCausalConv3d(8, 8, 3, padding=1).eval()
    setattr(
        norm,
        fp.CFG_ATTR,
        fp.FastPathConfig(
            fused_silu_dtypes=frozenset() if reason == "silu" else frozenset({torch.float32}),
            fuse_norm_cache=reason != "disabled",
        ),
    )
    if reason == "norm":
        norm = nn.Identity()
    dropout = nn.Dropout(0.0).eval()
    events = []
    if reason == "dropout_active":
        dropout.train().p = 0.5
    elif reason == "dropout_hook":
        dropout.register_forward_hook(lambda *_: events.append("hook"))
    elif reason == "dropout_custom":
        dropout.forward = lambda x: events.append("custom") or x
    x = torch.randn(1, 8, 1, 3, 5)
    cache = [torch.randn(1, 8, 2, 3, 5)]
    previous = cache[0]
    bias = torch.randn(8)
    if reason == "padding":
        fp._SPATIAL_PAD_VERDICTS[conv] = {fp._conv_verdict_key(conv, x, 2): False}

    def declined(*a, **kw):
        assert reason == "kernel", "ineligible fusion must decline before allocating or reducing"
        assert cache[0] is previous
        events.append("decline")
        return None

    def normalized(n, act, source, *, pending_bias):
        assert n is norm and source is x and pending_bias is bias
        events.append("norm")
        return source + 1

    def cached(c, source, slots, index, *, return_bias):
        assert c is conv and slots is cache and index == 0 and return_bias
        assert slots[0] is previous
        events.append("conv")
        return source, None

    monkeypatch.setattr(fp, "_kernels_allowed", lambda _x: True)
    monkeypatch.setattr(fp.nc, "norm_act_cat_time", declined)
    monkeypatch.setattr(fp, "_norm_act", normalized)
    monkeypatch.setattr(fp, "_run_cached_causal_conv", cached)
    fp._run_norm_act_cached_conv(
        norm, nn.SiLU(), conv, x, cache, 0, pending_bias=bias, after_norm=dropout, return_bias=True
    )
    assert events == (
        (["decline"] if reason == "kernel" else [])
        + ["norm"]
        + (["hook"] if reason == "dropout_hook" else ["custom"] if reason == "dropout_custom" else [])
        + ["conv"]
    )


def test_norm_cache_does_not_skip_global_dropout_hooks():
    dropout = nn.Dropout().eval()
    assert fp._can_bypass_dropout(dropout)
    handle = nn.modules.module.register_module_forward_hook(lambda *_: None)
    try:
        assert not fp._can_bypass_dropout(dropout)
    finally:
        handle.remove()


@torch.no_grad()
def test_norm_cache_kernel_declines_cpu_without_mutation():
    x = torch.randn(1, 8, 1, 3, 5)
    cache = torch.randn(1, 8, 2, 3, 5)
    previous = cache.clone()
    assert fp.nc.norm_act_cat_time(x, torch.ones(8), 8**0.5, cache, 2, channels_last=False, silu=True) is None
    bits_equal(cache, previous)
