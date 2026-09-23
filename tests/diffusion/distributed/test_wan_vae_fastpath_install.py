# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Wan VAE decoder fast path installer and its exact PyTorch fallbacks."""

from __future__ import annotations

import importlib
from functools import wraps
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from torch import nn

from vllm_omni.diffusion import registry as registry_module
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    REPORT_ATTR,
    decode_frames,
    install_wan_vae_fastpath,
    is_installed,
    uninstall_wan_vae_fastpath,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fastpath_forwards

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TINY_RESIDUAL = dict(
    base_dim=8,
    decoder_base_dim=8,
    z_dim=4,
    dim_mult=[1, 1],
    num_res_blocks=1,
    temperal_downsample=[False, True],
    is_residual=True,
)
TINY_RESIDUAL_PATCH2 = dict(TINY_RESIDUAL, patch_size=2, in_channels=12, out_channels=12)
TINY_WAN21 = dict(
    base_dim=8,
    z_dim=4,
    dim_mult=[1, 2],
    num_res_blocks=1,
    temperal_downsample=[False, True],
    is_residual=False,
)
CONFIGS = {
    "residual": TINY_RESIDUAL,
    "residual_patch2": TINY_RESIDUAL_PATCH2,
    "wan21": TINY_WAN21,
}


def _build_pair(config: dict, dtype: torch.dtype) -> tuple[AutoencoderKLWan, AutoencoderKLWan]:
    torch.manual_seed(0)
    reference = AutoencoderKLWan(**config).eval().to(dtype)
    candidate = AutoencoderKLWan(**config).eval().to(dtype)
    candidate.load_state_dict(reference.state_dict())
    return reference, candidate


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("frames", [1, 3])
def test_lossless_fallback_paths_are_bitwise_exact(config_name: str, dtype: torch.dtype, frames: int) -> None:
    """Without CUDA every kernel declines, so this exercises the restructured PyTorch paths."""
    config = CONFIGS[config_name]
    reference, candidate = _build_pair(config, dtype)
    report = install_wan_vae_fastpath(candidate, level="lossless")
    assert report.installed, report
    assert report.fused_silu_dtypes == ()

    torch.manual_seed(1)
    latents = torch.randn(1, config["z_dim"], frames, 6, 8).to(dtype)
    with torch.no_grad():
        expected = reference.decode(latents, return_dict=False)[0]
        actual = candidate.decode(latents, return_dict=False)[0]
    assert actual.stride() == expected.stride()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_frames_matches_reference_decode(config_name: str, dtype: torch.dtype) -> None:
    config = CONFIGS[config_name]
    reference, candidate = _build_pair(config, dtype)
    install_wan_vae_fastpath(candidate)
    for frames in (1, 2, 4):
        torch.manual_seed(2)
        latents = torch.randn(1, config["z_dim"], frames, 6, 8).to(dtype)
        with torch.no_grad():
            expected = reference._decode(latents, return_dict=False)[0]
            actual = decode_frames(candidate, latents)
        assert actual.stride() == expected.stride()
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_omni_wan_vae_decode_override_is_bitwise_exact(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    reference = AutoencoderKLWan(**TINY_RESIDUAL_PATCH2).eval().to(dtype)
    candidate = OmniAutoencoderKLWan(**TINY_RESIDUAL_PATCH2).eval().to(dtype)
    candidate.load_state_dict(reference.state_dict())
    latents = torch.randn(1, 4, 3, 6, 8).to(dtype)
    with torch.no_grad():
        expected = reference.decode(latents, return_dict=False)[0]
        before_install = candidate.decode(latents, return_dict=False)[0]
        install_wan_vae_fastpath(candidate)
        actual = candidate.decode(latents, return_dict=False)[0]
        as_dict = candidate.decode(latents).sample
    # The Omni wrapper runs bf16 decode under autocast (also on CPU), so compare
    # the override against the same wrapper before installation, and against the
    # plain diffusers decode only where no autocast is involved.
    assert actual.stride() == before_install.stride()
    assert torch.equal(actual, before_install)
    assert torch.equal(as_dict, before_install)
    if dtype is torch.float32:
        assert torch.equal(before_install, expected)

    # Tiling still dispatches to the diffusers implementation.
    candidate.use_tiling = True
    candidate.tile_sample_min_height = candidate.tile_sample_min_width = 16
    candidate.tile_sample_stride_height = candidate.tile_sample_stride_width = 16
    reference.use_tiling = True
    reference.tile_sample_min_height = reference.tile_sample_min_width = 16
    reference.tile_sample_stride_height = reference.tile_sample_stride_width = 16
    with torch.no_grad():
        expected_tiled = reference.decode(latents, return_dict=False)[0]
        actual_tiled = candidate.decode(latents, return_dict=False)[0]
    if dtype is torch.float32:
        assert torch.equal(actual_tiled, expected_tiled)
    assert actual_tiled.shape == expected_tiled.shape


def test_omni_wan_vae_decode_preserves_autograd() -> None:
    torch.manual_seed(0)
    reference = AutoencoderKLWan(**TINY_RESIDUAL).eval()
    candidate = OmniAutoencoderKLWan(**TINY_RESIDUAL).eval()
    candidate.load_state_dict(reference.state_dict())
    install_wan_vae_fastpath(candidate)

    reference_latents = torch.randn(1, 4, 2, 6, 8, requires_grad=True)
    candidate_latents = reference_latents.detach().clone().requires_grad_()
    expected = reference.decode(reference_latents, return_dict=False)[0]
    actual = candidate.decode(candidate_latents, return_dict=False)[0]

    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(candidate_latents.grad, reference_latents.grad)


def test_install_is_idempotent_and_reversible() -> None:
    _, vae = _build_pair(TINY_RESIDUAL, torch.float32)
    keys_before = set(vae.state_dict())
    report = install_wan_vae_fastpath(vae)
    assert report.installed and is_installed(vae)
    assert report.patched["WanDecoder3d"] == 1
    assert report.patched["WanCausalConv3d"] > 1
    assert report.patched["WanRMS_norm"] > 1
    assert set(vae.state_dict()) == keys_before
    assert "forward" in vae.decoder.__dict__

    assert install_wan_vae_fastpath(vae) is report
    assert install_wan_vae_fastpath(vae, level="channels_last") is report  # warns, keeps the first level

    uninstall_wan_vae_fastpath(vae)
    assert not is_installed(vae)
    assert "forward" not in vae.decoder.__dict__
    assert all("forward" not in module.__dict__ for module in vae.decoder.modules())
    assert all(fastpath_forwards.CFG_ATTR not in module.__dict__ for module in vae.decoder.modules())


@torch.no_grad()
@pytest.mark.parametrize("failure_stage", ["binding", "conversion", "report"])
def test_failed_install_restores_original_state(failure_stage: str, monkeypatch) -> None:
    installer = importlib.import_module(install_wan_vae_fastpath.__module__)
    _, vae = _build_pair(TINY_RESIDUAL, torch.float32)
    # Preserve mixed original layouts, an existing forward, gradients and a
    # buffer: Module.to can replace tensors as well as change their storage.
    vae.decoder.conv_in.to(memory_format=torch.channels_last_3d)
    vae.decoder.forward = vae.decoder.forward
    setattr(vae.decoder, fastpath_forwards.CFG_ATTR, object())
    vae.decoder.conv_out.weight.grad = torch.randn_like(vae.decoder.conv_out.weight)
    vae.post_quant_conv.register_buffer("rollback_probe", torch.randn(2, 3, 4, 5, 6))
    attributes = {
        module: {
            name: module.__dict__[name] for name in ("forward", fastpath_forwards.CFG_ATTR) if name in module.__dict__
        }
        for module in vae.modules()
    }
    parameters = dict(vae.named_parameters())
    buffers = dict(vae.named_buffers())
    gradients = {name: parameter.grad for name, parameter in parameters.items()}
    tensors = [*parameters.values(), *buffers.values(), *(grad for grad in gradients.values() if grad is not None)]
    originals = [(tensor, tensor.data_ptr(), tensor.stride(), tensor.clone()) for tensor in tensors]
    latents = torch.randn(1, 4, 2, 6, 8)
    expected = vae.decode(latents, return_dict=False)[0]

    with monkeypatch.context() as patch:
        if failure_stage == "binding":
            conv_type = type(vae.post_quant_conv)
            original_setattr = conv_type.__setattr__

            def fail_binding(module, name, value):
                original_setattr(module, name, value)
                if module is vae.post_quant_conv and name == fastpath_forwards.CFG_ATTR:
                    raise RuntimeError("injected installation failure")

            patch.setattr(conv_type, "__setattr__", fail_binding)
        elif failure_stage == "conversion":
            original_to = vae.post_quant_conv.to

            def fail_conversion(*args, **kwargs):
                original_to(*args, **kwargs)
                assert vae.decoder.conv_out.weight.is_contiguous(memory_format=torch.channels_last_3d)
                raise torch.OutOfMemoryError("injected installation failure")

            patch.setattr(vae.post_quant_conv, "to", fail_conversion)
        else:
            original_info = installer.logger.info

            def fail_report(message, *args, **kwargs):
                if message.startswith("Wan VAE fast path (%s) installed"):
                    assert is_installed(vae)
                    raise RuntimeError("injected installation failure")
                original_info(message, *args, **kwargs)

            patch.setattr(installer.logger, "info", fail_report)

        with pytest.raises(RuntimeError, match="injected installation failure"):
            install_wan_vae_fastpath(vae, level="channels_last")

    assert not is_installed(vae)
    assert not hasattr(vae, REPORT_ATTR)
    assert not hasattr(vae, installer._UNDO_ATTR)
    for module, original in attributes.items():
        actual = {
            name: module.__dict__[name] for name in ("forward", fastpath_forwards.CFG_ATTR) if name in module.__dict__
        }
        assert actual == original
    assert all(dict(vae.named_parameters())[name] is parameter for name, parameter in parameters.items())
    assert all(dict(vae.named_buffers())[name] is buffer for name, buffer in buffers.items())
    assert all(parameters[name].grad is grad for name, grad in gradients.items())
    for tensor, pointer, strides, values in originals:
        assert tensor.data_ptr() == pointer
        assert tensor.stride() == strides
        assert torch.equal(tensor, values)
    uninstall_wan_vae_fastpath(vae)  # Safe and unnecessary after automatic rollback.
    assert torch.equal(vae.decode(latents, return_dict=False)[0], expected)
    assert install_wan_vae_fastpath(vae, level="channels_last").installed
    uninstall_wan_vae_fastpath(vae)


def test_load_state_dict_after_install_updates_patched_modules() -> None:
    reference, candidate = _build_pair(TINY_RESIDUAL, torch.float32)
    install_wan_vae_fastpath(candidate)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.mul_(0.5)
    candidate.load_state_dict(reference.state_dict())
    latents = torch.randn(1, 4, 2, 6, 8)
    with torch.no_grad():
        expected = reference.decode(latents, return_dict=False)[0]
        actual = candidate.decode(latents, return_dict=False)[0]
    assert torch.equal(actual, expected)


def test_channels_last_level_converts_conv_weights_and_restores_them() -> None:
    reference, vae = _build_pair(TINY_RESIDUAL, torch.float32)
    report = install_wan_vae_fastpath(vae, level="channels_last")
    assert report.installed and report.channels_last
    conv3d = [m for m in vae.decoder.modules() if isinstance(m, nn.Conv3d)]
    conv2d = [m for m in vae.decoder.modules() if isinstance(m, nn.Conv2d)]
    assert conv3d and conv2d
    assert all(m.weight.is_contiguous(memory_format=torch.channels_last_3d) for m in conv3d)
    assert all(m.weight.is_contiguous(memory_format=torch.channels_last) for m in conv2d)
    assert vae.post_quant_conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
    latents = torch.randn(1, 4, 2, 6, 8)
    with torch.no_grad():
        expected = reference.decode(latents, return_dict=False)[0]
        out = vae.decode(latents, return_dict=False)[0]
    # One temporal upsampler: 1 + 2 output frames for 2 latent frames.
    assert expected.shape == (1, 3, 3, 12, 16)
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    uninstall_wan_vae_fastpath(vae)
    assert all(m.weight.is_contiguous() for m in conv3d + conv2d)


@torch.no_grad()
@pytest.mark.parametrize("assign", [False, True])
def test_uninstall_restores_mixed_layouts_and_keeps_weight_updates(assign: bool) -> None:
    _, vae = _build_pair(TINY_RESIDUAL, torch.float32)
    convs = [module for module in vae.decoder.modules() if isinstance(module, (nn.Conv2d, nn.Conv3d))]
    convs.append(vae.post_quant_conv)
    for index, conv in enumerate(convs):
        if index % 3 == 1:
            conv.to(memory_format=torch.channels_last_3d if isinstance(conv, nn.Conv3d) else torch.channels_last)
        elif index % 3 == 2:
            conv.weight.data = conv.weight.data.transpose(-1, -2)
    vae.post_quant_conv.register_buffer("layout_probe", torch.randn(2, 3, 4, 5, 6).transpose(-1, -2))
    vae.decoder.conv_out.weight.grad = torch.randn_like(vae.decoder.conv_out.weight)
    grad_strides = vae.decoder.conv_out.weight.grad.stride()
    strides = {name: tensor.stride() for name, tensor in vae.state_dict().items()}
    # An explicit alias is safe to replace, but must remain an instance
    # attribute after uninstall, with its exact original identity.
    original_forward = vae.decoder.forward
    vae.decoder.forward = original_forward
    original_cfg = object()
    setattr(vae.decoder, fastpath_forwards.CFG_ATTR, original_cfg)

    assert install_wan_vae_fastpath(vae, level="channels_last").installed
    updated = {name: tensor.contiguous() + 0.25 for name, tensor in vae.state_dict().items()}
    vae.load_state_dict(updated, assign=assign)
    uninstall_wan_vae_fastpath(vae)

    assert vae.decoder.__dict__["forward"] is original_forward
    assert getattr(vae.decoder, fastpath_forwards.CFG_ATTR) is original_cfg
    for name, tensor in vae.state_dict().items():
        assert tensor.stride() == strides[name], name
        assert torch.equal(tensor, updated[name]), name
    if not assign:
        assert vae.decoder.conv_out.weight.grad.stride() == grad_strides


@torch.no_grad()
@pytest.mark.parametrize("level", ["lossless", "channels_last"])
@pytest.mark.parametrize(
    "target",
    [
        "decoder",
        "decoder.conv_in",
        "decoder.up_blocks.0.upsampler",
        "decoder.up_blocks.0.upsampler.resample",
        "decoder.up_blocks.0.upsampler.resample.1",
        "decoder.up_blocks.0.avg_shortcut",
        "decoder.norm_out",
        "decoder.nonlinearity",
        "post_quant_conv",
    ],
)
def test_installer_declines_forward_wrappers_it_would_bypass(target: str, level: str) -> None:
    _, vae = _build_pair(TINY_RESIDUAL, torch.float32)
    module = vae.get_submodule(target)
    original_forward = module.forward
    calls = []

    @wraps(original_forward)
    def wrapped(*args, **kwargs):
        calls.append(True)
        return original_forward(*args, **kwargs)

    module.forward = wrapped
    originals = {name: (tensor.data_ptr(), tensor.stride()) for name, tensor in vae.state_dict().items()}
    report = install_wan_vae_fastpath(vae, level=level)
    assert not report.installed and target in report.reason and "custom forward" in report.reason
    assert not is_installed(vae) and not hasattr(vae, REPORT_ATTR)
    assert all(fastpath_forwards.CFG_ATTR not in child.__dict__ for child in vae.modules())
    assert {name: (tensor.data_ptr(), tensor.stride()) for name, tensor in vae.state_dict().items()} == originals
    uninstall_wan_vae_fastpath(vae)
    assert module.forward is wrapped
    vae.decode(torch.randn(1, 4, 2, 6, 8))
    assert calls


@torch.no_grad()
@pytest.mark.parametrize("pre_hook", [False, True])
def test_installer_declines_convolution_hooks_it_would_bypass(pre_hook: bool) -> None:
    _, vae = _build_pair(TINY_RESIDUAL, torch.float32)
    conv = vae.decoder.conv_in
    calls = []

    def hook(*args):
        calls.append(True)

    handle = conv.register_forward_pre_hook(hook) if pre_hook else conv.register_forward_hook(hook)
    try:
        report = install_wan_vae_fastpath(vae, level="channels_last")
        assert not report.installed and "decoder.conv_in" in report.reason and "forward hooks" in report.reason
        vae.decode(torch.randn(1, 4, 2, 6, 8))
        assert calls
    finally:
        handle.remove()


@torch.no_grad()
def test_installer_preserves_wrappers_and_hooks_on_modules_it_still_calls() -> None:
    _, vae = _build_pair(TINY_RESIDUAL, torch.float32)
    original_forward = vae.decoder.mid_block.forward
    wrapper_calls = []
    hook_calls = []

    def wrapped(*args, **kwargs):
        wrapper_calls.append(True)
        return original_forward(*args, **kwargs)

    def hook(*args):
        hook_calls.append(True)

    vae.decoder.mid_block.forward = wrapped
    handle = vae.decoder.register_forward_hook(hook)
    try:
        latents = torch.randn(1, 4, 2, 6, 8)
        expected = vae.decode(latents).sample
        wrapper_calls.clear()
        hook_calls.clear()
        assert install_wan_vae_fastpath(vae).installed
        actual = vae.decode(latents).sample
        assert torch.equal(actual, expected)
        assert len(wrapper_calls) == len(hook_calls) == 2
        uninstall_wan_vae_fastpath(vae)
        assert vae.decoder.mid_block.forward is wrapped
    finally:
        handle.remove()


def test_installer_refuses_unsupported_targets() -> None:
    _, vae = _build_pair(TINY_RESIDUAL, torch.float32)

    assert install_wan_vae_fastpath(vae, level="off").installed is False
    assert not is_installed(vae)

    vae._vllm_omni_wan_spatial_shard_installed = True
    report = install_wan_vae_fastpath(vae)
    assert not report.installed and "spatial-shard" in report.reason
    del vae._vllm_omni_wan_spatial_shard_installed

    vae.distributed_executor = SimpleNamespace(parallel_mode="spatial_shard_height")
    report = install_wan_vae_fastpath(vae)
    assert not report.installed and "spatial_shard_height" in report.reason
    del vae.distributed_executor

    report = install_wan_vae_fastpath(nn.Linear(2, 2))
    assert not report.installed and "AutoencoderKLWan" in report.reason

    with pytest.raises(ValueError, match="vae_fast_path"):
        install_wan_vae_fastpath(vae, level="bogus")


@pytest.mark.parametrize(("batch", "frames"), [(1, 1), (1, 2), (2, 1)])
def test_resample_views_keep_channels_last_recognizable(batch: int, frames: int) -> None:
    """Regression: ``reshape`` gives a size-1 batch dim a stride the layout heuristic rejects."""
    import torch.nn.functional as F

    x = torch.randn(batch, 8, frames, 6, 10).contiguous(memory_format=torch.channels_last_3d)
    merged = fastpath_forwards._merge_batch_and_frames(x)
    reference = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, 8, 6, 10)
    assert merged.data_ptr() == x.data_ptr()
    assert torch.equal(merged, reference)
    upsampled = F.interpolate(merged, scale_factor=(2.0, 2.0), mode="nearest-exact")
    assert upsampled.stride(1) == 1, "nearest upsample must keep channels_last for the following Conv2d"
    assert torch.equal(upsampled, F.interpolate(reference.contiguous(), scale_factor=(2.0, 2.0), mode="nearest-exact"))

    split = fastpath_forwards._split_batch_and_frames(upsampled, batch, frames)
    assert split.shape == (batch, 8, frames, 12, 20)
    assert split.data_ptr() == upsampled.data_ptr()
    assert split.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(split, upsampled.view(batch, frames, 8, 12, 20).permute(0, 2, 1, 3, 4))

    plain = torch.randn(batch, 8, frames, 6, 10)
    assert torch.equal(
        fastpath_forwards._merge_batch_and_frames(plain), plain.permute(0, 2, 1, 3, 4).reshape(-1, 8, 6, 10)
    )


@torch.no_grad()
def test_pending_conv_bias_is_added_exactly_when_no_kernel_consumes_it() -> None:
    """Without the channels-last kernel the un-added conv bias is applied with ATen's rounding."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        norm = WanRMS_norm(8, images=False).to(dtype)
        act = nn.SiLU()
        x = torch.randn(1, 8, 2, 4, 4).to(dtype)
        bias = torch.randn(8).to(dtype)
        biased = x.clone()
        biased.add_(bias.view(1, -1, 1, 1, 1))
        expected = act(norm(biased))
        assert torch.equal(fastpath_forwards._norm_act(norm, act, x, pending_bias=bias), expected)
        setattr(norm, fastpath_forwards.CFG_ATTR, fastpath_forwards.FastPathConfig(channels_last=True))
        assert torch.equal(fastpath_forwards._norm_act(norm, act, x, pending_bias=bias), expected)
        out = fastpath_forwards.rms_norm_fastpath(norm, x, bias=bias)
        assert out is not None and torch.equal(out, norm(biased))


@torch.no_grad()
def test_resample_return_bias_is_none_without_kernels() -> None:
    """On CPU the Conv2d keeps its bias, so the up block adds nothing twice."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample, WanResidualUpBlock

    torch.manual_seed(0)
    resample = WanResample(8, mode="upsample2d").eval()
    assert fastpath_forwards._is_upsample_conv_pair(resample.resample)
    assert not fastpath_forwards._is_upsample_conv_pair(WanResample(8, mode="downsample2d").resample)
    x = torch.randn(1, 8, 2, 6, 10)
    out, bias = fastpath_forwards.resample_forward(resample, x, [None], [0], return_bias=True)
    assert bias is None
    assert torch.equal(out, WanResample.forward(resample, x, feat_cache=[None], feat_idx=[0]))

    # First chunk of a temporal up block: one input frame, the time conv is skipped ("Rep").
    block = WanResidualUpBlock(8, 8, num_res_blocks=1, temperal_upsample=True, up_flag=True).eval()
    for module in block.modules():
        setattr(module, fastpath_forwards.CFG_ATTR, fastpath_forwards.FastPathConfig())
    x = torch.randn(1, 8, 1, 6, 10)
    cache_len = 6
    expected = WanResidualUpBlock.forward(block, x, feat_cache=[None] * cache_len, feat_idx=[0], first_chunk=True)
    actual = fastpath_forwards.residual_up_block_forward(
        block, x, feat_cache=[None] * cache_len, feat_idx=[0], first_chunk=True
    )
    assert torch.equal(actual, expected)


def test_upsample_forward_only_fuses_nearest_2x() -> None:
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanUpsample

    assert fastpath_forwards._is_nearest_2x(WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"))
    assert fastpath_forwards._is_nearest_2x(WanUpsample(scale_factor=2, mode="nearest"))
    assert not fastpath_forwards._is_nearest_2x(WanUpsample(scale_factor=(2.0, 3.0), mode="nearest-exact"))
    assert not fastpath_forwards._is_nearest_2x(WanUpsample(scale_factor=(2.0, 2.0), mode="bilinear"))
    assert not fastpath_forwards._is_nearest_2x(WanUpsample(size=(12, 20), mode="nearest-exact"))

    module = WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact")
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(3, 8, 6, 10).to(dtype)
        for layout in (torch.contiguous_format, torch.channels_last):
            x = x.contiguous(memory_format=layout)
            expected = WanUpsample.forward(module, x)
            actual = fastpath_forwards.upsample_forward(module, x)
            assert actual.stride() == expected.stride()
            assert torch.equal(actual, expected)


def test_rms_norm_vae_substitute_is_not_matched() -> None:
    from vllm_omni.diffusion.layers.norm import RMSNormVAE

    assert not fastpath_forwards.is_diffusers_rms_norm(RMSNormVAE(8, images=False))
    _, vae = _build_pair(TINY_RESIDUAL, torch.float32)
    norm = vae.decoder.norm_out
    assert fastpath_forwards.is_diffusers_rms_norm(norm)


@torch.no_grad()
def test_rms_norm_fastpath_declines_tensor_bias_and_dtype_mismatch() -> None:
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    norm = WanRMS_norm(8, images=False, bias=True)
    assert fastpath_forwards.rms_norm_fastpath(norm, torch.randn(1, 8, 1, 4, 4)) is None

    norm = WanRMS_norm(8, images=False)
    assert fastpath_forwards.rms_norm_fastpath(norm, torch.randn(1, 8, 1, 4, 4).to(torch.bfloat16)) is None

    x = torch.randn(1, 8, 2, 4, 4)
    out = fastpath_forwards.rms_norm_fastpath(norm, x)
    assert out is not None and torch.equal(out, norm(x))


def test_rms_norm_forward_preserves_autograd() -> None:
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    torch.manual_seed(0)
    reference = WanRMS_norm(8, images=False)
    candidate = WanRMS_norm(8, images=False)
    candidate.load_state_dict(reference.state_dict())
    reference_input = torch.randn(1, 8, 1, 4, 4, requires_grad=True)
    candidate_input = reference_input.detach().clone().requires_grad_()

    expected = reference(reference_input)
    assert fastpath_forwards.rms_norm_fastpath(candidate, candidate_input) is None
    actual = fastpath_forwards.rms_norm_forward(candidate, candidate_input)
    torch.testing.assert_close(actual, expected)

    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(candidate_input.grad, reference_input.grad)
    torch.testing.assert_close(candidate.gamma.grad, reference.gamma.grad)


def test_omni_diffusion_config_validates_vae_fast_path() -> None:
    assert OmniDiffusionConfig(model="x").vae_fast_path == "lossless"
    assert OmniDiffusionConfig(model="x", vae_fast_path="channels_last").vae_fast_path == "channels_last"
    with pytest.raises(ValueError, match="vae_fast_path"):
        OmniDiffusionConfig(model="x", vae_fast_path="fast")


class _StubPipeline(nn.Module):
    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae


@pytest.mark.parametrize(("level", "expected"), [("lossless", True), ("channels_last", True), ("off", False)])
def test_registry_hook_installs_on_cuda_platform(mocker, level: str, expected: bool) -> None:
    _, vae = _build_pair(TINY_RESIDUAL, torch.float32)
    platform = mocker.Mock()
    platform.is_cuda.return_value = True
    mocker.patch.object(registry_module, "current_omni_platform", platform)

    registry_module._apply_wan_vae_fastpath_if_enabled(_StubPipeline(vae), SimpleNamespace(vae_fast_path=level))
    assert is_installed(vae) is expected
    if expected:
        assert getattr(vae, REPORT_ATTR).level == level


def test_registry_hook_skips_non_cuda_platform_and_non_wan_vaes(mocker) -> None:
    _, vae = _build_pair(TINY_RESIDUAL, torch.float32)
    platform = mocker.Mock()
    platform.is_cuda.return_value = False
    mocker.patch.object(registry_module, "current_omni_platform", platform)
    registry_module._apply_wan_vae_fastpath_if_enabled(_StubPipeline(vae), SimpleNamespace(vae_fast_path="lossless"))
    assert not is_installed(vae)

    platform.is_cuda.return_value = True
    other = nn.Linear(2, 2)
    registry_module._apply_wan_vae_fastpath_if_enabled(_StubPipeline(other), SimpleNamespace(vae_fast_path="lossless"))
    assert not hasattr(other, REPORT_ATTR)
