# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Wan VAE decoder fast path installer and its exact PyTorch fallbacks."""

from __future__ import annotations

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
    block = WanResidualUpBlock(8, 8, num_res_blocks=1, temporal_upsample=True, up_flag=True).eval()
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


def test_rms_norm_fastpath_declines_tensor_bias_and_dtype_mismatch() -> None:
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    norm = WanRMS_norm(8, images=False, bias=True)
    assert fastpath_forwards.rms_norm_fastpath(norm, torch.randn(1, 8, 1, 4, 4)) is None

    norm = WanRMS_norm(8, images=False)
    assert fastpath_forwards.rms_norm_fastpath(norm, torch.randn(1, 8, 1, 4, 4).to(torch.bfloat16)) is None

    x = torch.randn(1, 8, 2, 4, 4)
    out = fastpath_forwards.rms_norm_fastpath(norm, x)
    assert out is not None and torch.equal(out, norm(x))


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
