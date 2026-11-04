# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HunyuanVideo-1.5 VAE decoder fast path: exact replicate pad, bitwise decode, installer contract."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import HunyuanVideo15RMS_norm
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion import registry as registry_module
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.hunyuan_video_15_vae_fastpath import (
    REPORT_ATTR,
    _fast_norm,
    install_hunyuan_video_15_vae_fastpath,
    rms_norm_forward,
)
from vllm_omni.diffusion.layers.replicate_pad import replicate_pad

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_CUDA = hardware_test(res={"cuda": "L4"}, num_cards=1)
_NEEDS_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
TINY = dict(latent_channels=4, block_out_channels=(8, 16, 16, 16, 16), layers_per_block=1)
HV15_PAD = (1, 1, 1, 1, 2, 0)


def _tiny_vae(dtype: torch.dtype = torch.float32, device: str = "cpu") -> AutoencoderKLHunyuanVideo15:
    torch.manual_seed(0)
    vae = AutoencoderKLHunyuanVideo15(**TINY).eval()
    for module in vae.modules():
        if isinstance(module, HunyuanVideo15RMS_norm):
            module.gamma.data.uniform_(0.5, 1.5)
    return vae.to(device=device, dtype=dtype)


def _decode(vae: AutoencoderKLHunyuanVideo15, z: torch.Tensor, tiling: bool) -> torch.Tensor:
    if tiling:
        vae.enable_tiling(
            tile_sample_min_height=64, tile_sample_min_width=64, tile_latent_min_height=4, tile_latent_min_width=4
        )
    with torch.no_grad():
        return vae.decode(z).sample


@_CUDA
@_NEEDS_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("pad", [HV15_PAD, (0, 0, 0, 0, 2, 0), (2, 1, 0, 3, 1, 2), (0,) * 6])
@pytest.mark.parametrize("layout", ["contiguous", "frame_major", "strided"])
def test_replicate_pad_matches_f_pad(dtype, pad, layout) -> None:
    x = torch.randn(2, 5, 4, 9, 7, device="cuda").to(dtype)
    if layout == "frame_major":
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == "strided":
        x = x[..., ::2]
    expected = F.pad(x, pad, mode="replicate")
    with torch.no_grad():
        actual = replicate_pad(x, pad)
    assert torch.equal(actual, expected)
    assert actual.stride() == expected.stride()


@_CUDA
@_NEEDS_CUDA
def test_replicate_pad_falls_back_under_grad() -> None:
    x = torch.randn(1, 3, 2, 4, 4, device="cuda", requires_grad=True)
    out = replicate_pad(x, HV15_PAD)
    assert out.requires_grad and torch.equal(out, F.pad(x, HV15_PAD, mode="replicate"))
    out.sum().backward()
    assert x.grad is not None


@_CUDA
@_NEEDS_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["contiguous", "frame_major", "sliced"])
def test_fast_norm_matches_reference(dtype, layout) -> None:
    torch.manual_seed(0)
    x = (torch.randn(1, 128, 33, 64, 65, device="cuda") * 3).to(dtype)
    if layout == "frame_major":
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == "sliced":  # non-dense: the reference's x.float() re-lays it out
        x = x[..., :64]
    norm = HunyuanVideo15RMS_norm(128, images=False).to("cuda", dtype)
    norm.gamma.data.uniform_(0.5, 1.5)
    with torch.no_grad():
        actual = rms_norm_forward(norm, x)
        expected = type(norm).forward(norm, x)
    assert torch.equal(actual, expected) and actual.stride() == expected.stride()


@_CUDA
@_NEEDS_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tiling", [False, True])
def test_installed_decode_is_bitwise(dtype, tiling) -> None:
    reference = _tiny_vae(dtype, "cuda")
    candidate = copy.deepcopy(reference)
    assert install_hunyuan_video_15_vae_fastpath(candidate).installed
    z = torch.randn(1, TINY["latent_channels"], 3, 6, 6, device="cuda", dtype=dtype)
    assert torch.equal(_decode(candidate, z, tiling), _decode(reference, z, tiling))


@pytest.mark.cpu
def test_fast_norm_declines_low_precision_inputs_past_32bit_reduction() -> None:
    norm = HunyuanVideo15RMS_norm(128, images=False).to(torch.bfloat16)
    x = torch.empty((1, 128, 65, 256, 256), dtype=torch.bfloat16, device="meta")  # fp32 copy > 2**31 bytes
    with torch.no_grad():
        assert _fast_norm(norm, x) is None


@pytest.mark.cpu
def test_report_and_idempotency() -> None:
    vae = _tiny_vae()
    reference = copy.deepcopy(vae)
    report = install_hunyuan_video_15_vae_fastpath(vae, level="channels_last")
    assert report.installed and report.level == "lossless" and report.fused_silu_dtypes == ()
    assert set(report.patched) == {"HunyuanVideo15CausalConv3d", "HunyuanVideo15ResnetBlock", "HunyuanVideo15RMS_norm"}
    assert install_hunyuan_video_15_vae_fastpath(vae) is report
    assert vae.encoder.conv_in.forward.__func__ is type(vae.encoder.conv_in).forward
    z = torch.randn(1, TINY["latent_channels"], 2, 4, 4)
    assert torch.equal(_decode(vae, z, False), _decode(reference, z, False))


@pytest.mark.cpu
def test_off_and_unsupported_modules_are_untouched() -> None:
    vae = _tiny_vae()
    assert install_hunyuan_video_15_vae_fastpath(vae, level="off").reason == "disabled"
    assert not hasattr(vae, REPORT_ATTR)
    report = install_hunyuan_video_15_vae_fastpath(nn.Linear(2, 2))
    assert not report.installed and "AutoencoderKLHunyuanVideo15" in report.reason
    vae.decoder.conv_out.forward = lambda x: x
    assert "custom forward" in install_hunyuan_video_15_vae_fastpath(vae).reason


@pytest.mark.cpu
@pytest.mark.parametrize(("level", "expected"), [("lossless", True), ("off", False)])
def test_registry_hook_installs_hunyuan_video_15(mocker, level, expected) -> None:
    vae = _tiny_vae()
    platform = mocker.Mock()
    platform.is_cuda.return_value = True
    mocker.patch.object(registry_module, "current_omni_platform", platform)
    pipeline = nn.Module()
    pipeline.vae = vae
    registry_module._apply_wan_vae_fastpath_if_enabled(pipeline, OmniDiffusionConfig(model="x", vae_fast_path=level))
    assert hasattr(vae, REPORT_ATTR) is expected
