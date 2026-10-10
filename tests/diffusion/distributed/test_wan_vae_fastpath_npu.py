# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend exactness tests for Wan VAE causal-Conv3D spatial padding."""

from __future__ import annotations

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fp
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import install_wan_vae_fastpath

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

TINY_VAE_CONFIG = dict(
    base_dim=8,
    decoder_base_dim=8,
    z_dim=4,
    dim_mult=[1, 1],
    num_res_blocks=1,
    temperal_downsample=[False, True],
    is_residual=True,
)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_installer_limits_npu_fast_path_to_causal_conv() -> None:
    vae = AutoencoderKLWan(**TINY_VAE_CONFIG).eval().to(device="npu", dtype=torch.float32)
    report = install_wan_vae_fastpath(vae, level="lossless")

    assert report.installed, report.reason
    assert set(report.patched) == {"WanCausalConv3d"}
    assert report.patched["WanCausalConv3d"] > 1


@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "in_channels,out_channels,frames,height,width,cache_frames",
    [
        (3, 5, 4, 7, 9, 0),
        (3, 5, 4, 7, 9, 2),
        (16, 32, 1, 48, 80, 0),
        (16, 32, 1, 48, 80, 2),
        (128, 128, 1, 48, 80, 0),
        (128, 128, 1, 48, 80, 2),
    ],
)
@torch.no_grad()
def test_causal_conv_internal_spatial_padding_is_bitwise_exact(
    dtype: torch.dtype,
    in_channels: int,
    out_channels: int,
    frames: int,
    height: int,
    width: int,
    cache_frames: int,
) -> None:
    torch.manual_seed(1101)
    conv = WanCausalConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1).to(device="npu", dtype=dtype)
    x = torch.randn(1, in_channels, frames, height, width, device="npu", dtype=dtype)
    cache = (
        None
        if cache_frames == 0
        else torch.randn(1, in_channels, cache_frames, height, width, device="npu", dtype=dtype)
    )

    expected = WanCausalConv3d.forward(conv, x, cache)
    actual = fp.causal_conv_forward(conv, x, cache)

    assert torch.equal(actual, expected)
    verdicts = fp._SPATIAL_PAD_VERDICTS[conv]
    assert len(verdicts) == 1
    assert next(iter(verdicts.values()))
    assert torch.equal(fp.causal_conv_forward(conv, x, cache), expected)
