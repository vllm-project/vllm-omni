# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend NPU smoke tests for SeedVR2 modules, no checkpoint required.

Both tests build the released 3B architecture with random weights and run a
tiny workload directly on NPU. They validate operator support and the
platform-portable execution paths (fp32 RoPE, grouped window SDPA, causal
Conv3d VAE) before the full checkpoint e2e in ``test_seedvr2_e2e.py`` is
attempted on the same device.
"""

from __future__ import annotations

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


requires_npu = pytest.mark.skipif(not _npu_available(), reason="requires Ascend NPU")


@hardware_test(res={"npu": "A2"}, num_cards=1)
@requires_npu
def test_seedvr2_nadit_forward_npu() -> None:
    from vllm_omni.diffusion.models.seedvr2.nadit import SEEDVR2_3B_CONFIG, SeedVR2NaDiT

    device = torch.device("npu")
    model = SeedVR2NaDiT(**SEEDVR2_3B_CONFIG, use_varlen_kernel=False).to(device=device, dtype=torch.float16)
    # The varlen kernel is not requested, so every layer must resolve to the
    # portable grouped window SDPA path regardless of the selected backend.
    summary = model.attention_path_summary()
    assert summary["layers_per_path"].get("grouped_sdpa") == SEEDVR2_3B_CONFIG["num_layers"], summary

    generator = torch.Generator(device=device).manual_seed(7723)
    # ``vid_shape`` is the latent grid (5 = 4n+1 frames at 128x256 after the
    # VAE's 8x spatial compression); tokens are (frames, height // 2, width // 2).
    frames, height, width = 5, 16, 32
    text_len = 8
    vid = torch.randn(
        (frames * height * width, SEEDVR2_3B_CONFIG["vid_in_channels"]),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    txt = torch.randn(
        (text_len, SEEDVR2_3B_CONFIG["txt_in_dim"]), generator=generator, device=device, dtype=torch.float16
    )
    vid_shape = torch.tensor([[frames, height, width]], device=device, dtype=torch.long)
    txt_shape = torch.tensor([[text_len]], device=device, dtype=torch.long)
    timestep = torch.tensor([1000.0], device=device, dtype=torch.float16)

    with torch.inference_mode():
        output = model(vid, txt, vid_shape, txt_shape, timestep, runtime=None)

    sample = output.vid_sample
    # ``vid_sample`` rows span the latent grid (patch (1, 2, 2) undone) with
    # one channel vector per latent cell.
    assert sample.shape[-1] == SEEDVR2_3B_CONFIG["vid_out_channels"]
    assert sample.numel() == frames * height * width * SEEDVR2_3B_CONFIG["vid_out_channels"]
    assert torch.isfinite(sample.float()).all(), "SeedVR2 NaDiT forward produced non-finite values on NPU"
    stats = model.attention_path_summary()
    assert stats["grouped_sdpa_calls"] > 0, stats


@hardware_test(res={"npu": "A2"}, num_cards=1)
@requires_npu
def test_seedvr2_vae_roundtrip_npu() -> None:
    from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE

    device = torch.device("npu")
    vae = SeedVR2VAE().to(device=device, dtype=torch.float16)
    generator = torch.Generator(device=device).manual_seed(7723)
    # 4n+1 frames at the smallest legal size; the causal VAE compresses time by
    # 4x (1 + (T - 1) / 4) and space by 8x, giving a (5, 8, 14) -> (2, 8, 14) latent.
    video = torch.rand((1, 3, 5, 64, 112), generator=generator, device=device, dtype=torch.float16) * 2 - 1

    with torch.inference_mode():
        latent = vae.encode(video).sample(generator=generator)
        decoded = vae.decode(latent)

    assert latent.shape == (1, 16, 2, 8, 14)
    assert decoded.shape == video.shape
    assert torch.isfinite(decoded.float()).all(), "SeedVR2 VAE round trip produced non-finite values on NPU"
