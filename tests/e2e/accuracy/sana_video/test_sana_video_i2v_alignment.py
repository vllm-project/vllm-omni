# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.benchmark,
]

VARIANTS = {
    "480p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        "vae_class": "AutoencoderKLWan",
        "native_vae_class": "DistributedAutoencoderKLWan",
        "dtype": torch.float32,
        "latent_channels": 16,
        "spatial_scale": 8,
        "temporal_scale": 4,
        "frames": 5,
        "max_abs": 1e-6,
        "relative_l2": 1e-6,
    },
    "720p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        "vae_class": "AutoencoderKLLTX2Video",
        "native_vae_class": "DistributedAutoencoderKLLTX2Video",
        "dtype": torch.bfloat16,
        "latent_channels": 128,
        "spatial_scale": 32,
        "temporal_scale": 8,
        "frames": 9,
        "max_abs": 0.04,
        "relative_l2": 0.004,
    },
}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "_hardware",
    [pytest.param(None, marks=hardware_marks(res={"cuda": "H100"}))],
)
def test_sana_video_i2v_first_frame_latents_match_diffusers(variant: str, _hardware) -> None:
    del _hardware
    from diffusers import (
        AutoencoderKLLTX2Video,
        AutoencoderKLWan,
    )
    from diffusers import (
        SanaImageToVideoPipeline as DiffusersSanaImageToVideoPipeline,
    )

    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
        DistributedAutoencoderKLLTX2Video,
    )
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
    )
    from vllm_omni.diffusion.models.sana_video import SanaImageToVideoPipeline

    config = VARIANTS[variant]
    vae_classes = {
        "AutoencoderKLLTX2Video": AutoencoderKLLTX2Video,
        "AutoencoderKLWan": AutoencoderKLWan,
    }
    native_vae_classes = {
        "DistributedAutoencoderKLLTX2Video": DistributedAutoencoderKLLTX2Video,
        "DistributedAutoencoderKLWan": DistributedAutoencoderKLWan,
    }
    reference_vae = (
        vae_classes[config["vae_class"]]
        .from_pretrained(
            config["model"],
            subfolder="vae",
            torch_dtype=config["dtype"],
        )
        .to("cuda")
    )
    native_vae = (
        native_vae_classes[config["native_vae_class"]]
        .from_config(dict(reference_vae.config))
        .to(device="cuda", dtype=config["dtype"])
    )
    native_vae.load_state_dict(reference_vae.state_dict())
    reference_vae.eval()
    native_vae.eval()

    height, width = 192, 320
    num_latent_frames = (config["frames"] - 1) // config["temporal_scale"] + 1
    image = torch.linspace(
        -1,
        1,
        3 * height * width,
        dtype=torch.float32,
        device="cuda",
    ).reshape(1, 3, height, width)
    noise = torch.randn(
        (
            1,
            config["latent_channels"],
            num_latent_frames,
            height // config["spatial_scale"],
            width // config["spatial_scale"],
        ),
        generator=torch.Generator(device="cuda").manual_seed(7),
        device="cuda",
        dtype=torch.float32,
    )
    reference_pipeline = SimpleNamespace(
        vae=reference_vae,
        vae_scale_factor_temporal=config["temporal_scale"],
        vae_scale_factor_spatial=config["spatial_scale"],
    )
    native_pipeline = SimpleNamespace(
        vae=native_vae,
        vae_scale_factor_temporal=config["temporal_scale"],
        vae_scale_factor_spatial=config["spatial_scale"],
        device=torch.device("cuda"),
    )

    with torch.inference_mode():
        expected = DiffusersSanaImageToVideoPipeline.prepare_latents(
            reference_pipeline,
            image,
            1,
            config["latent_channels"],
            height,
            width,
            config["frames"],
            torch.float32,
            torch.device("cuda"),
            None,
            noise.clone(),
        )
        actual = SanaImageToVideoPipeline._prepare_i2v_latents(
            native_pipeline,
            image,
            1,
            config["latent_channels"],
            height,
            width,
            config["frames"],
            torch.float32,
            None,
            noise.clone(),
        )

    # Conditioning replaces only the first latent frame.
    torch.testing.assert_close(actual[:, :, 1:], noise[:, :, 1:], rtol=0, atol=0)
    error = (actual[:, :, :1] - expected[:, :, :1]).float()
    expected_first_frame = expected[:, :, :1].float()
    max_abs = error.abs().max()
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected_first_frame)
    cosine = torch.nn.functional.cosine_similarity(
        actual[:, :, :1].flatten(),
        expected[:, :, :1].flatten(),
        dim=0,
    )
    assert max_abs.item() <= config["max_abs"]
    assert relative_l2.item() <= config["relative_l2"]
    assert cosine.item() >= 0.9999
