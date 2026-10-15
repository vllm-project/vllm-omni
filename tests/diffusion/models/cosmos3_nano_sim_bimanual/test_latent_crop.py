# SPDX-License-Identifier: Apache-2.0
"""Keep VAE input pixels separate from the admitted generation geometry."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual import pipeline_cosmos3_nano_sim_bimanual as pipeline
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import (
    Cosmos3NanoSimBimanualResolutionPolicy,
)


def params() -> SimpleNamespace:
    return SimpleNamespace(height=None, width=None, extra_args={})


@pytest.mark.parametrize(
    "extra, canvas, output", [({}, (480, 832), (352, 640)), ({"resolution": 256}, (192, 320), (176, 320))]
)
def test_preprocessing_keeps_padded_canvas(monkeypatch: pytest.MonkeyPatch, extra, canvas, output) -> None:
    policy = Cosmos3NanoSimBimanualResolutionPolicy()
    monkeypatch.setattr(pipeline.Cosmos3NanoSimBimanualManifest, "from_od_config", lambda _: None)
    monkeypatch.setattr(pipeline, "_resolution_policy", lambda *_: policy)
    preprocess = pipeline.get_cosmos3_nano_sim_bimanual_pre_process_func(
        SimpleNamespace(model_config={"guardrails": False}, tf_model_config=None)
    )
    request = SimpleNamespace(
        sampling_params=params(),
        prompt={"multi_modal_data": {"image": Image.new("RGB", (640, 360))}},
    )
    request.sampling_params.extra_args = extra.copy()
    result = preprocess(request)
    assert result.prompt["additional_information"]["preprocessed_image"].shape[-2:] == canvas
    assert request.prompt["multi_modal_data"]["image"].size == (640, 360)
    assert result.sampling_params.extra_args == extra
    assert (result.sampling_params.height, result.sampling_params.width) == output


@pytest.mark.parametrize("crop", [False, True])
def test_encode_then_crop_reuses_base_helper(crop: bool) -> None:
    pipe = object.__new__(pipeline.Cosmos3NanoSimBimanualPipeline)
    pipe.transformer = SimpleNamespace(latent_channel_size=1)
    pipe.vae_scale_factor_spatial = 16
    latent = torch.arange(30 * 52).reshape(1, 1, 1, 30, 52).float()
    image = torch.zeros(1, 3, 480, 832)

    def encode(value: torch.Tensor) -> torch.Tensor:
        assert value is image
        return latent

    pipe._encode_conditioning_image_latent = encode
    sp = params()
    geometry = (
        Cosmos3NanoSimBimanualResolutionPolicy().resolve(352, 640)
        if crop
        else (Cosmos3NanoSimBimanualResolutionPolicy().resolve(480, 832))
    )
    output = pipe._initial_condition_latent({"additional_information": {"preprocessed_image": image}}, sp, geometry)
    expected = latent[..., :22, :40] if crop else latent
    assert torch.equal(output, expected)
