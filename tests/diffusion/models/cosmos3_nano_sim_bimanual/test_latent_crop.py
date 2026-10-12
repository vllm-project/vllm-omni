# SPDX-License-Identifier: Apache-2.0
"""Keep VAE input pixels separate from the admitted generation geometry."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual import pipeline_cosmos3_nano_sim_bimanual as pipeline
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import (
    Cosmos3NanoSimBimanualResolutionPolicy,
    conditioning_image_size,
    resolve_cosmos3_nano_sim_bimanual_geometry,
)


def params(size: object = None, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(height=None, width=None, extra_args={"conditioning_image_size": size}, **kwargs)


def test_cropped_geometry_and_legacy_dimensions() -> None:
    policy = Cosmos3NanoSimBimanualResolutionPolicy()
    cropped = resolve_cosmos3_nano_sim_bimanual_geometry(params([480, 832, 360, 640]), None, policy)
    assert cropped.session_key == (352, 640)
    assert (cropped.latent_height, cropped.latent_width) == (22, 40)
    assert cropped.vision_tokens_per_frame == 220
    legacy = SimpleNamespace(height=480, width=832, extra_args={})
    assert resolve_cosmos3_nano_sim_bimanual_geometry(legacy, None, policy).vision_tokens_per_frame == 390
    legacy.extra_args["conditioning_image_size"] = [480, 832, 360, 640]
    with pytest.raises(ValueError, match="Explicit height/width"):
        resolve_cosmos3_nano_sim_bimanual_geometry(legacy, None, policy)


@pytest.mark.parametrize(
    "size", [[480, 832], [480, 832, 481, 640], [480, 832, 0, 640], [480, 832, True, 640], [480, 832, 360.5, 640]]
)
def test_invalid_metadata(size: object) -> None:
    with pytest.raises(ValueError):
        conditioning_image_size(params(size))


def test_preprocessing_keeps_padded_canvas(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = Cosmos3NanoSimBimanualResolutionPolicy()
    monkeypatch.setattr(pipeline.Cosmos3NanoSimBimanualManifest, "from_od_config", lambda _: None)
    monkeypatch.setattr(pipeline, "_resolution_policy", lambda *_: policy)
    seen = []

    def delegate(request: SimpleNamespace) -> SimpleNamespace:
        seen.append((request.sampling_params.height, request.sampling_params.width))
        assert request.prompt["multi_modal_data"]["image"].size == (832, 480)
        return request

    monkeypatch.setattr(pipeline, "get_cosmos3_pre_process_func", lambda _: delegate)
    preprocess = pipeline.get_cosmos3_nano_sim_bimanual_pre_process_func(None)
    request = SimpleNamespace(
        sampling_params=params([480, 832, 360, 640]),
        prompt={"multi_modal_data": {"image": Image.new("RGB", (832, 480))}},
    )
    result = preprocess(request)
    assert seen == [(480, 832)]
    assert (result.sampling_params.height, result.sampling_params.width) == (352, 640)
    request.prompt["multi_modal_data"]["image"] = Image.new("RGB", (640, 360))
    with pytest.raises(ValueError, match="padded canvas"):
        preprocess(request)


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
    sp = params([480, 832, 360, 640] if crop else None)
    geometry = (
        Cosmos3NanoSimBimanualResolutionPolicy().resolve(352, 640)
        if crop
        else (Cosmos3NanoSimBimanualResolutionPolicy().resolve(480, 832))
    )
    output = pipe._initial_condition_latent({"additional_information": {"preprocessed_image": image}}, sp, geometry)
    expected = latent[..., :22, :40] if crop else latent
    assert torch.equal(output, expected)
