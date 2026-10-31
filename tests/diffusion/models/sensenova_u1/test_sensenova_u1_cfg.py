# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion import output_formatter
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline
from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _pipeline_without_init() -> SenseNovaU1Pipeline:
    return object.__new__(SenseNovaU1Pipeline)


@pytest.mark.parametrize("kwargs", [None, {"step_i": 0}])
def test_combine_cfg_noise_requires_is_it2i(kwargs):
    pipe = _pipeline_without_init()
    out_cond = (torch.ones(1, 2, 3),)
    out_uncond = (torch.zeros(1, 2, 3),)

    with pytest.raises(ValueError, match="is_it2i"):
        pipe.combine_cfg_noise(
            out_cond,
            out_uncond,
            cfg_scale=4.0,
            cfg_norm="cfg_zero_star",
            kwargs=kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"is_it2i": False}, {"is_it2i": False, "step_i": None}],
)
def test_cfg_zero_star_requires_step_i(kwargs):
    pipe = _pipeline_without_init()
    out_cond = (torch.ones(1, 2, 3),)
    out_uncond = (torch.zeros(1, 2, 3),)
    with pytest.raises(ValueError, match="step_i"):
        pipe.combine_cfg_noise(
            out_cond,
            out_uncond,
            cfg_scale=4.0,
            cfg_norm="cfg_zero_star",
            kwargs=kwargs,
        )


def test_cfg_zero_star_accepts_step_i():
    pipe = _pipeline_without_init()
    out_cond = (torch.ones(1, 2, 3),)
    out_uncond = (torch.zeros(1, 2, 3),)
    result = pipe.combine_cfg_noise(
        out_cond,
        out_uncond,
        cfg_scale=4.0,
        cfg_norm="cfg_zero_star",
        kwargs={"is_it2i": False, "step_i": 0},
    )

    assert result.shape == out_cond[0].shape
    assert torch.equal(result, torch.zeros_like(out_cond[0]))
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("think_text", ["", "A black and a white image."])
def test_denoising_preserves_all_images_and_metadata(batch_size, think_text, monkeypatch):
    pipe = _pipeline_without_init()
    torch.nn.Module.__init__(pipe)
    pipe.patch_size = 2
    pipe.merge_size = 1
    pipe.model_cfg = SimpleNamespace(add_noise_scale_embedding=False)
    pipe.fm_modules = {"timestep_embedder": lambda t: t[:, None]}
    monkeypatch.setattr(pipe, "_extract_feature", lambda x, **kwargs: torch.zeros(x.shape[0], 3))
    monkeypatch.setattr(pipe, "_denoise", lambda image, ns, t, z, *args: torch.zeros_like(z))
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    ns = SimpleNamespace(
        image_prediction=torch.stack([torch.full((3, 4, 4), value) for value in (-1.0, 1.0)[:batch_size]]),
        timesteps=torch.tensor([1.0, 0.0]),
        grid_h=2,
        grid_w=2,
        token_h=2,
        token_w=2,
        grid_hw=torch.tensor([[2, 2]]),
    )
    params = SimpleNamespace(num_steps=1, batch_size=batch_size, image_size=(4, 4))
    result = pipe._run_denoising_loop(ns, {}, params, think_text)
    formatted = format_diffusion_outputs(
        request=OmniDiffusionRequest(
            prompt="Black and white images",
            sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=batch_size, seed=42),
            request_id="batch-images",
        ),
        od_config=SimpleNamespace(model_class_name="SenseNovaU1Pipeline"),
        diffusion_output=result,
        output_data=result.output,
        postprocess_output=normalize_diffusion_postprocess_output(result.output),
    )
    assert len(formatted) == 1
    assert len(formatted[0].images) == batch_size
    assert [image.getpixel((0, 0)) for image in formatted[0].images] == [(0, 0, 0), (255, 255, 255)][:batch_size]
    expected_metadata = {"text": {"think_text": think_text}} if think_text else {}
    assert result.output["metadata"] == expected_metadata
