# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from torch import nn
from transformers import BatchEncoding, BatchFeature
from transformers.modeling_outputs import BaseModelOutputWithPast

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    QwenImageEditPlusPipeline,
    get_qwen_image_edit_plus_pre_process_func,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    ModulateIndexPrepare,
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType, OmniTextPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

PIPELINE_MODULE = "vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus"


def _request(prompt: OmniPromptType, **sampling_kwargs) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(seed=42, **sampling_kwargs),
        request_id="qwen-edit-plus-regression",
    )


@pytest.fixture
def preprocessor(tmp_path: Path, mocker):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps({"z_dim": 16}))
    image_processor = mocker.Mock(spec=VaeImageProcessor)
    image_processor.resize.return_value = Image.new("RGB", (32, 32))
    image_processor.preprocess.return_value = torch.zeros(1, 3, 32, 32)
    mocker.patch(f"{PIPELINE_MODULE}.VaeImageProcessor", return_value=image_processor)
    process = get_qwen_image_edit_plus_pre_process_func(OmniDiffusionConfig(model=str(tmp_path)))
    return process, image_processor


@pytest.fixture
def pipeline(mocker):
    # Exercise the real preprocessing/encoding/latent methods without loading weights.
    pipe = object.__new__(QwenImageEditPlusPipeline)
    nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.vae_scale_factor = 8
    pipe.latent_channels = 4
    pipe.tokenizer_max_length = 1024
    pipe.prompt_template_encode = "edit-template: {}"
    pipe.prompt_template_encode_start_idx = 64
    input_ids = torch.arange(80).reshape(1, 80)
    attention_mask = torch.ones_like(input_ids)
    pipe.tokenizer = mocker.Mock(return_value=BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask}))
    pipe.processor = mocker.Mock(
        return_value=BatchFeature(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": torch.zeros(1, 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        )
    )
    hidden_states = torch.arange(320, dtype=torch.float32).reshape(1, 80, 4)
    pipe.text_encoder = mocker.Mock(
        dtype=torch.float32,
        return_value=BaseModelOutputWithPast(hidden_states=(hidden_states,)),
    )
    pipe.transformer = mocker.Mock(spec=QwenImageTransformer2DModel, in_channels=16, guidance_embeds=False)
    mocker.patch.object(pipe, "_encode_vae_image")
    mocker.patch.object(pipe, "check_cfg_parallel_validity")
    mocker.patch.object(pipe, "prepare_timesteps", return_value=(torch.tensor([1.0, 0.5]), 2))
    mocker.patch.object(pipe, "diffuse", side_effect=lambda *args, **kwargs: args[4])
    return pipe


@pytest.mark.parametrize("plain_string", [False, True], ids=["omni-text-prompt", "plain-string"])
@pytest.mark.parametrize(
    "height,width,expected_size",
    [(None, None, (1024, 1024)), (529, 769, (528, 768)), (1, 7, (16, 16))],
    ids=["defaults", "aligned", "minimum"],
)
def test_text_only_preprocess_normalizes_dimensions(preprocessor, plain_string, height, width, expected_size):
    process, image_processor = preprocessor
    prompt = "A green turtle" if plain_string else OmniTextPrompt(prompt="A green turtle", negative_prompt=" ")
    request = _request(prompt, height=height, width=width)

    assert process(request) is request

    assert isinstance(request.prompt, dict)
    prepared_prompt = cast(OmniTextPrompt, request.prompt)
    assert prepared_prompt["prompt"] == "A green turtle"
    if not plain_string:
        assert prepared_prompt["negative_prompt"] == " "
    assert (request.sampling_params.height, request.sampling_params.width) == expected_size
    info = prepared_prompt["additional_information"]
    assert info["condition_images"] is None
    assert info["vae_images"] is None
    assert info["condition_image_sizes"] == []
    assert info["vae_image_sizes"] == []
    assert (info["calculated_height"], info["calculated_width"]) == expected_size
    image_processor.resize.assert_not_called()
    image_processor.preprocess.assert_not_called()


@pytest.mark.parametrize("image_count", [1, 2])
def test_image_preprocess_retains_conditioning(preprocessor, image_count):
    process, image_processor = preprocessor
    images = [Image.new("RGB", (32, 32)) for _ in range(image_count)]
    request = _request(OmniTextPrompt(prompt="Edit the turtle", multi_modal_data={"image": images}))

    process(request)

    assert isinstance(request.prompt, dict)
    info = cast(OmniTextPrompt, request.prompt)["additional_information"]
    assert len(info["condition_images"]) == image_count
    assert len(info["vae_images"]) == image_count
    assert info["condition_image_sizes"] == [(384, 384)] * image_count
    assert info["vae_image_sizes"] == [(1024, 1024)] * image_count
    assert image_processor.resize.call_count == image_count
    assert image_processor.preprocess.call_count == image_count
    assert all(image.shape == (1, 3, 1, 32, 32) for image in info["vae_images"])


def test_qwen_image_edit_plus_rejects_too_many_input_images(preprocessor):
    process, _ = preprocessor
    image = Image.new("RGB", (32, 32))
    request = _request(OmniTextPrompt(prompt="combine", multi_modal_data={"image": [image] * 5}))

    with pytest.raises(ValueError, match=r"At most 4 images are supported by this model"):
        process(request)


def test_empty_image_list_is_not_text_only(preprocessor):
    process, _ = preprocessor
    request = _request(OmniTextPrompt(prompt="A turtle", multi_modal_data={"image": []}))

    with pytest.raises(ValueError, match="Input image list cannot be empty"):
        process(request)


@pytest.mark.parametrize("prompt_name", ["prompt", "negative_prompt"])
def test_text_only_encoder_omits_image_inputs(pipeline, prompt_name):
    embeds, mask = pipeline.encode_prompt(prompt="A turtle", image=None, prompt_name=prompt_name)

    pipeline.processor.assert_not_called()
    assert set(pipeline.text_encoder.call_args.kwargs) == {
        "input_ids",
        "attention_mask",
        "output_hidden_states",
    }
    for call in pipeline.tokenizer.call_args_list:
        text = call.args[0][0]
        assert "Describe the image by detailing" in text
        assert "<|image_pad|>" not in text
        assert "<|vision_start|>" not in text
        assert "Picture" not in text
    hidden = pipeline.text_encoder.return_value.hidden_states[-1]
    torch.testing.assert_close(embeds, hidden[:, 34:])
    assert mask.shape == (1, 46)
    assert mask.all()
    assert pipeline.prompt_template_encode == "edit-template: {}"
    assert pipeline.prompt_template_encode_start_idx == 64


@pytest.mark.parametrize("prompt_name", ["prompt", "negative_prompt"])
def test_text_only_encoder_clips_to_requested_sequence_length(pipeline, prompt_name):
    embeds, mask = pipeline.encode_prompt(
        prompt="A turtle",
        image=None,
        max_sequence_length=16,
        num_images_per_prompt=2,
        prompt_name=prompt_name,
    )

    hidden = pipeline.text_encoder.return_value.hidden_states[-1]
    torch.testing.assert_close(embeds, hidden[:, 34:50].repeat(2, 1, 1))
    assert mask.shape == (2, 16)
    assert mask.all()


def test_precomputed_embeddings_are_not_clipped_without_images(pipeline):
    original = torch.arange(184, dtype=torch.float32).reshape(1, 46, 4)
    original_mask = torch.ones(1, 46, dtype=torch.long)

    embeds, mask = pipeline.encode_prompt(
        prompt="A turtle",
        image=None,
        prompt_embeds=original,
        prompt_embeds_mask=original_mask,
        max_sequence_length=16,
    )

    torch.testing.assert_close(embeds, original)
    torch.testing.assert_close(mask, original_mask)
    pipeline.tokenizer.assert_not_called()
    pipeline.text_encoder.assert_not_called()


@pytest.mark.parametrize("image_count", [1, 2])
def test_image_encoder_keeps_edit_template_after_text_only_call(pipeline, image_count):
    pipeline.encode_prompt(prompt="A turtle", image=None)
    images = [Image.new("RGB", (32, 32)) for _ in range(image_count)]

    embeds, mask = pipeline.encode_prompt(prompt="Paint it blue", image=images, max_sequence_length=8)

    processor_call = pipeline.processor.call_args.kwargs
    assert processor_call["images"] is images
    text = processor_call["text"][0]
    assert text.startswith("edit-template: ")
    assert text.count("<|image_pad|>") == image_count
    assert f"Picture {image_count}:" in text
    model_inputs = pipeline.processor.return_value
    assert pipeline.text_encoder.call_args.kwargs["pixel_values"] is model_inputs.pixel_values
    assert pipeline.text_encoder.call_args.kwargs["image_grid_thw"] is model_inputs.image_grid_thw
    hidden = pipeline.text_encoder.return_value.hidden_states[-1]
    torch.testing.assert_close(embeds, hidden[:, 64:])
    # Expanded image tokens are not capped by the text-only prompt budget.
    assert mask.shape == (1, 16)


def test_text_only_latents_are_seeded_noise_without_vae_encoding(pipeline):
    def sample(seed):
        return pipeline.prepare_latents(
            images=None,
            batch_size=1,
            num_channels_latents=4,
            height=32,
            width=48,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )

    latents, image_latents = sample(42)
    repeated, _ = sample(42)
    changed, _ = sample(43)

    assert image_latents is None
    assert latents.shape == (1, 6, 16)
    assert torch.isfinite(latents).all()
    torch.testing.assert_close(latents, repeated)
    assert not torch.equal(latents, changed)
    pipeline._encode_vae_image.assert_not_called()


@pytest.mark.parametrize("negative_prompt", [None, " "])
@pytest.mark.parametrize("num_outputs", [1, 2])
def test_text_only_forward_uses_output_only_shapes(preprocessor, pipeline, mocker, negative_prompt, num_outputs):
    process, _ = preprocessor
    prompt = OmniTextPrompt(prompt="A turtle")
    if negative_prompt is not None:
        prompt["negative_prompt"] = negative_prompt
    request = _request(
        prompt,
        height=32,
        width=48,
        output_type="latent",
        true_cfg_scale=4.0,
        num_inference_steps=2,
        num_outputs_per_prompt=num_outputs,
        generator=torch.Generator(device="cpu").manual_seed(42),
    )
    prepare_latents = mocker.spy(pipeline, "prepare_latents")

    result = pipeline.forward(DiffusionRequestBatch(requests=[process(request)]))

    assert prepare_latents.call_args.args[0] is None
    assert result.output.shape == (num_outputs, 6, 16)
    diffusion_call = pipeline.diffuse.call_args
    assert diffusion_call.kwargs["image_latents"] is None
    assert diffusion_call.args[5] == [[(1, 2, 3)]]
    assert diffusion_call.args[9] is (negative_prompt is not None)
    assert pipeline.text_encoder.call_count == (1 if negative_prompt is None else 2)
    pipeline.processor.assert_not_called()
    pipeline._encode_vae_image.assert_not_called()

    # Keep the edit checkpoint's zero_cond_t: every noise token uses the real timestep.
    timesteps, modulation = ModulateIndexPrepare(zero_cond_t=True)(torch.tensor([0.5]), diffusion_call.args[5])
    torch.testing.assert_close(timesteps, torch.tensor([0.5, 0.0]))
    assert modulation.shape == (1, 6)
    assert torch.count_nonzero(modulation) == 0


@pytest.mark.parametrize(
    "info",
    [
        pytest.param({}, id="missing-both"),
        pytest.param({"condition_images": None}, id="missing-vae"),
        pytest.param({"vae_images": None}, id="missing-conditioning"),
        pytest.param({"condition_images": None, "vae_images": None}, id="image-with-text-only-metadata"),
        pytest.param({"condition_images": None, "vae_images": []}, id="vae-without-conditioning"),
        pytest.param({"condition_images": [], "vae_images": None}, id="conditioning-without-vae"),
    ],
)
def test_real_image_without_preprocessing_still_fails(pipeline, info):
    request = _request(
        OmniTextPrompt(
            prompt="Edit the turtle",
            multi_modal_data={"image": Image.new("RGB", (32, 32))},
            additional_information=info,
        )
    )

    with pytest.raises(RuntimeError, match="Missing preprocess images"):
        pipeline.forward(DiffusionRequestBatch(requests=[request]))
    pipeline.text_encoder.assert_not_called()
    pipeline.diffuse.assert_not_called()
