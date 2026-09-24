# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pickle

import pytest
import torch
from vllm import SamplingParams

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.output_formatter import (
    format_diffusion_outputs,
    normalize_diffusion_postprocess_output,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.conditioning import MINIMAX_H3_ENCODER_REQUEST_KEY
from vllm_omni.model_executor.stage_input_processors.minimax_h3 import (
    diffusion2decoder,
    prepare_encoder_prompt,
    prepare_encoder_prompt_with_decoder,
)
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _format_denoise_output(video_latents, audio_latents, metadata, *, finished=True):
    output = DiffusionOutput(
        output={
            "payload": {"trajectory": {"latents": {"video": video_latents, "audio": audio_latents}}},
            "metadata": metadata,
        },
        to_cpu=True,
        finished=finished,
    )
    # Serialization and formatting must preserve both video and audio latents.
    output = pickle.loads(pickle.dumps(output))
    [formatted] = format_diffusion_outputs(
        request=OmniDiffusionRequest(
            prompt="A bird sings beside a river.",
            request_id="decode-request",
            sampling_params=OmniDiffusionSamplingParams(seed=42, num_outputs_per_prompt=len(video_latents)),
        ),
        od_config=OmniDiffusionConfig(model_class_name="MiniMaxH3Pipeline"),
        diffusion_output=output,
        output_data=output.output,
        postprocess_output=normalize_diffusion_postprocess_output(output.output),
    )
    assert isinstance(formatted, OmniRequestOutput)
    return pickle.loads(pickle.dumps(formatted))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("preencode_mp4", [False, True])
def test_decoder_handoff_preserves_latent_lists_and_decode_parameters(dtype, preencode_mp4):
    video_latents = [
        torch.arange(48, dtype=dtype).reshape(1, 2, 3, 2, 4),
        torch.arange(64, dtype=dtype).reshape(1, 2, 4, 2, 4).neg(),
    ]
    audio_latents = [
        torch.arange(30, dtype=dtype).reshape(1, 3, 10),
        torch.arange(36, dtype=dtype).reshape(1, 3, 12).neg(),
    ]
    decode_parameters = {
        "height": 256,
        "width": 448,
        "preencode_mp4": preencode_mp4,
        "video_codec_options": {"crf": "18", "preset": "fast"},
        "preencode_batch_frames": 7,
    }
    source = _format_denoise_output(video_latents, audio_latents, {"minimax_h3_decode": decode_parameters})
    assert source.images == []
    assert set(source.latents) == {"video", "audio"}
    assert source.multimodal_output["metadata"]["minimax_h3_decode"] == decode_parameters

    original_info = {
        "global_request_id": ["decode-request"],
        "encoder_output": {"conditioning": torch.ones(3)},
        "hidden_states": {"layers": {0: torch.zeros(2)}},
    }
    original_media = {"video": "original-reference.mp4"}
    prompt = {
        "prompt": "A bird sings beside a river.",
        "prompt_token_ids": [2, 3],
        "additional_information": original_info,
        "multi_modal_data": original_media,
    }
    decoder_prompt = diffusion2decoder([source], prompt=prompt)
    assert decoder_prompt is not prompt
    assert decoder_prompt["prompt"] == prompt["prompt"]
    assert decoder_prompt["prompt_token_ids"] == prompt["prompt_token_ids"]
    assert decoder_prompt["multi_modal_data"] is None
    assert set(decoder_prompt["additional_information"]) == {"global_request_id", "minimax_h3_decode"}
    assert decoder_prompt["additional_information"]["global_request_id"] == "decode-request"
    assert prompt["additional_information"] is original_info
    assert "encoder_output" in original_info
    assert prompt["multi_modal_data"] is original_media

    decoder_prompt = pickle.loads(pickle.dumps(decoder_prompt))
    handoff = decoder_prompt["additional_information"]["minimax_h3_decode"]
    assert {key: handoff[key] for key in decode_parameters} == decode_parameters
    for name, expected in (("video_latents", video_latents), ("audio_latents", audio_latents)):
        assert isinstance(handoff[name], list)
        assert len(handoff[name]) == len(expected)
        for actual, reference in zip(handoff[name], expected, strict=True):
            assert actual.device.type == "cpu"
            assert actual.shape == reference.shape
            assert actual.dtype == reference.dtype
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_decoder_handoff_rejects_missing_decode_metadata():
    source = _format_denoise_output([torch.ones(1, 2, 3, 2, 4)], [torch.ones(1, 3, 10)], {})
    with pytest.raises(RuntimeError, match="decoder latent payload"):
        diffusion2decoder([source], prompt={"prompt": "A bird sings."})


def test_decoder_handoff_waits_for_finished_denoise_output():
    source = _format_denoise_output([torch.ones(1, 2, 3, 2, 4)], [torch.ones(1, 3, 10)], {}, finished=False)
    assert not source.finished
    assert diffusion2decoder([source], prompt={"prompt": "A bird sings."}) is None


def test_three_stage_encoder_prompt_uses_dit_sampling_parameters():
    prompt = {"prompt": "A bird sings beside a river."}
    encoder_params = SamplingParams(max_tokens=1, temperature=0.0)
    dit_params = OmniDiffusionSamplingParams(
        height=256,
        width=448,
        num_frames=107,
        fps=24,
        extra_args={"task": "t2va", "aspect_ratio": "16:9", "duration": 4.4},
    )
    # Decoder params differ deliberately; using them would reject text-only T2VA as Ref2VA.
    decoder_params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        extra_args={"task": "ref2va"},
    )
    expected = prepare_encoder_prompt(prompt, [encoder_params, dit_params])
    prepared = prepare_encoder_prompt_with_decoder(prompt, [encoder_params, dit_params, decoder_params])
    assert prepared == expected
    request_metadata = prepared["additional_information"]["meta"][MINIMAX_H3_ENCODER_REQUEST_KEY]
    assert request_metadata["task"] == "t2va"
    assert (request_metadata["height"], request_metadata["width"]) == (256, 448)
    assert prepared["multi_modal_data"] is None
    assert prompt == {"prompt": "A bird sings beside a river."}
