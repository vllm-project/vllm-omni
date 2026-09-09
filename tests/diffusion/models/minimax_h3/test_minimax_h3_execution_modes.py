# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def pipeline():
    model = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(model)
    model.load_text_encoder = True
    model.load_vae_encoder = True
    model.device = torch.device("cpu")
    model.partition = "combined"
    model.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    model.default_video_shift = 12.0
    model.default_audio_shift = 3.0
    model.od_config = SimpleNamespace(enable_layerwise_offload=False)
    model._quality_policy = SimpleNamespace(resolve=Mock(return_value=SimpleNamespace(cache_dit=None)))
    model._cache_dit_runtime = SimpleNamespace(prepare=Mock())
    model.encode_prompt = Mock(
        return_value=(torch.ones(2, 5120, dtype=torch.bfloat16), torch.ones(2, dtype=torch.long))
    )
    model.video_vae = Mock()
    model.video_vae.is_distributed_enabled.return_value = False
    model.video_vae.encode_image.side_effect = lambda image: torch.ones((image.height // 32) * (image.width // 32), 96)
    model.audio_vae = Mock()
    model.transformer = torch.nn.Identity()
    model.diffuse = Mock(return_value=(torch.zeros(1), torch.zeros(1)))
    model.decode = Mock(return_value=(torch.zeros(1, 3, 2, 4, 4), torch.zeros(1, 2, 8)))
    return model


def _request(task, *, payload=None):
    prompt: dict[str, Any] = {"prompt": "A person waves."}
    if task != "t2va":
        prompt["multi_modal_data"] = {"image": Image.new("RGB", (256, 256))}
    if payload is not None:
        prompt["additional_information"] = {"encoder_output": payload}
    sampling = OmniDiffusionSamplingParams(
        height=32, width=32, num_frames=96, num_inference_steps=2, extra_args={"task": task, "aspect_ratio": "1:1"}
    )
    return SimpleNamespace(prompts=[prompt], sampling_params=sampling)


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_single_stage_encodes_without_stage_zero(pipeline, task):
    output = pipeline.forward(_request(task))
    pipeline.encode_prompt.assert_called_once()
    assert pipeline.video_vae.encode_image.call_count == int(task != "t2va")
    pipeline.diffuse.assert_called_once()
    assert pipeline.diffuse.call_args.kwargs["task"] == task
    video, audio = output.output
    torch.testing.assert_close(video, torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8))
    torch.testing.assert_close(audio, pipeline.decode.return_value[1])


def test_single_stage_reuses_legacy_text_but_encodes_media(pipeline):
    payload = {
        "hidden_states": torch.full((2, 5120), 3.0, dtype=torch.bfloat16),
        "token_tags": torch.ones(2, dtype=torch.long),
    }
    pipeline.forward(_request("fl2va", payload=payload))
    pipeline.encode_prompt.assert_not_called()
    pipeline.video_vae.encode_image.assert_called_once()
    torch.testing.assert_close(pipeline.diffuse.call_args.kwargs["text_embeddings"], payload["hidden_states"])


def test_text_disaggregation_keeps_local_media_encoding(pipeline):
    payload = {
        "hidden_states": torch.full((2, 5120), 3.0, dtype=torch.bfloat16),
        "token_tags": torch.ones(2, dtype=torch.long),
    }
    pipeline.load_text_encoder = False
    pipeline.load_vae_encoder = True
    pipeline.forward(_request("fl2va", payload=payload))
    pipeline.encode_prompt.assert_not_called()
    pipeline.video_vae.encode_image.assert_called_once()


@pytest.mark.parametrize("payload", [None, {}, {"hidden_states": torch.ones(2, 5120), "token_tags": torch.ones(2)}])
def test_stage_one_rejects_missing_or_legacy_conditioning(pipeline, payload):
    pipeline.load_text_encoder = False
    pipeline.load_vae_encoder = False
    with pytest.raises(OmniClientError):
        pipeline.forward(_request("t2va", payload=payload))
    pipeline.encode_prompt.assert_not_called()
    pipeline.video_vae.encode_image.assert_not_called()
    pipeline.diffuse.assert_not_called()


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("load_text_encoder", [True, False])
def test_prepare_encode_uses_the_same_mode_and_conditioning(pipeline, task, load_text_encoder):
    from vllm_omni.diffusion.worker.utils import StepRequestState

    request = _request(task)
    if not load_text_encoder:
        conditioning = pipeline._prepare_local_conditioning(request.prompts[0], request.sampling_params)
        request.prompts[0] = {"additional_information": {"encoder_output": conditioning.to_omni_payload()}}
        pipeline.encode_prompt.reset_mock()
        pipeline.video_vae.encode_image.reset_mock()
    pipeline.load_text_encoder = load_text_encoder
    pipeline.load_vae_encoder = load_text_encoder
    state = StepRequestState(request_id="step", prompt=request.prompts[0], sampling=request.sampling_params)
    assert pipeline.prepare_encode(state) is state
    assert state.latents is not None
    assert state.latents.shape[1] == 96
    assert state.total_steps == 1
    assert pipeline.encode_prompt.call_count == int(load_text_encoder)
    assert pipeline.video_vae.encode_image.call_count == int(load_text_encoder and task != "t2va")


def test_stage_one_step_execution_never_falls_back(pipeline):
    from vllm_omni.diffusion.worker.utils import StepRequestState

    pipeline.load_text_encoder = False
    pipeline.load_vae_encoder = False
    request = _request("t2va")
    with pytest.raises(OmniClientError, match="requires encoder conditioning"):
        pipeline.prepare_encode(
            StepRequestState(request_id="step", prompt=request.prompts[0], sampling=request.sampling_params)
        )
    pipeline.encode_prompt.assert_not_called()


def test_multiple_outputs_encode_only_once(pipeline):
    request = _request("fl2va")
    request.sampling_params.num_outputs_per_prompt = 3
    request.sampling_params.seed = 42
    pipeline.forward(request)
    pipeline.encode_prompt.assert_called_once()
    pipeline.video_vae.encode_image.assert_called_once()
    assert [call.kwargs["seed"] for call in pipeline.diffuse.call_args_list] == [42, 43, 44]


def test_ref2va_partition_preserves_image_only_implicit_task(pipeline):
    request = _request("fl2va")
    del request.sampling_params.extra_args["task"]
    pipeline.partition = "ref2va"
    pipeline.supported_tasks = frozenset({"ref2va"})
    pipeline.forward(request)
    assert pipeline.diffuse.call_args.kwargs["task"] == "ref2va"


def test_invalid_legacy_text_is_not_ignored(pipeline):
    with pytest.raises(OmniClientError, match="encoder wire requires"):
        pipeline.forward(_request("fl2va", payload={"hidden_states": "invalid"}))
    pipeline.encode_prompt.assert_not_called()
    pipeline.video_vae.encode_image.assert_not_called()


def test_text_weight_loading_preserves_encoder_specific_lifecycle(pipeline):
    class Encoder(torch.nn.Module):
        def load_weights(self, weights):
            return {name for name, _ in weights}

    pipeline.text_encoder = Encoder()
    pipeline.video_vae = torch.nn.Identity()
    pipeline.audio_vae = torch.nn.Identity()
    assert pipeline.load_weights([("text_encoder.layer.weight", torch.ones(1))]) == {"text_encoder.layer.weight"}
    pipeline.load_text_encoder = False
    pipeline.load_vae_encoder = False
    pipeline.text_encoder = None
    with pytest.raises(ValueError, match="disabled in this deployment"):
        pipeline.load_weights([("text_encoder.layer.weight", torch.ones(1))])


@pytest.mark.parametrize("connector_handoff", [False, True], ids=["inline", "full_payload"])
def test_mixed_reference_conditioning_matches_stage_zero_handoff(pipeline, monkeypatch, connector_handoff):
    import numpy as np

    from vllm_omni.data_entry_keys import flatten_payload
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing
    from vllm_omni.model_executor.models.minimax_h3.conditioning import (
        MiniMaxH3EncoderConditioning,
        MiniMaxH3TextConditioning,
    )
    from vllm_omni.model_executor.models.minimax_h3.encoder import MiniMaxH3Encoder
    from vllm_omni.model_executor.stage_input_processors.minimax_h3 import (
        encoder2diffusion,
        encoder2diffusion_full_payload,
        prepare_encoder_prompt,
    )

    frames = np.zeros((48, 32, 32, 3), dtype=np.uint8)
    decode_frames = Mock(return_value=frames)
    sampler = Mock(wraps=processing.sample_reference_video_frames)
    monkeypatch.setattr(processing, "load_video_frames", decode_frames)
    monkeypatch.setattr(processing, "sample_reference_video_frames", sampler)
    monkeypatch.setattr(
        processing,
        "prepare_reference_videos",
        Mock(
            return_value=[{"prepared_path": "prepared.mp4", "original_path": "original.mp4", "input_has_audio": True}]
        ),
    )
    monkeypatch.setattr(processing, "load_video_audio", Mock(return_value=(torch.full((64000,), 3.0), 32000)))
    pipeline.video_vae.encode_video.return_value = (torch.full((1, 96), 2.0), (1, 2, 2))

    def encode_audio(waveform, sample_rate):
        length = round(waveform.shape[-1] / sample_rate * 40)
        return torch.full((length * 2, 32), float(waveform[0])), length

    pipeline.audio_vae.encode_waveform.side_effect = encode_audio
    request = _request("ref2va")
    request.prompts[0]["multi_modal_data"].update(video="original.mp4", audio=(torch.full((192000,), 4.0), 32000))
    pipeline.forward(request)
    local_kwargs = pipeline.diffuse.call_args.kwargs

    transformed = prepare_encoder_prompt(request.prompts[0], [request.sampling_params])
    stage_zero = object.__new__(MiniMaxH3Encoder)
    torch.nn.Module.__init__(stage_zero)
    stage_zero._component_leader = True
    stage_zero.video_vae = pipeline.video_vae
    stage_zero.audio_vae = pipeline.audio_vae
    media = stage_zero._encode_media(stage_zero._media_input(transformed["additional_information"]))
    text = MiniMaxH3TextConditioning(*pipeline.encode_prompt.return_value)
    payload = MiniMaxH3EncoderConditioning.from_components(text, media).to_omni_payload()
    wire = flatten_payload(payload)
    source = SimpleNamespace(
        finished=True, outputs=[SimpleNamespace(multimodal_output=None if connector_handoff else wire)]
    )
    bridged = encoder2diffusion([source], request.prompts[0])
    assert bridged["multi_modal_data"] is None
    if connector_handoff:
        assert "encoder_output" not in bridged["additional_information"]
        transferred = encoder2diffusion_full_payload(pooling_output=wire)
        assert set(transferred) == {"encoder_output"}
        received = transferred["encoder_output"]
        for section, key in (
            ("hidden_states", "output"),
            ("meta", "token_role_ids"),
            ("embed", "embedding"),
            ("embed", "speech_feat"),
            ("kv_metadata", "minimax_h3_encoder_layout"),
        ):
            torch.testing.assert_close(received[section][key], payload[section][key])
        # Simulate the generic diffusion runner merging the received payload;
        # native NIXL transport is covered separately, not by this CPU test.
        bridged["additional_information"].update(transferred)
    pipeline.load_text_encoder = False
    pipeline.load_vae_encoder = False
    pipeline.encode_prompt.reset_mock()
    pipeline.video_vae.encode_image.reset_mock()
    pipeline.video_vae.encode_video.reset_mock()
    pipeline.forward(SimpleNamespace(prompts=[bridged], sampling_params=request.sampling_params))
    external_kwargs = pipeline.diffuse.call_args.kwargs
    for key, value in local_kwargs.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, external_kwargs[key])
        else:
            assert value == external_kwargs[key]
    pipeline.encode_prompt.assert_not_called()
    pipeline.video_vae.encode_image.assert_not_called()
    pipeline.video_vae.encode_video.assert_not_called()
    assert decode_frames.call_count == sampler.call_count == 2
    assert all(call.kwargs["decoded_frames"] is frames for call in sampler.call_args_list)
    assert [block["kind"] for block in local_kwargs["ref_blocks"]] == ["image", "video_audio", "audio"]
    assert local_kwargs["audio_condition_lengths"] == [80, round(107 / 24 * 40)]
    assert all(
        call.args[0].shape[-1] == round(107 / 24 * 32000)
        for call in pipeline.audio_vae.encode_waveform.call_args_list[1::2]
    )
