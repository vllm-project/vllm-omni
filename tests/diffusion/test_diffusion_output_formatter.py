# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion import output_formatter
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.output_formatter import (
    format_diffusion_outputs,
    format_empty_diffusion_outputs,
    normalize_diffusion_postprocess_output,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _request(
    prompt: str | dict | None = None,
    *,
    num_outputs_per_prompt: int = 1,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt or "prompt",
        request_id="req-1",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=1,
            num_outputs_per_prompt=num_outputs_per_prompt,
            resolution=512,
        ),
    )


def _config(model_class_name: str = "mock_model") -> SimpleNamespace:
    return SimpleNamespace(model_class_name=model_class_name)


def test_formatter_preserves_single_video_audio_actions_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    postprocess_output = normalize_diffusion_postprocess_output(
        {
            "payload": {
                "video": ["frame-0"],
                "audio": "audio-0",
                "actions": "action-0",
            },
            "metadata": {
                "audio": {"sample_rate": 48000},
                "video": {"fps": 24.0},
                "actions": {"action_mode": "policy"},
            },
        }
    )

    results = format_diffusion_outputs(
        request=_request("prompt-0"),
        od_config=_config(),
        diffusion_output=DiffusionOutput(
            output=None,
            stage_durations={"execute": 1.25},
            peak_memory_mb=321.0,
        ),
        output_data={"raw": "output"},
        postprocess_output=postprocess_output,
    )

    assert len(results) == 1
    result = results[0]
    assert result.images == ["frame-0"]
    assert result.prompt == "prompt-0"
    assert result.final_output_type == "image"
    assert result.multimodal_output == {
        "metadata": {
            "audio": {"sample_rate": 48000},
            "video": {"fps": 24.0},
            "actions": {"action_mode": "policy"},
        },
        "audio": "audio-0",
        "audio_sample_rate": 48000,
        "fps": 24.0,
        "actions": "action-0",
    }
    assert result.stage_durations == {"execute": 1.25}
    assert result.peak_memory_mb == 321.0
    assert result.metrics == {
        "image_num": 1,
        "resolution": 512,
        "width": None,
        "height": None,
    }


def test_formatter_normalizes_payload_metadata_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    postprocess_output = normalize_diffusion_postprocess_output(
        {
            "payload": {
                "video": ["frame-0"],
                "actions": "action-0",
            },
            "metadata": {
                "actions": {
                    "raw_action_dim": 2,
                    "action_mode": "policy",
                    "domain_id": 7,
                },
                "common": {
                    "action_only_output": True,
                },
                "internal": {
                    "robolab_action_postprocess": object(),
                },
            },
        }
    )

    assert postprocess_output.outputs == {
        "video": ["frame-0"],
        "actions": "action-0",
    }
    assert postprocess_output.primary_key == "video"
    assert postprocess_output.metadata == {
        "actions": {
            "raw_action_dim": 2,
            "action_mode": "policy",
            "domain_id": 7,
        },
        "common": {
            "action_only_output": True,
        },
    }
    assert "internal" not in postprocess_output.metadata

    [result] = format_diffusion_outputs(
        request=_request("prompt-0"),
        od_config=_config(),
        diffusion_output=DiffusionOutput(output=None),
        output_data={"raw": "output"},
        postprocess_output=postprocess_output,
    )

    assert result.multimodal_output == {
        "metadata": postprocess_output.metadata,
        "actions": "action-0",
    }


def test_formatter_maps_trajectory_payload_to_request_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    latents = torch.zeros(2, 1, 4)
    timesteps = torch.tensor([1.0, 0.5])
    log_probs = torch.tensor([-0.1, -0.2])
    postprocess_output = normalize_diffusion_postprocess_output(
        {
            "payload": {
                "image": "image-0",
                "trajectory": {
                    "latents": latents,
                    "timesteps": timesteps,
                    "log_probs": log_probs,
                },
            },
            "metadata": {"trajectory": {"type": "denoising"}},
        }
    )

    [result] = format_diffusion_outputs(
        request=_request("prompt-0"),
        od_config=_config(),
        diffusion_output=DiffusionOutput(output=None),
        output_data={"raw": "output"},
        postprocess_output=postprocess_output,
    )

    assert result.images == ["image-0"]
    assert result.latents is latents
    assert result.trajectory_latents is latents
    assert result.trajectory_timesteps is timesteps
    assert result.trajectory_log_probs is log_probs
    assert result.multimodal_output == {
        "metadata": {"trajectory": {"type": "denoising"}},
        "trajectory": {
            "latents": latents,
            "timesteps": timesteps,
            "log_probs": log_probs,
        },
    }


def test_formatter_preserves_text_envelope_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    postprocess_output = normalize_diffusion_postprocess_output(
        {
            "payload": {"text": "caption"},
            "metadata": {"text": {"text_output": "caption"}},
        }
    )

    [result] = format_diffusion_outputs(
        request=_request("describe this"),
        od_config=_config(),
        diffusion_output=DiffusionOutput(output=None),
        output_data={"payload": {"text": "caption"}},
        postprocess_output=postprocess_output,
    )

    assert result.images == []
    assert result.prompt == "describe this"
    assert result.final_output_type == "text"
    assert result.multimodal_output == {
        "metadata": {"text": {"text_output": "caption"}},
        "text": "caption",
    }


@pytest.mark.parametrize(
    "text_metadata",
    [
        pytest.param({"think_text": "reasoning"}, id="sensenova-bagel-thinking"),
        pytest.param({"ar_generated_text": "reasoning"}, id="hunyuan-image3-cot"),
    ],
)
def test_formatter_preserves_image_with_reasoning_metadata(
    monkeypatch: pytest.MonkeyPatch,
    text_metadata: dict[str, str],
) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    metadata = {"text": text_metadata}
    postprocess_output = normalize_diffusion_postprocess_output(
        {
            "payload": {"image": "image-0"},
            "metadata": metadata,
        }
    )

    [result] = format_diffusion_outputs(
        request=_request("generate an image"),
        od_config=_config(),
        diffusion_output=DiffusionOutput(output=None),
        output_data=None,
        postprocess_output=postprocess_output,
    )

    assert postprocess_output.primary_key == "image"
    assert result.images == ["image-0"]
    assert result.final_output_type == "image"
    assert result.multimodal_output == {"metadata": metadata}


def test_formatter_preserves_audio_output_with_model_sample_rate_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AudioModel:
        audio_sample_rate = 44100

    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: True)
    monkeypatch.setattr(
        output_formatter.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda _: AudioModel,
    )
    postprocess_output = normalize_diffusion_postprocess_output(["waveform"])

    [result] = format_diffusion_outputs(
        request=_request("speak"),
        od_config=_config("audio_model"),
        diffusion_output=DiffusionOutput(output=["waveform"]),
        output_data=["waveform"],
        postprocess_output=postprocess_output,
    )

    assert result.images == []
    assert result.prompt == "speak"
    assert result.final_output_type == "audio"
    assert result.multimodal_output == {
        "audio": "waveform",
        "audio_sample_rate": 44100,
    }


def test_formatter_preserves_audio_model_video_audio_and_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: True)
    postprocess_output = normalize_diffusion_postprocess_output(
        {
            "video": ["frame-0"],
            "audio": "audio-0",
            "actions": "action-0",
            "audio_sample_rate": 16000,
            "fps": 30.0,
        }
    )

    [result] = format_diffusion_outputs(
        request=_request("watch and listen"),
        od_config=_config("audio_video_model"),
        diffusion_output=DiffusionOutput(output=None),
        output_data={"raw": "output"},
        postprocess_output=postprocess_output,
    )

    assert result.images == ["frame-0"]
    assert result.prompt == "watch and listen"
    assert result.final_output_type == "image"
    assert result.multimodal_output == {
        "metadata": {
            "audio": {"sample_rate": 16000},
            "video": {"fps": 30.0},
        },
        "audio": "audio-0",
        "audio_sample_rate": 16000,
        "fps": 30.0,
        "actions": "action-0",
    }


def test_formatter_preserves_audio_only_postprocess_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: True)
    postprocess_output = normalize_diffusion_postprocess_output(
        {
            "audio": "waveform",
            "audio_sample_rate": 24000,
        }
    )

    [result] = format_diffusion_outputs(
        request=_request("speak"),
        od_config=_config("audio_model"),
        diffusion_output=DiffusionOutput(output=None),
        output_data={"raw": "output"},
        postprocess_output=postprocess_output,
    )

    assert result.images == []
    assert result.prompt == "speak"
    assert result.final_output_type == "audio"
    assert result.multimodal_output == {
        "metadata": {
            "audio": {"sample_rate": 24000},
        },
        "audio": "waveform",
        "audio_sample_rate": 24000,
    }


def test_formatter_preserves_single_prompt_multiple_audio_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: True)
    monkeypatch.setattr(
        output_formatter.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda _: None,
    )
    postprocess_output = normalize_diffusion_postprocess_output(["waveform-0", "waveform-1"])

    [result] = format_diffusion_outputs(
        request=_request("speak", num_outputs_per_prompt=2),
        od_config=_config("audio_model"),
        diffusion_output=DiffusionOutput(output=["waveform-0", "waveform-1"]),
        output_data=["waveform-0", "waveform-1"],
        postprocess_output=postprocess_output,
    )

    assert result.images == []
    assert result.prompt == "speak"
    assert result.final_output_type == "audio"
    assert result.multimodal_output == {"audio": ["waveform-0", "waveform-1"]}


def test_formatter_preserves_single_prompt_audio_and_action_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    postprocess_output = normalize_diffusion_postprocess_output(
        {
            "video": ["frame-0", "frame-1"],
            "audio": ["audio-0", "audio-1"],
            "actions": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "fps": 12.5,
        }
    )

    results = format_diffusion_outputs(
        request=_request("prompt-0"),
        od_config=_config(),
        diffusion_output=DiffusionOutput(output=None),
        output_data={"raw": "output"},
        postprocess_output=postprocess_output,
    )

    assert len(results) == 1
    assert results[0].images == ["frame-0", "frame-1"]
    assert results[0].prompt == "prompt-0"
    assert results[0].multimodal_output["metadata"] == {"video": {"fps": 12.5}}
    assert results[0].multimodal_output["audio"] == ["audio-0", "audio-1"]
    assert results[0].multimodal_output["fps"] == 12.5
    torch.testing.assert_close(
        results[0].multimodal_output["actions"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_format_empty_diffusion_outputs_preserves_empty_response_shape() -> None:
    results = format_empty_diffusion_outputs(_request("prompt-0"))

    assert len(results) == 1
    assert [result.prompt for result in results] == ["prompt-0"]
    assert [result.images for result in results] == [[]]
    assert [result.metrics for result in results] == [{}]
