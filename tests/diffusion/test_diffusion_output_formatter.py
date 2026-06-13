# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion import output_formatter
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.output_formatter import (
    DiffusionStepTimings,
    format_diffusion_outputs,
    format_empty_diffusion_outputs,
    normalize_diffusion_postprocess_output,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _request(
    prompts: list[str] | None = None,
    *,
    num_outputs_per_prompt: int = 1,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=prompts or ["prompt"],
        request_id="req-1",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=1,
            num_outputs_per_prompt=num_outputs_per_prompt,
            resolution=512,
        ),
    )


def _config(model_class_name: str = "mock_model") -> SimpleNamespace:
    return SimpleNamespace(model_class_name=model_class_name)


def _timings() -> DiffusionStepTimings:
    return DiffusionStepTimings(
        preprocess_time_s=0.01,
        exec_time_s=0.02,
        postprocess_time_s=0.03,
        total_time_ms=60.0,
    )


def test_formatter_preserves_single_video_audio_actions_and_custom_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    custom_output = {"from_output": "kept"}
    postprocess_output = normalize_diffusion_postprocess_output(
        {
            "video": ["frame-0"],
            "audio": "audio-0",
            "actions": "action-0",
            "custom_output": {"from_postprocess": "merged"},
            "audio_sample_rate": 48000,
            "fps": 24.0,
        },
        custom_output,
    )

    results = format_diffusion_outputs(
        request=_request(["prompt-0"]),
        od_config=_config(),
        diffusion_output=DiffusionOutput(
            output=None,
            custom_output=custom_output,
            stage_durations={"execute": 1.25},
            peak_memory_mb=321.0,
        ),
        output_data={"raw": "output"},
        postprocess_output=postprocess_output,
        timings=_timings(),
    )

    assert len(results) == 1
    result = results[0]
    assert result.images == ["frame-0"]
    assert result.prompt == "prompt-0"
    assert result.final_output_type == "image"
    assert result.custom_output == {
        "from_output": "kept",
        "from_postprocess": "merged",
    }
    assert custom_output == {"from_output": "kept"}
    assert result.multimodal_output == {
        "audio": "audio-0",
        "audio_sample_rate": 48000,
        "fps": 24.0,
        "actions": "action-0",
    }
    assert result.stage_durations == {"execute": 1.25}
    assert result.peak_memory_mb == 321.0
    assert result.metrics == {
        "preprocess_time_ms": 10.0,
        "diffusion_engine_exec_time_ms": 20.0,
        "diffusion_engine_total_time_ms": 60.0,
        "image_num": 1,
        "resolution": 512,
        "postprocess_time_ms": 30.0,
    }


def test_formatter_preserves_text_custom_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    postprocess_output = normalize_diffusion_postprocess_output(
        "caption",
        {"text_output": "caption"},
    )

    [result] = format_diffusion_outputs(
        request=_request(["describe this"]),
        od_config=_config(),
        diffusion_output=DiffusionOutput(output="caption"),
        output_data="caption",
        postprocess_output=postprocess_output,
        timings=_timings(),
    )

    assert result.images == []
    assert result.prompt == "describe this"
    assert result.final_output_type == "text"
    assert result.custom_output == {"text_output": "caption"}


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
    postprocess_output = normalize_diffusion_postprocess_output(["waveform"], {})

    [result] = format_diffusion_outputs(
        request=_request(["speak"]),
        od_config=_config("audio_model"),
        diffusion_output=DiffusionOutput(output=["waveform"]),
        output_data=["waveform"],
        postprocess_output=postprocess_output,
        timings=_timings(),
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
        },
        {},
    )

    [result] = format_diffusion_outputs(
        request=_request(["watch and listen"]),
        od_config=_config("audio_video_model"),
        diffusion_output=DiffusionOutput(output=None),
        output_data={"raw": "output"},
        postprocess_output=postprocess_output,
        timings=_timings(),
    )

    assert result.images == ["frame-0"]
    assert result.prompt == "watch and listen"
    assert result.final_output_type == "image"
    assert result.multimodal_output == {
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
        },
        {},
    )

    [result] = format_diffusion_outputs(
        request=_request(["speak"]),
        od_config=_config("audio_model"),
        diffusion_output=DiffusionOutput(output=None),
        output_data={"raw": "output"},
        postprocess_output=postprocess_output,
        timings=_timings(),
    )

    assert result.images == []
    assert result.prompt == "speak"
    assert result.final_output_type == "audio"
    assert result.multimodal_output == {
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
    postprocess_output = normalize_diffusion_postprocess_output(["waveform-0", "waveform-1"], {})

    [result] = format_diffusion_outputs(
        request=_request(["speak"], num_outputs_per_prompt=2),
        od_config=_config("audio_model"),
        diffusion_output=DiffusionOutput(output=["waveform-0", "waveform-1"]),
        output_data=["waveform-0", "waveform-1"],
        postprocess_output=postprocess_output,
        timings=_timings(),
    )

    assert result.images == []
    assert result.prompt == "speak"
    assert result.final_output_type == "audio"
    assert result.multimodal_output == {"audio": ["waveform-0", "waveform-1"]}


def test_formatter_preserves_multi_prompt_audio_and_action_slicing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    postprocess_output = normalize_diffusion_postprocess_output(
        {
            "video": ["frame-0", "frame-1"],
            "audio": ["audio-0", "audio-1"],
            "actions": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "custom_output": {"shared": True},
            "fps": 12.5,
        },
        {},
    )

    results = format_diffusion_outputs(
        request=_request(["prompt-0", "prompt-1"]),
        od_config=_config(),
        diffusion_output=DiffusionOutput(output=None),
        output_data={"raw": "output"},
        postprocess_output=postprocess_output,
        timings=_timings(),
    )

    assert len(results) == 2
    assert results[0].images == ["frame-0"]
    assert results[1].images == ["frame-1"]
    assert results[0].custom_output == {"shared": True}
    assert results[1].custom_output == {"shared": True}
    assert results[0].multimodal_output["audio"] == "audio-0"
    assert results[1].multimodal_output["audio"] == "audio-1"
    assert results[0].multimodal_output["fps"] == 12.5
    assert results[1].multimodal_output["fps"] == 12.5
    torch.testing.assert_close(
        results[0].multimodal_output["actions"],
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        results[1].multimodal_output["actions"],
        torch.tensor([3.0, 4.0]),
    )


def test_format_empty_diffusion_outputs_preserves_empty_response_shape() -> None:
    results = format_empty_diffusion_outputs(_request(["prompt-0", "prompt-1"]))

    assert len(results) == 2
    assert [result.prompt for result in results] == ["prompt-0", "prompt-1"]
    assert [result.images for result in results] == [[], []]
    assert [result.metrics for result in results] == [{}, {}]
