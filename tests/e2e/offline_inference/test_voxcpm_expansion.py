# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E test for VoxCPM offline inference."""

from typing import Any

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

VOXCPM_MODEL = "OpenBMB/VoxCPM1.5"
STAGE_CONFIG = get_deploy_config_path("voxcpm.yaml")
SAMPLE_RATE = 24000

# (model, stage_config_path) for ``@pytest.mark.parametrize("omni_runner", ..., indirect=True)``
_OMNI_RUNNER_PARAM = (VOXCPM_MODEL, STAGE_CONFIG)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


def _build_prompt(text: str) -> dict[str, Any]:
    return {
        "prompt_token_ids": [1],
        "additional_information": {
            "text": [text],
            "cfg_value": [2.0],
            "inference_timesteps": [10],
            "min_len": [2],
            "max_new_tokens": [1024],
        },
    }


def _extract_audio_tensor(multimodal_output: dict[str, Any]) -> torch.Tensor:
    audio = multimodal_output.get("audio", multimodal_output.get("model_outputs"))
    assert audio is not None, f"No audio output found, keys={list(multimodal_output.keys())}"

    if isinstance(audio, list):
        parts: list[torch.Tensor] = []
        for item in audio:
            if item is None:
                continue
            tensor = torch.as_tensor(item)
            if tensor.numel() == 0:
                continue
            parts.append(tensor.float().cpu().reshape(-1))
        return torch.cat(parts, dim=-1) if parts else torch.zeros((0,), dtype=torch.float32)

    return torch.as_tensor(audio).float().cpu().reshape(-1)


def _extract_final_multimodal_output(outputs) -> dict[str, Any]:
    for item in reversed(outputs):
        request_output = getattr(item, "request_output", None)
        if request_output is not None:
            multimodal_output = getattr(request_output, "multimodal_output", None)
            if isinstance(multimodal_output, dict):
                return multimodal_output
            completions = getattr(request_output, "outputs", None) or []
            for completion in completions:
                multimodal_output = getattr(completion, "multimodal_output", None)
                if isinstance(multimodal_output, dict):
                    return multimodal_output

        multimodal_output = getattr(item, "multimodal_output", None)
        if isinstance(multimodal_output, dict):
            return multimodal_output

    raise AssertionError("No multimodal audio output found in VoxCPM generate results")


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_voxcpm_zero_shot_001(omni_runner: OmniRunner) -> None:
    outputs = list(omni_runner.omni.generate(_build_prompt("Hello, this is a VoxCPM offline inference test.")))

    assert outputs, "No outputs returned"

    multimodal_output = _extract_final_multimodal_output(outputs)
    audio = _extract_audio_tensor(multimodal_output)
    assert audio.numel() > SAMPLE_RATE // 2, f"Audio too short: {audio.numel()} samples"

    duration_s = audio.shape[0] / SAMPLE_RATE
    assert 0.5 < duration_s < 30.0, f"Audio duration out of range: {duration_s:.2f}s"

    peak = float(torch.max(torch.abs(audio)).item()) if audio.numel() > 0 else 0.0
    assert peak > 0.01, "Generated audio appears to be silence"

    audio_np = audio.numpy()
    rms = float(np.sqrt(np.mean(np.square(audio_np)))) if audio_np.size else 0.0
    assert rms > 1e-4, "Generated audio RMS too low"
