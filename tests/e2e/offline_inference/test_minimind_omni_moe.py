# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for the MiniMind-Omni MoE pipeline."""

from __future__ import annotations

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = os.environ.get("MINIMIND_OMNI_MOE_MODEL", "jingyaogong/minimind-3o-moe")
_CI_DEPLOY = get_deploy_config_path("ci/minimind_omni_moe.yaml")
_OUTPUT_DIR = Path(os.environ.get("MINIMIND_OMNI_MOE_OUTPUT_DIR", "/tmp/minimind_omni_moe_outputs"))
_AUDIO_SAMPLE_RATE = int(os.environ.get("MINIMIND_OMNI_MOE_SAMPLE_RATE", "24000"))


_DEFAULT_PROMPT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def assert_real_text_if_enabled(response: Any, run_level: str) -> None:
    if run_level not in {"advanced_model", "full_model"}:
        return
    assert response.text_content is not None
    assert response.text_content.strip()


def assert_real_audio_if_enabled(response: Any, run_level: str) -> None:
    if run_level not in {"advanced_model", "full_model"}:
        return
    audio = response.audio_content
    assert audio is not None
    if isinstance(audio, torch.Tensor):
        assert audio.numel() > 0


def build_minimind_prompt(prompt: str) -> str:
    template = os.environ.get("MINIMIND_OMNI_PROMPT_TEMPLATE", _DEFAULT_PROMPT_TEMPLATE)
    return template.format(prompt=prompt, user=prompt)


def run_minimind_request(omni_runner: Any, prompt: str, modalities: list[str]) -> SimpleNamespace:
    request = {
        "prompt": build_minimind_prompt(prompt),
        "modalities": modalities,
    }
    outputs = omni_runner.generate([request], omni_runner.get_default_sampling_params_list())

    response = SimpleNamespace(text_content=None, audio_content=None, success=False)
    for stage_output in outputs:
        request_output = stage_output.request_output
        output = request_output.outputs[0]
        if getattr(stage_output, "final_output_type", None) == "text":
            response.text_content = output.text
        elif getattr(stage_output, "final_output_type", None) == "audio":
            mm_output = output.multimodal_output or {}
            response.audio_content = mm_output.get("audio")

    response.success = ("text" not in modalities or response.text_content is not None) and (
        "audio" not in modalities or response.audio_content is not None
    )
    return response


def audio_to_numpy(audio: Any) -> np.ndarray:
    if isinstance(audio, list):
        tensors = [item.detach().cpu().reshape(-1) for item in audio if isinstance(item, torch.Tensor)]
        if tensors:
            audio = torch.cat(tensors, dim=0)

    if isinstance(audio, torch.Tensor):
        array = audio.detach().cpu().float().numpy()
    else:
        array = np.asarray(audio, dtype=np.float32)

    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 2 and array.shape[0] <= 2 and array.shape[1] > array.shape[0]:
        array = array.T
    elif array.ndim > 2:
        array = array.reshape(-1)
    return array


def save_response_outputs(
    response: Any,
    stem: str,
    *,
    output_dir: Path,
    sample_rate: int,
    log_prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if response.text_content is not None:
        text_path = output_dir / f"{stem}.txt"
        text_path.write_text(str(response.text_content), encoding="utf-8")
        print(f"{log_prefix} text output saved to: {text_path}")

    if response.audio_content is not None:
        audio_path = output_dir / f"{stem}.wav"
        sf.write(str(audio_path), audio_to_numpy(response.audio_content), sample_rate)
        print(f"{log_prefix} audio output saved to: {audio_path}")


def get_eager_config() -> str:
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: {"enforce_eager": True},
                1: {"enforce_eager": True},
                2: {"enforce_eager": True},
            },
        },
    )


test_params = [
    (
        MODEL,
        get_eager_config(),
        {"stage_init_timeout": 600, "init_timeout": 1200},
    )
]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, run_level) -> None:
    """Test MiniMind-Omni MoE thinker text generation."""
    response = run_minimind_request(
        omni_runner,
        prompt="请简单介绍一下你自己",
        modalities=["text"],
    )
    assert response.success
    save_response_outputs(
        response,
        "text_to_text",
        output_dir=_OUTPUT_DIR,
        sample_rate=_AUDIO_SAMPLE_RATE,
        log_prefix="MiniMind-Omni MoE",
    )
    assert_real_text_if_enabled(response, run_level)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, run_level) -> None:
    """Test MiniMind-Omni MoE thinker -> talker -> code2wav generation."""
    response = run_minimind_request(
        omni_runner,
        prompt="请简单介绍一下你自己",
        modalities=["audio"],
    )
    assert response.success
    save_response_outputs(
        response,
        "text_to_audio",
        output_dir=_OUTPUT_DIR,
        sample_rate=_AUDIO_SAMPLE_RATE,
        log_prefix="MiniMind-Omni MoE",
    )
    assert_real_audio_if_enabled(response, run_level)
