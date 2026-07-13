# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for IndexTTS2 two-stage pipeline.

Stage 0 (GPT AR) → Stage 1 (S2Mel + BigVGAN). Output is 22050 Hz mono WAV.
Covers: basic TTS.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni import Omni
from vllm_omni.model_executor.models.indextts2.prompt_utils import (
    build_indextts2_prefill_prompt_ids,
)

MODEL_NAME = "IndexTeam/IndexTTS-2"
STAGE_CONFIG = get_deploy_config_path("indextts2.yaml")

_OMNI_RUNNER_PARAM = (
    MODEL_NAME,
    STAGE_CONFIG,
)

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "indextts2"
REF_AUDIO = str(ASSETS_DIR / "ref_audio.wav")

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]

SAMPLE_RATE = 22050


def _audio_from_mm(mm: dict) -> torch.Tensor | None:
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if isinstance(audio, list):
        chunks = [chunk.reshape(-1) for chunk in audio if isinstance(chunk, torch.Tensor) and chunk.numel() > 0]
        return torch.cat(chunks, dim=0) if chunks else None
    return audio if isinstance(audio, torch.Tensor) else None


def _sample_rate_from_mm(mm: dict) -> int:
    sr = mm.get("sr")
    if isinstance(sr, list):
        sr = sr[-1] if sr else None
    if hasattr(sr, "item"):
        return int(sr.item())
    return int(sr) if sr is not None else SAMPLE_RATE


def _build_request(text: str, ref_audio: str, **extra) -> dict:
    info = {
        "text": [text],
        "voice": [ref_audio],
    }
    info.update(extra)
    return {
        "prompt_token_ids": build_indextts2_prefill_prompt_ids(MODEL_NAME, text),
        "additional_information": info,
    }


def _collect_audio(omni: Omni, request: dict) -> tuple[torch.Tensor, int]:
    for omni_out in omni.generate(request):
        mm = omni_out.multimodal_output
        assert mm is not None, "Expected multimodal_output"
        audio = _audio_from_mm(mm)
        assert audio is not None, "Expected audio output"
        return audio.cpu(), _sample_rate_from_mm(mm)
    raise AssertionError("No outputs received")


def _assert_valid_audio(audio: torch.Tensor, sr: int) -> None:
    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert torch.any(audio != 0).item()


@hardware_test(res={"cuda": "L4"})
def test_basic_english(omni_runner: OmniRunner) -> None:
    """English TTS: text + ref_audio, no emotion conditioning."""
    req = _build_request("Hello, this is a voice synthesis test.", REF_AUDIO)
    audio, sr = _collect_audio(omni_runner.omni, req)
    _assert_valid_audio(audio, sr)
