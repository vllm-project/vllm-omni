# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline E2E smoke test for AuK (encoder stage -> diffusion stage).

AuK ships as ``config.yaml`` plus loose safetensors and needs the Qwen2.5-Omni-3B
snapshot next to it, so the test expects an assembled directory produced by
``tools/prepare_auk_checkpoint.py`` and skips when it is not available:

    VLLM_OMNI_AUK_MODEL_DIR=/path/to/auk-omni python -m pytest tests/e2e/offline_inference/test_auk.py

Two requests are exercised: zero-shot TTS from a reference clip and text-only
instruct TTS. The checks are structural (duration, sample rate, level,
finiteness); fidelity against the upstream implementation is covered by the
parity suites under ``tests/diffusion/models/auk``.
"""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_extras.auk import HOP, SAMPLE_RATE, auk_prompt, auk_sampling_params

MODEL_DIR_ENV = "VLLM_OMNI_AUK_MODEL_DIR"

# Any clean speech clip works as the cloning reference; reuse the vendored
# CosyVoice3 prompt so no new binary asset is needed.
REFERENCE_WAV_PATH = get_asset_path("cosyvoice3/zero_shot_prompt.wav")
ZERO_SHOT_TEXT = "The quick brown fox jumps over the lazy dog."
ZERO_SHOT_SECONDS = 3.0
INSTRUCT_TEXT = (
    'Please generate speech content "Welcome back, how was your day?" based on the following '
    'description: "A calm young woman speaking warmly and slowly."'
)
INSTRUCT_SECONDS = 2.5
# 8 Euler steps keep the test short; the base recipe is 32.
TEST_NFE = 8

_model_dir = os.environ.get(MODEL_DIR_ENV)
pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.skipif(not _model_dir, reason=f"set {MODEL_DIR_ENV} to an assembled AuK directory"),
    pytest.mark.parametrize(
        "omni_runner",
        [pytest.param((str(Path(_model_dir or ".").resolve()), get_deploy_config_path("auk.yaml"), {}), id="auk")],
        indirect=True,
    ),
]


@functools.lru_cache(maxsize=1)
def _load_reference_wav() -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(REFERENCE_WAV_PATH), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _audio_from_output(output) -> tuple[np.ndarray, int]:
    mm = getattr(output, "multimodal_output", None) or {}
    assert "audio" in mm, f"no audio in the final output; keys={sorted(mm)}"
    audio = mm["audio"]
    if isinstance(audio, list):
        audio = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in audio])
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = mm.get("audio_sample_rate", SAMPLE_RATE)
    if isinstance(sr, list) and sr:
        sr = sr[-1]
    if hasattr(sr, "item"):
        sr = sr.item()
    return audio, int(sr)


def _check_waveform(audio: np.ndarray, sr: int, *, gen_seconds: float) -> None:
    assert sr == SAMPLE_RATE
    expected = math.ceil(gen_seconds * SAMPLE_RATE / HOP) * HOP
    assert audio.size == expected, f"expected {expected} samples for {gen_seconds}s, got {audio.size}"
    assert np.isfinite(audio).all(), "non-finite samples"
    assert np.abs(audio).max() <= 1.0 + 1e-6, "audio exceeds full scale"
    assert float(np.sqrt(np.mean(audio**2))) > 1e-3, "audio is silent"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_auk_zero_shot_tts(omni_runner: OmniRunner) -> None:
    """A reference clip plus an instruction yields speech of the requested length."""
    wav, sr = _load_reference_wav()
    prompt = auk_prompt(
        f"Say the following with the same voice: '{ZERO_SHOT_TEXT}'",
        (wav, sr),
        gen_seconds=ZERO_SHOT_SECONDS,
    )
    outputs = omni_runner.omni.generate(prompt, auk_sampling_params(nfe=TEST_NFE, cfg=2.0, seed=0))
    assert outputs, "no outputs returned"
    audio, out_sr = _audio_from_output(outputs[0])
    _check_waveform(audio, out_sr, gen_seconds=ZERO_SHOT_SECONDS)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_auk_instruct_tts_text_only(omni_runner: OmniRunner) -> None:
    """A text-only request (no reference audio) takes the no-prompt-audio path."""
    prompt = auk_prompt(INSTRUCT_TEXT, None, gen_seconds=INSTRUCT_SECONDS)
    assert prompt["prompt"].rstrip().endswith("<|im_start|>assistant\n".rstrip())
    assert "|<no_prompt_audio>|" in prompt["prompt"]
    outputs = omni_runner.omni.generate(prompt, auk_sampling_params(nfe=TEST_NFE, cfg=2.0, seed=0))
    assert outputs, "no outputs returned"
    audio, out_sr = _audio_from_output(outputs[0])
    _check_waveform(audio, out_sr, gen_seconds=INSTRUCT_SECONDS)
