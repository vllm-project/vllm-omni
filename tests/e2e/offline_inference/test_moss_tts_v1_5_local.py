# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for MOSS-TTS-Local-Transformer-v1.5.

MOSS-TTS-Local-Transformer-v1.5 (``MossTTSLocalModel``) is the Qwen3-backbone
variant with a 1-layer GPT2-style local depth transformer (n_vq=12, 48 kHz
stereo codec). It ships its own ``AutoProcessor`` (``processing_moss_tts.py``)
and reuses the same delay-style voice_clone code path as MOSS-TTS-v1.5 —
just a different n_vq/codec and output sample rate (48 kHz vs 24 kHz).
"""

from __future__ import annotations

import gc
import math
import struct
import wave
from pathlib import Path

import pytest
import torch
from transformers import AutoProcessor
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
DEPLOY_CONFIG = get_deploy_config_path("ci/moss_tts_local.yaml")
_OMNI_RUNNER_PARAM = (
    MODEL,
    DEPLOY_CONFIG,
    {"stage_init_timeout": 600, "trust_remote_code": True},
)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]

SAMPLE_RATE = 48000
_REF_SAMPLE_RATE = 48000
_REF_DURATION_S = 2.0

_DEFAULT_SAMPLING = SamplingParams(
    temperature=1.7,
    top_p=0.8,
    top_k=25,
    max_tokens=2048,
    seed=42,
    detokenize=False,
)


def _write_synthetic_reference_wav(path: Path) -> None:
    """Write a deterministic voice-like reference clip (stdlib only).

    The clip is a decaying harmonic stack (220/440/660/880 Hz) with a slow
    amplitude envelope, 48 kHz mono 16-bit PCM. CI is network-restricted, so
    the reference is synthesized locally instead of fetched from GitHub.
    """
    n_samples = int(_REF_SAMPLE_RATE * _REF_DURATION_S)
    frames = bytearray()
    partials = (
        (220.0, 0.6, 0.0),
        (440.0, 0.3, 0.0),
        (660.0, 0.15, 0.0),
        (880.0, 0.08, 0.5),
    )
    for i in range(n_samples):
        t = i / _REF_SAMPLE_RATE
        envelope = math.exp(-3.0 * t / _REF_DURATION_S) * (
            0.5 + 0.5 * math.sin(2.0 * math.pi * 3.5 * t)
        )
        sample = sum(amp * math.sin(2.0 * math.pi * freq * t + phase) for freq, amp, phase in partials)
        sample = max(-1.0, min(1.0, sample * envelope))
        frames.extend(struct.pack("<h", int(sample * 32767)))
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(_REF_SAMPLE_RATE)
        f.writeframes(bytes(frames))


@pytest.fixture(scope="session")
def ref_audio_path(tmp_path_factory) -> str:
    target = tmp_path_factory.mktemp("moss_tts_local_ref") / "zh_reference.wav"
    _write_synthetic_reference_wav(target)
    return str(target)


def _build_request(text: str, ref_audio_path: str, language: str = "English") -> dict:
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    message = processor.build_user_message(
        text=text,
        reference=[ref_audio_path],
        language=language,
    )
    unified = processor(conversations=[[message]], mode="generation")["input_ids"][0]
    del processor
    gc.collect()

    return {
        "prompt_token_ids": unified[:, 0].tolist(),
        "additional_information": {"codes": {"ref": unified[:, 1:].contiguous().to(torch.int64)}},
    }


def _sampling_for(omni_runner: OmniRunner) -> SamplingParams | list[SamplingParams]:
    omni = omni_runner.omni
    if omni.num_stages == 1:
        return _DEFAULT_SAMPLING
    params = omni_runner.get_default_sampling_params_list()
    params[0] = _DEFAULT_SAMPLING
    return params


def _collect_audio(omni_runner: OmniRunner, request: dict) -> tuple[torch.Tensor, int]:
    for stage_outputs in omni_runner.omni.generate(request, _sampling_for(omni_runner)):
        mm = stage_outputs.multimodal_output
        if not mm:
            continue
        audio = mm.get("audio")
        if audio is None:
            audio = mm.get("model_outputs")
        if audio is None:
            continue
        if isinstance(audio, list):
            audio = torch.cat(
                [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0],
                dim=0,
            )
        if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
            continue
        sr = mm.get("sr")
        return audio.reshape(-1).cpu(), int(sr.item()) if sr is not None else SAMPLE_RATE
    raise AssertionError("No stage outputs received")


@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"})
def test_moss_tts_v15_local_voice_clone(omni_runner: OmniRunner, ref_audio_path) -> None:
    """MOSS-TTS-Local-Transformer-v1.5: voice_clone produces non-empty 48 kHz audio."""
    req = _build_request("Hello, this is a MOSS-TTS Local voice cloning test.", ref_audio_path)
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"


@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"})
def test_moss_tts_v15_local_voice_clone_chinese(omni_runner: OmniRunner, ref_audio_path) -> None:
    """Chinese text produces non-empty 48 kHz audio."""
    req = _build_request("你好，这是本地延迟模型的语音克隆测试。", ref_audio_path, language="Chinese")
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)