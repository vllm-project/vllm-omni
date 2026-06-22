# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for MOSS-TTS delay pipeline (MossTTSDelayModel).

Uses MOSS-VoiceGenerator (1.7B) — the smallest MossTTSDelayModel variant — so
the test runs on a single L4 without requiring an 80 GB GPU.

MOSS-TTS-Realtime coverage lives in ``test_moss_tts_realtime_expansion.py`` (one
module-scoped ``omni_runner`` per file; see skill invariant I4).

No determinism test: MossTTSDelayModel with stochastic talker sampling produces
variable-length output even with a fixed request seed; bit-exact waveform
reproducibility is not guaranteed across sequential ``generate`` calls.
"""

from __future__ import annotations

import os
import urllib.request

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "OpenMOSS-Team/MOSS-VoiceGenerator"
DEPLOY_CONFIG = get_deploy_config_path("moss_voice_generator.yaml")
_OMNI_RUNNER_PARAM = (MODEL, DEPLOY_CONFIG, {"stage_init_timeout": 300})

pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]

SAMPLE_RATE = 24_000
REF_AUDIO_URL = "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/assets/audio/reference_zh_1.wav"

_DEFAULT_SAMPLING = SamplingParams(
    temperature=1.7,
    top_p=0.8,
    top_k=25,
    max_tokens=512,
    seed=42,
    detokenize=False,
)


@pytest.fixture(scope="session")
def ref_audio_path(tmp_path_factory) -> str:
    """Download the upstream reference clip once per session."""
    cache_dir = tmp_path_factory.mktemp("moss_tts_ref")
    target = cache_dir / "zh_1.wav"
    try:
        with urllib.request.urlopen(REF_AUDIO_URL, timeout=30) as resp:
            target.write_bytes(resp.read())
    except Exception as exc:
        msg = f"Cannot fetch reference clip {REF_AUDIO_URL}: {exc}"
        if os.environ.get("MOSS_TTS_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)
    if not target.exists() or target.stat().st_size == 0:
        pytest.fail(f"Reference clip empty after download: {target}")
    return str(target)


def _build_request(text: str, ref_audio_path: str, seed: int = 42) -> dict:
    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": {
            "task_type": ["voice_clone"],
            "text": [text],
            "mode": ["voice_clone"],
            "prompt_audio_path": [ref_audio_path],
            "seed": [seed],
        },
    }


def _sampling_for(omni_runner: OmniRunner) -> SamplingParams | list[SamplingParams]:
    omni = omni_runner.omni
    if omni.num_stages == 1:
        return _DEFAULT_SAMPLING
    params = omni_runner.get_default_sampling_params_list()
    params[0] = _DEFAULT_SAMPLING
    return params


def _audio_from_stage(stage_outputs) -> tuple[torch.Tensor, int] | None:
    mm = stage_outputs.multimodal_output
    if not mm:
        return None
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if audio is None:
        return None
    if isinstance(audio, list):
        audio = torch.cat(
            [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0],
            dim=0,
        )
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        return None
    sr = mm.get("sr")
    sample_rate = int(sr.item()) if sr is not None else SAMPLE_RATE
    return audio.reshape(-1).cpu(), sample_rate


def _collect_audio(omni_runner: OmniRunner, request: dict) -> tuple[torch.Tensor, int]:
    for stage_outputs in omni_runner.omni.generate(request, _sampling_for(omni_runner)):
        parsed = _audio_from_stage(stage_outputs)
        if parsed is not None:
            return parsed
    raise AssertionError("No stage outputs received")


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_english(omni_runner: OmniRunner, ref_audio_path) -> None:
    """MossTTSDelayModel: English voice_clone produces non-empty 24 kHz audio."""
    req = _build_request("Hello, this is a MOSS-TTS voice cloning test.", ref_audio_path)
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_chinese(omni_runner: OmniRunner, ref_audio_path) -> None:
    """MossTTSDelayModel: Chinese input produces non-empty audio."""
    req = _build_request("你好，这是语音合成测试。", ref_audio_path)
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_delay_batch(omni_runner: OmniRunner, ref_audio_path) -> None:
    """MossTTSDelayModel: batch of two requests each returns non-empty audio."""
    requests = [
        _build_request("First sentence.", ref_audio_path),
        _build_request("Second sentence.", ref_audio_path),
    ]
    results: list[torch.Tensor] = []
    for stage_outputs in omni_runner.omni.generate(requests, _sampling_for(omni_runner)):
        parsed = _audio_from_stage(stage_outputs)
        if parsed is not None:
            results.append(parsed[0])

    assert len(results) == 2, f"Expected 2 outputs, got {len(results)}"
    for i, audio in enumerate(results):
        assert audio.numel() > 0, f"Audio[{i}] is empty"
