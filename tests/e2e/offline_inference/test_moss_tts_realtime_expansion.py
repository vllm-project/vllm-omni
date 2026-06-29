# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for MOSS-TTS-Realtime (MossTTSRealtime, 1.7B).

Exercises the local-transformer generation path on L4. Delay-model coverage is in
``test_moss_tts_expansion.py``.

No determinism test: MossTTSRealtime is sensitive to async scheduling timing;
waveform reproducibility is not guaranteed across sequential ``generate`` calls.
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

MODEL = "OpenMOSS-Team/MOSS-TTS-Realtime"
DEPLOY_CONFIG = get_deploy_config_path("moss_tts_realtime.yaml")
_OMNI_RUNNER_PARAM = (MODEL, DEPLOY_CONFIG, {"stage_init_timeout": 300})

pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/4700"),
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
    cache_dir = tmp_path_factory.mktemp("moss_tts_realtime_ref")
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
def test_moss_tts_realtime_english(omni_runner: OmniRunner, ref_audio_path) -> None:
    """MossTTSRealtime: English voice_clone produces non-empty 24 kHz audio."""
    req = _build_request("This is a real-time TTS streaming test.", ref_audio_path)
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"
