# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for MOSS-VoiceGenerator (MossTTSDelayModel, 1.7B).

Weekly expansion coverage (``pytest.mark.slow``) for the delay-model path.
Upstream keeps the same tests in ``test_moss_tts.py``; this branch uses the
``*_expansion.py`` naming convention and deletes the base file on merge.

MOSS-VoiceGenerator synthesizes speech from a text *instruction* that describes
the desired voice (e.g. "a warm female voice with an American accent").  It does
NOT accept a reference audio clip.  The request format requires running
AutoProcessor.from_pretrained once per call to encode the (text, instruction)
pair into the (prompt_token_ids, codes.ref) grid that the MossTTSDelayModel
talker expects — same as examples/offline_inference/text_to_speech/moss_tts/
end2end.py:_build_unified_codes.

Realtime coverage: ``test_moss_tts_realtime_expansion.py``.

No determinism test: VoiceGenerator produces variable-length output even with
a fixed seed; waveform length reproducibility is not guaranteed.
"""

from __future__ import annotations

import gc
import os

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = "OpenMOSS-Team/MOSS-VoiceGenerator"
SAMPLE_RATE = 24_000

# Stage 0 = AR talker; max_tokens=512 keeps tests fast (~20 s of audio).
# Stage 1 = codec decoder; greedy, large context to hold all codec tokens.
_STAGE0_PARAMS = SamplingParams(
    temperature=1.7,
    top_p=0.8,
    top_k=25,
    max_tokens=512,
    seed=42,
    detokenize=False,
)
_STAGE1_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=65536,
    seed=42,
    detokenize=False,
)
_SAMPLING = [_STAGE0_PARAMS, _STAGE1_PARAMS]


def _get_test_config() -> str:
    """Derive a CI-friendly config from moss_voice_generator.yaml."""
    return modify_stage_config(
        get_deploy_config_path("moss_voice_generator.yaml"),
        updates={
            "stages": {
                0: {
                    "max_num_seqs": 1,
                    "gpu_memory_utilization": 0.45,
                },
            },
        },
    )


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/4700"),
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [(MODEL, _get_test_config())], indirect=True),
]


def _build_request(text: str, instruction: str) -> dict:
    """Build a VoiceGenerator request using AutoProcessor."""
    from transformers import AutoProcessor

    try:
        proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    except Exception as exc:
        msg = f"Cannot load AutoProcessor for {MODEL}: {exc}"
        if os.environ.get("MOSS_TTS_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)

    user_msg = proc.build_user_message(text=text, instruction=instruction)
    batch = proc(conversations=[[user_msg]], mode="generation")
    unified = batch["input_ids"][0]
    text_ids = unified[:, 0].tolist()
    audio_codes = unified[:, 1:].contiguous().to(torch.int64)
    del proc
    gc.collect()

    return {
        "prompt_token_ids": text_ids,
        "additional_information": {"codes": {"ref": audio_codes}},
    }


def _collect_audio(omni_runner: OmniRunner, request: dict) -> tuple[torch.Tensor, int]:
    chunks: list[torch.Tensor] = []
    sr_final = SAMPLE_RATE
    for out in omni_runner.omni.generate(request, _SAMPLING):
        mm = out.multimodal_output
        if not mm:
            continue
        audio = mm.get("audio")
        if audio is None:
            audio = mm.get("model_outputs")
        if audio is None:
            continue
        sr = mm.get("sr")
        if sr is not None:
            sr_final = int(sr.item() if hasattr(sr, "item") else sr)
        if isinstance(audio, list):
            audio = torch.cat(
                [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0],
                dim=0,
            )
        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
            chunks.append(audio.reshape(-1).cpu())
    if not chunks:
        raise AssertionError("No audio output received from generate()")
    return torch.cat(chunks, dim=0), sr_final


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_moss_tts_delay_english(omni_runner: OmniRunner) -> None:
    """VoiceGenerator: English instruction produces non-empty 24 kHz audio."""
    req = _build_request(
        "Hello, this is a MOSS voice design test.",
        "a warm female voice with an American accent",
    )
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_moss_tts_delay_chinese(omni_runner: OmniRunner) -> None:
    """VoiceGenerator: Chinese input produces non-empty audio."""
    req = _build_request(
        "你好，这是语音合成测试。",
        "一个清晰温暖的女声",
    )
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_moss_tts_delay_batch(omni_runner: OmniRunner) -> None:
    """VoiceGenerator: batch of two requests each returns non-empty audio."""
    requests = [
        _build_request("First sentence.", "a warm female voice"),
        _build_request("Second sentence.", "a young male voice"),
    ]
    results: list[torch.Tensor] = []
    for out in omni_runner.omni.generate(requests, _SAMPLING):
        mm = out.multimodal_output
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
        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
            results.append(audio.reshape(-1).cpu())

    assert len(results) >= 2, f"Expected at least 2 audio outputs, got {len(results)}"
    for i, audio in enumerate(results):
        assert audio.numel() > 0, f"Audio chunk[{i}] is empty"
