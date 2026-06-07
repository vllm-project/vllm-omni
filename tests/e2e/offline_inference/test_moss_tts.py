# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for MOSS-VoiceGenerator (MossTTSDelayModel, 1.7B).

Uses the standard omni_runner + pytestmark pattern (one module-scoped engine
per file, skill invariant I4).  The realtime variant lives in the sibling file
test_moss_tts_realtime.py.

MOSS-VoiceGenerator synthesizes speech from a text *instruction* that describes
the desired voice (e.g. "a warm female voice with an American accent").  It does
NOT accept a reference audio clip.  The request format requires running
AutoProcessor.from_pretrained once per call to encode the (text, instruction)
pair into the (prompt_token_ids, codes.ref) grid that the MossTTSDelayModel
talker expects — same as examples/offline_inference/text_to_speech/moss_tts/
end2end.py:_build_unified_codes.

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Deploy config
# ---------------------------------------------------------------------------


def _get_test_config() -> str:
    """Derive a CI-friendly config from moss_voice_generator.yaml.

    Reduces Stage 0 gpu_memory_utilization from 0.60 → 0.45 to leave headroom
    for Stage 1 (0.12) on a shared L4/A10G.  max_num_seqs=1 keeps per-test
    peak memory predictable.
    """
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


# ---------------------------------------------------------------------------
# pytestmark — one engine for the whole module
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [(MODEL, _get_test_config())], indirect=True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_request(text: str, instruction: str) -> dict:
    """Build a VoiceGenerator request using AutoProcessor.

    Calls proc.build_user_message + proc() to produce the unified-codes grid
    (L, 1+n_vq) and splits it into prompt_token_ids + codes.ref, exactly as
    end2end.py:_build_unified_codes does for the voice_generator variant.
    The processor (including its CPU-resident audio_tokenizer) is freed before
    returning so it does not compete with the running engine.

    Set MOSS_TTS_SKIP_ON_NET_FAIL=1 to skip in air-gapped environments where
    the HF snapshot is not cached.
    """
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
    unified = batch["input_ids"][0]  # (L, 1+n_vq)
    text_ids = unified[:, 0].tolist()
    audio_codes = unified[:, 1:].contiguous().to(torch.int64)  # (L, n_vq)
    del proc
    gc.collect()

    return {
        "prompt_token_ids": text_ids,
        "additional_information": {"codes": {"ref": audio_codes}},
    }


def _collect_audio(omni_runner: OmniRunner, request: dict) -> tuple[torch.Tensor, int]:
    """Run one request and return (waveform_cpu, sample_rate).

    Iterates all OmniRequestOutputs from generate() via the top-level
    multimodal_output property (which aggregates from completion outputs).
    AR-stage outputs have empty multimodal_output and are silently skipped.
    With async_chunk=True and FINAL_ONLY, the codec stage consolidates its
    chunks into a single tensor before yielding — but we handle list fallback
    for robustness.
    """
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_moss_tts_english(omni_runner: OmniRunner) -> None:
    """VoiceGenerator: English instruction produces non-empty 24 kHz audio."""
    req = _build_request(
        "Hello, this is a MOSS voice design test.",
        "a warm female voice with an American accent",
    )
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_moss_tts_chinese(omni_runner: OmniRunner) -> None:
    """VoiceGenerator: Chinese input produces non-empty audio."""
    req = _build_request(
        "你好，这是语音合成测试。",
        "一个清晰温暖的女声",
    )
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_moss_tts_batch(omni_runner: OmniRunner) -> None:
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

    # With async_chunk streaming + FINAL_ONLY, each request produces one
    # consolidated output. The >= 2 guard handles backward-compatibility if
    # consolidation behavior changes.
    assert len(results) >= 2, f"Expected at least 2 audio outputs, got {len(results)}"
    for i, audio in enumerate(results):
        assert audio.numel() > 0, f"Audio chunk[{i}] is empty"
