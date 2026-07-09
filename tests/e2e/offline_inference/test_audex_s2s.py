# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline E2E test for the Audex cascaded speech-to-speech pipeline.

One ``nemotron_labs_audex_full`` deployment serves all three official
passes. The routing contract is the hard gate here: text-modality passes
(ASR, chat) finish at stage 0 and must never produce stage-1 audio; the
audio-modality TTS pass streams through the causal speech decoder.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt

MODEL = "nvidia/Nemotron-Labs-Audex-2B"
MODEL_DIR_ENV = "VLLM_OMNI_AUDEX_MODEL_DIR"
SAMPLE_RATE = 16_000

_ASSET = Path(__file__).resolve().parents[2] / "assets" / "audex" / "asr_weather_en.wav"

ASR_PROMPT = (
    "<|im_start|>user\n<so_embedding>\nTranscribe the input speech.<|im_end|>\n<|im_start|>assistant\n<think></think>"
)

_audex_deployment = get_deploy_config_path("nemotron_labs_audex_full.yaml")
_audex_model = os.environ.get(MODEL_DIR_ENV) or MODEL
_OMNI_RUNNER_PARAMS = [
    pytest.param(
        (_audex_model, _audex_deployment, {"async_chunk": True}),
        id="s2s_full",
    ),
]
pytestmark = [
    pytest.mark.slow,
    pytest.mark.parametrize("omni_runner", _OMNI_RUNNER_PARAMS, indirect=True),
]


def _concat_audio(audio_val) -> np.ndarray:
    if isinstance(audio_val, list):
        tensors = [t.detach().cpu().float().reshape(-1) for t in audio_val if isinstance(t, torch.Tensor)]
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).numpy().astype(np.float32, copy=False)
    if isinstance(audio_val, torch.Tensor):
        return audio_val.detach().cpu().float().reshape(-1).numpy()
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_audex_s2s_cascade_round_trip(omni_runner: OmniRunner, run_level: str) -> None:
    """ASR → chat → TTS over one deployment; text passes never emit audio."""
    real_weights = run_level in {"advanced_model", "full_model"}
    audio, sr = sf.read(str(_ASSET), dtype="float32")

    # Pass 1 — ASR (text-final).
    asr_outputs = omni_runner.omni.generate(
        [
            {
                "prompt": ASR_PROMPT,
                "multi_modal_data": {"audio": (audio, sr)},
                "modalities": ["text"],
            }
        ]
    )
    assert len(asr_outputs) == 1
    asr_out = asr_outputs[0]
    transcript = (asr_out.outputs[0].text or "").strip()
    mm = getattr(asr_out, "multimodal_output", None) or {}
    stage1_audio = _concat_audio(mm.get("audio")) if "audio" in mm else np.zeros((0,), dtype=np.float32)
    assert stage1_audio.size == 0, "Text-modality ASR pass produced stage-1 audio (routing regression)"

    # Pass 2 — chat (text-final).
    chat_prompt = f"<|im_start|>user\n{transcript or 'Hello.'}<|im_end|>\n<|im_start|>assistant\n<think></think>"
    chat_outputs = omni_runner.omni.generate([{"prompt": chat_prompt, "modalities": ["text"]}])
    answer = (chat_outputs[0].outputs[0].text or "").strip()

    if real_weights:
        assert transcript, "ASR pass produced an empty transcript"
        assert answer, "Chat pass produced an empty answer"

    # Pass 3 — TTS (audio-final; streams through code2wav). Both final
    # stages may emit an output record; the audio-bearing one must exist.
    tts_text = answer if answer else "The weather is nice today."
    tts_outputs = omni_runner.omni.generate([{"prompt": build_cond_prompt(tts_text[:200]), "modalities": ["audio"]}])
    assert tts_outputs, "TTS pass returned no outputs"
    speech = np.zeros((0,), dtype=np.float32)
    for req_output in tts_outputs:
        mm = getattr(req_output, "multimodal_output", None) or {}
        if "audio" in mm:
            speech = _concat_audio(mm["audio"])
            break
    assert speech.size > 0, "TTS pass produced empty audio"

    if real_weights:
        rms = float(np.sqrt(np.mean(np.square(speech))))
        assert rms > 1e-3, f"TTS-pass audio near-silent (rms={rms})"
