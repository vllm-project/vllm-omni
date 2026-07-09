# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline E2E smoke test for Audex audio understanding (thinker-only).

Single-stage ``nemotron_labs_audex_thinker_only`` pipeline on
``checkpoint_folder_full``: WAV (+ instruction) in, text out. The vendored
asset was synthesized by the Audex TTS pipeline itself, so its transcript is
known exactly.

WER is a trend metric (recorded in the assertion message, no numeric hard
gate); the hard gates are structural: non-empty, non-degenerate text.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "nvidia/Nemotron-Labs-Audex-2B"
MODEL_DIR_ENV = "VLLM_OMNI_AUDEX_MODEL_DIR"

_ASSET = Path(__file__).resolve().parents[2] / "assets" / "audex" / "asr_weather_en.wav"
_REFERENCE = "the weather is so good today let us go hiking in the mountains"

ASR_PROMPT = (
    "<|im_start|>user\n<so_embedding>\nTranscribe the input speech.<|im_end|>\n<|im_start|>assistant\n<think></think>"
)

_audex_deployment = get_deploy_config_path("nemotron_labs_audex_thinker_only.yaml")
_audex_model = os.environ.get(MODEL_DIR_ENV) or MODEL
_OMNI_RUNNER_PARAMS = [
    pytest.param(
        (_audex_model, _audex_deployment, {"async_chunk": False}),
        id="thinker_only",
    ),
]
pytestmark = [
    pytest.mark.slow,
    pytest.mark.parametrize("omni_runner", _OMNI_RUNNER_PARAMS, indirect=True),
]


def _normalize(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text).split()


def _wer(hyp: str, ref: str) -> float:
    hyp_words, ref_words = _normalize(hyp), _normalize(ref)
    if not ref_words:
        return 0.0
    rows = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, 1):
        prev, rows[0] = rows[0], i
        for j, hyp_word in enumerate(hyp_words, 1):
            cur = min(
                rows[j] + 1,
                rows[j - 1] + 1,
                prev + (ref_word != hyp_word),
            )
            prev, rows[j] = rows[j], cur
    return rows[-1] / len(ref_words)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_audex_offline_asr_smoke(omni_runner: OmniRunner, run_level: str) -> None:
    """Transcribing the vendored WAV yields coherent, non-degenerate text."""
    audio, sr = sf.read(str(_ASSET), dtype="float32")
    outputs = omni_runner.omni.generate([{"prompt": ASR_PROMPT, "multi_modal_data": {"audio": (audio, sr)}}])

    assert len(outputs) == 1
    text = outputs[0].outputs[0].text if outputs[0].outputs else ""
    assert isinstance(text, str)

    if run_level in {"advanced_model", "full_model"}:
        words = _normalize(text)
        assert words, f"Empty transcription: {text!r}"
        # Degenerate repetition guard: no word may dominate the output.
        most_common = max(words.count(w) for w in set(words))
        assert most_common <= max(3, len(words) // 2), f"Degenerate transcript: {text!r}"
        wer = _wer(text, _REFERENCE)
        # Trend metric (recorded, not gated): substantially-correct check only.
        assert wer <= 0.9, f"Transcript unrelated to reference (WER={wer:.2f}): {text!r}"
        print(f"[trend] Audex S2T WER vs reference: {wer:.3f} ({text!r})")
