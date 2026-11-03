# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline E2E smoke test for Audex audio understanding (thinker-only).

Single-stage ``audex_thinker_only`` pipeline on
``checkpoint_folder_full``: WAV (+ instruction) in, text out. The input clip
is vLLM's public ``mary_had_lamb`` audio asset (downloaded on first use), so
its transcript is known.

WER is a trend metric (recorded in the assertion message, no numeric hard
gate); the hard gates are structural: non-empty, non-degenerate text.
"""

from __future__ import annotations

import os

import pytest
from vllm.assets.audio import AudioAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "nvidia/Nemotron-Labs-Audex-2B"
MODEL_DIR_ENV = "VLLM_OMNI_AUDEX_MODEL_DIR"

# Full transcript of the mary_had_lamb asset (Edison's phonograph recording);
# a shortened reference would count the extra spoken words as insertions and
# inflate WER past the trend threshold.
_REFERENCE = (
    "the first words i spoke in the original phonograph a little piece of practical poetry "
    "mary had a little lamb its fleece was white as snow and everywhere that mary went the "
    "lamb was sure to go"
)

ASR_PROMPT = (
    "<|im_start|>user\n<so_embedding>\nTranscribe the input speech.<|im_end|>\n<|im_start|>assistant\n<think></think>"
)

_audex_deployment = get_deploy_config_path("audex_thinker_only.yaml")
_audex_model = os.environ.get(MODEL_DIR_ENV) or MODEL
_OMNI_RUNNER_PARAMS = [
    pytest.param(
        (_audex_model, _audex_deployment, {"async_chunk": False}),
        id="thinker_only",
    ),
]
# 30B-A3B variant: opt-in (pulls ~60 GB); enable with AUDEX_E2E_30B=1.
_AUDEX_30B_MODEL = os.environ.get("VLLM_OMNI_AUDEX_30B_MODEL_DIR") or "nvidia/Nemotron-Labs-Audex-30B-A3B"
if os.environ.get("AUDEX_E2E_30B") == "1":
    _OMNI_RUNNER_PARAMS.append(
        pytest.param(
            (_AUDEX_30B_MODEL, get_deploy_config_path("audex_thinker_only_30b.yaml"), {"async_chunk": False}),
            id="thinker_only_30b",
        )
    )
pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
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


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_audex_offline_asr_smoke(omni_runner: OmniRunner, run_level: str) -> None:
    """Transcribing the public audio asset yields coherent, non-degenerate text."""
    audio, sr = AudioAsset("mary_had_lamb").audio_and_sample_rate
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
