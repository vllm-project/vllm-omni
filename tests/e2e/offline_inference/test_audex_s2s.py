# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline E2E test for the Audex cascaded speech-to-speech pipeline.

One ``audex_s2s`` deployment serves all three official
passes. The routing contract is the hard gate here: text-modality passes
(ASR, chat) finish at stage 0 and must never produce stage-1 audio; the
audio-modality TTS pass streams through the causal speech decoder.
"""

from __future__ import annotations

import copy
import os
import re
import time

import numpy as np
import pytest
from vllm.assets.audio import AudioAsset

from tests.helpers.mark import hardware_test
from tests.helpers.media import concat_audio
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt

MODEL = "nvidia/Nemotron-Labs-Audex-2B"
MODEL_DIR_ENV = "VLLM_OMNI_AUDEX_MODEL_DIR"
SAMPLE_RATE = 16_000

ASR_PROMPT = (
    "<|im_start|>user\n<so_embedding>\nTranscribe the input speech.<|im_end|>\n<|im_start|>assistant\n<think></think>"
)

# Official chat-pass recipe (examples/offline_inference/audex/speech_to_speech.py,
# mirroring the official cascaded web demo) — 30B-A3B ONLY. The
# formatting-instruction prefix is load-bearing on the 30B-A3B: without it a bare
# transcript turn is treated as a TTS request and the model emits literal
# <speechcodec_N> tokens instead of text, and under the yaml's near-greedy
# defaults its reasoning block never closes. The 2B must NOT use this recipe:
# it is long-verified stable on a bare prompt + engine-default sampling, while
# the official temp-1.0 seed-0 draw hallucinates off-language text (yet closes
# </think> cleanly, so the seed retry never fires) and fails the round-trip
# WER gate — measured, reproduced byte-identically.
SYSTEM_PROMPT = "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools."
TEXT_INSTRUCTION = (
    "SYSTEM & FORMATTING INSTRUCTION: You are Audex, created by NVIDIA based on the Nemotron-Cascade-2 architecture. "
    "You may output your reasoning, followed by your final response. "
    "You may format your reasoning block however you like. "
    "However, your final response must strictly follow these rules: "
    "* Write in plain, unformatted prose like a book or newspaper article. "
    "* Do not use markdown, bullet points, lists, or headers. "
    "* Standard numbers, abbreviations, and symbols are acceptable. "
    "* [CRITICAL] You must press enter after every single sentence, placing each sentence on its own separate line."
)
CHAT_MAX_TOKENS = 2048  # official --text-max-tokens default
# The official demo samples this pass unseeded; some trajectories deliberate
# past the token budget without ever closing the think block, so retry a few
# deterministic seeds until one closes.
CHAT_SEEDS = (0, 1, 2)

_audex_deployment = get_deploy_config_path("audex_s2s.yaml")
_audex_model = os.environ.get(MODEL_DIR_ENV) or MODEL
_OMNI_RUNNER_PARAMS = [
    pytest.param(
        (_audex_model, _audex_deployment, {"async_chunk": True}),
        id="s2s_full",
    ),
]
# 30B-A3B variant: opt-in (pulls ~60 GB); enable with AUDEX_E2E_30B=1.
_AUDEX_30B_MODEL = os.environ.get("VLLM_OMNI_AUDEX_30B_MODEL_DIR") or "nvidia/Nemotron-Labs-Audex-30B-A3B"
if os.environ.get("AUDEX_E2E_30B") == "1":
    _OMNI_RUNNER_PARAMS.append(
        pytest.param(
            (_AUDEX_30B_MODEL, get_deploy_config_path("audex_s2s_30b.yaml"), {"async_chunk": True}),
            id="s2s_30b",
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
            cur = min(rows[j] + 1, rows[j - 1] + 1, prev + (ref_word != hyp_word))
            prev, rows[j] = rows[j], cur
    return rows[-1] / len(ref_words)


def _chat_prompt(user_text: str) -> str:
    # Official chat-pass recipe: system prompt + formatting instruction
    # prefixed to the user text, thinking ENABLED via an open <think> block
    # (chat_template.jinja with enable_thinking=True renders
    # "<|im_start|>assistant\n<think>\n"). Priming the block open keeps the
    # reasoning inside <think>...</think> so _extract_final_response can drop
    # it; with a closed block the 30B leaks the reasoning as plain text.
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{TEXT_INSTRUCTION}\n\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def _extract_final_response(text: str) -> str:
    # Mirror the official demo's extract_spoken_answer: keep only what
    # follows the LAST closing think tag, falling back to the full text when
    # the model skipped reasoning or never closed the block. Benign no-op for
    # the 2B, whose answers carry no think block.
    if "</think>" in text:
        answer = text.rsplit("</think>", 1)[-1].strip()
        if answer:
            return answer
    return text.strip()


def _chat_params(runner: OmniRunner, seed: int):
    # Official demo text-pass sampling (--text-temperature 1.0, --text-top-p
    # 0.95, --text-max-tokens 2048). The yaml stage-0 defaults (temperature
    # 0.1, top_k 80) are TTS-thinker settings; under near-greedy decoding the
    # 30B loops inside its reasoning block until the token budget runs out
    # and never emits </think>.
    params = copy.deepcopy(runner.omni.resolve_sampling_params_list(None))
    stage0 = params[0]
    stage0.temperature = 1.0
    stage0.top_p = 0.95
    stage0.top_k = -1
    stage0.max_tokens = CHAT_MAX_TOKENS
    stage0.seed = seed
    return params


def _chat_once(runner: OmniRunner, prompt: str, seed: int) -> tuple[str, bool]:
    (req_output,) = runner.omni.generate([{"prompt": prompt, "modalities": ["text"]}], _chat_params(runner, seed))
    out = req_output.outputs[0]
    text = (out.text or "").strip()
    truncated = getattr(out, "finish_reason", None) == "length" or len(out.token_ids or []) >= CHAT_MAX_TOKENS
    return text, truncated


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_audex_s2s_cascade_round_trip(omni_runner: OmniRunner, run_level: str) -> None:
    """ASR → chat → TTS over one deployment; text passes never emit audio."""
    real_weights = run_level in {"advanced_model", "full_model"}
    # Spoken question: vLLM's public audio asset (downloaded on first use).
    audio, sr = AudioAsset("mary_had_lamb").audio_and_sample_rate

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
    stage1_audio = concat_audio(mm.get("audio")) if "audio" in mm else np.zeros((0,), dtype=np.float32)
    assert stage1_audio.size == 0, "Text-modality ASR pass produced stage-1 audio (routing regression)"

    # Pass 2 — chat (text-final). The recipe is size-conditional (see the
    # SYSTEM_PROMPT comment above): the 30B requires the official recipe,
    # the 2B regresses under it.
    if omni_runner.model_name == _AUDEX_30B_MODEL:
        # Official recipe. The raw completion is
        # <reasoning...></think><answer>; speak only the final response.
        # Retry with the next seed only when the budget ran out before the
        # reasoning block closed (a natural stop without </think> is a
        # complete answer); if every seed spins, keep the full text
        # (official demo behavior).
        chat_prompt = _chat_prompt(transcript or "Hello.")
        raw = ""
        for seed in CHAT_SEEDS:
            raw, truncated = _chat_once(omni_runner, chat_prompt, seed)
            if "</think>" in raw or not truncated:
                break
        answer = _extract_final_response(raw)
    else:
        # 2B: bare prompt with a closed think block, engine-default sampling
        # (yaml near-greedy), no retry; the answer is used as-is.
        chat_prompt = f"<|im_start|>user\n{transcript or 'Hello.'}<|im_end|>\n<|im_start|>assistant\n<think></think>"
        (chat_output,) = omni_runner.omni.generate([{"prompt": chat_prompt, "modalities": ["text"]}])
        answer = (chat_output.outputs[0].text or "").strip()

    if real_weights:
        assert transcript, "ASR pass produced an empty transcript"
        assert answer, "Chat pass produced an empty answer"

    # Pass 3 — TTS (audio-final; streams through code2wav). Both final
    # stages may emit an output record; collect the audio-bearing one.
    # The formatting instruction puts every sentence on its own line;
    # flatten to plain prose for the TTS thinker.
    tts_text = re.sub(r"\s+", " ", answer).strip() if answer else "The weather is nice today."
    tts_text = tts_text[:200]
    tts_prompt = {"prompt": build_cond_prompt(tts_text), "modalities": ["audio"]}
    tts_outputs = omni_runner.omni.generate([tts_prompt])
    assert tts_outputs, "TTS pass returned no outputs"
    speech = np.zeros((0,), dtype=np.float32)
    for req_output in tts_outputs:
        mm = getattr(req_output, "multimodal_output", None) or {}
        if "audio" in mm:
            speech = concat_audio(mm["audio"])
            break
    assert speech.size > 0, "TTS pass produced empty audio"

    if real_weights:
        rms = float(np.sqrt(np.mean(np.square(speech))))
        assert rms > 1e-3, f"TTS-pass audio near-silent (rms={rms})"

        # Pass 4 — output verification: transcribe the synthesized answer
        # with the SAME deployment's ASR pass and compare to the answer
        # text (self-contained; no external ASR dependency).
        verify_outputs = omni_runner.omni.generate(
            [
                {
                    "prompt": ASR_PROMPT,
                    "multi_modal_data": {"audio": (speech, SAMPLE_RATE)},
                    "modalities": ["text"],
                }
            ]
        )
        verify_text = (verify_outputs[0].outputs[0].text or "").strip()
        assert verify_text, "Output-verification ASR pass returned empty text"
        # The full checkpoint prefixes transcripts with a language-tag
        # sentence and quotes the content; extract the quoted payload.
        if "'" in verify_text and len(verify_text.split("'")) >= 3:
            verify_text = verify_text.split("'")[1].strip()
        wer = _wer(verify_text, tts_text)
        # Hard gate: the synthesized WAV must transcribe back to the answer
        # (lenient threshold absorbs ASR quoting/prefix differences).
        assert wer <= 0.5, f"Output WAV does not match the answer (WER={wer:.2f}): {verify_text!r} vs {tts_text!r}"
        print(f"[trend] Audex S2S output-ASR WER vs answer: {wer:.3f} ({verify_text!r})")

    # Final pass — time-to-first-audio for the streaming TTS path. This
    # MUST be the last engine call: the py_generator API closes the engine
    # when the generator is exhausted.
    t_start = time.perf_counter()
    first_audio_s: float | None = None
    for req_output in omni_runner.omni.generate([tts_prompt], py_generator=True):
        mm = getattr(req_output, "multimodal_output", None) or {}
        if "audio" in mm and concat_audio(mm["audio"]).size > 0:
            if first_audio_s is None:
                first_audio_s = time.perf_counter() - t_start
    assert first_audio_s is not None, "streaming TTS never produced an audio-bearing chunk"
    print(f"[trend] Audex S2S TTS-pass first-audio latency: {first_audio_s * 1000.0:.0f} ms")
