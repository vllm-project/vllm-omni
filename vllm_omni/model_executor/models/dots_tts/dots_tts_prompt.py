# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dots.tts prompt builder for vllm-omni's Omni engine + serving layer.

Mirrors upstream ``build_generation_schedule`` (rednote-hilab/dots.tts @
a393d2e data/pipelines/tokenizing.py:169) for the
no-prompt-audio + audio-placeholder path — text-only zero-shot TTS.

V1 returns real Qwen2 token IDs (not placeholders ``[1] * N`` like
voxcpm2's builder).  Reason: our talker's preprocess() runs
``self.model.embed_tokens(input_ids)`` directly, so real IDs flow
through naturally.  voxcpm2 uses placeholders because its prefill mixes
reference-audio embeddings (no token IDs); dots.tts V1 zero-shot is
purely text + audio_gen_start, all real IDs.

Voice clone (reference audio + speaker x-vector) is intentionally not
wired here — V1 is zero-shot only.  Add reference_audio /
voice_profile / prompt_text branches in a follow-up (mirror voxcpm2
``build_voxcpm2_prompt``).
"""

from __future__ import annotations

from typing import Any

from vllm.inputs import tokens_input

# Upstream rednote-hilab/dots.tts @ a393d2e
# data/pipelines/tts_pipeline.py:17-18
_TTS_TEXT_PREFIX = "[文本]"
_TTS_AUDIO_PREFIX = "[文本对应语音]"
_AUDIO_GEN_START_TOKEN = "<|audio_gen_start|>"


def build_dots_tts_prompt(
    tokenizer: Any,
    text: str,
) -> dict[str, Any]:
    """Build a dots.tts prefill prompt dict for ``Omni.generate()``.

    Returns ``{"prompt_token_ids": [...], "additional_information": {...}}``
    where ``prompt_token_ids`` is the real Qwen2 token sequence:

        [text_tokens of "[文本]<text>[文本对应语音]", audio_gen_start_id]

    talker's preprocess() embeds these into ``hidden_states`` for the AR
    base LM.  Decode then drives via ``state.curr_embed_for_next`` (DiT
    → patch_encoder AR loopback).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"text must be a non-empty string, got {text!r}")

    prompt_text = f"{_TTS_TEXT_PREFIX}{text}{_TTS_AUDIO_PREFIX}"
    text_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    audio_gen_start_id = tokenizer.convert_tokens_to_ids(_AUDIO_GEN_START_TOKEN)
    if not isinstance(audio_gen_start_id, int) or audio_gen_start_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Tokenizer does not know {_AUDIO_GEN_START_TOKEN!r} "
            f"(got id={audio_gen_start_id!r}).  Did you load the "
            f"dots.tts tokenizer (with added_tokens.json)?"
        )
    prompt_token_ids = list(text_ids) + [audio_gen_start_id]

    prompt = tokens_input(prompt_token_ids=prompt_token_ids)
    prompt["additional_information"] = {
        # Reserved for future voice clone (ref_audio / speaker x-vector).
        # V1 zero-shot leaves this empty; preprocess() ignores it.
    }
    return prompt
