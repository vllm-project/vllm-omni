# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact legacy-antmmf prompt contract used to train Ming-3.0-TTS."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import regex as re
import torch

from vllm_omni.model_executor.models.ming_tts.audio_prep import (
    _normalize_prompt_waveform,
    count_prompt_waveform_patches,
    pad_prompt_waveform,
)
from vllm_omni.model_executor.models.ming_tts.prompt_assembly import (
    BASE_CAPTION_TEMPLATE,
    build_moe_prompt_token_ids,
)

from .constants import (
    AUDIO_END_TOKEN_ID,
    AUDIO_PATCH_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    FLOW_PATCH_SIZE,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_PROMPT_AUDIO_START,
    KEY_SPEAKER_EMBEDDING,
    KEY_SPEAKER_POSITIONS,
    KEY_SPEAKER_SAMPLE_RATES,
    KEY_SPEAKER_WAVEFORM,
    KEY_SPEAKER_WAVEFORM_LENGTHS,
    MAX_DECODE_STEPS,
    SPEAKER_END_TOKEN_ID,
    SPEAKER_START_TOKEN_ID,
    TEXT_EOS_TOKEN_ID,
)

USER_PREFIX = "<role>HUMAN</role>Please generate speech based on the following description.\n  speaker_1:"
TEXT_PREFIX = "\n Text input:\n"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"
EXPECTED_TOKENS = {
    "<audioPatch>": AUDIO_PATCH_TOKEN_ID,
    "<audio>": AUDIO_START_TOKEN_ID,
    "</audio>": AUDIO_END_TOKEN_ID,
    "<spk>": SPEAKER_START_TOKEN_ID,
    "</spk>": SPEAKER_END_TOKEN_ID,
}


TARGET_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hk": "Canton",
    "cantonese": "Canton",
}
TARGET_LANGUAGE_NAMES.update({name.lower(): name for name in TARGET_LANGUAGE_NAMES.values()})


def normalize_target_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = TARGET_LANGUAGE_NAMES.get(language.strip().lower().replace("_", "-"))
    if normalized is None:
        raise ValueError(f"Unsupported Ming3 target language: {language!r}")
    return normalized


def create_ming3_instruction(instruction: str | dict[str, str] | None) -> str | None:
    """Preserve full captions; serialize short controls in the release task order."""
    if not isinstance(instruction, dict):
        return instruction
    template = BASE_CAPTION_TEMPLATE["audio_sequence"][0]
    # Emotion retains its archived order; other Tiny tasks put emotion before style.
    fields = (
        tuple(template)
        if set(instruction) == {"情感"}
        else ("序号", "说话人", "方言", "情感", "风格", "语速", "基频", "音量", "BGM", "IP")
    )
    item = {key: instruction.get(key, template[key]) for key in fields}
    return json.dumps({"audio_sequence": [item]}, ensure_ascii=False)


def resolve_podcast_decode_window(text: str, *, max_decode_steps: int | None = None) -> tuple[int, int]:
    minimum_frames = maximum_frames = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*speaker_[1-9][0-9]*:\s*(\S.*)", line)
        if match is None:
            raise ValueError("Podcast requires one 'speaker_N: text' turn per line")
        turn = match[1].strip()
        units = len(turn.split()) if turn.isascii() else len(turn)
        minimum_frames += units
        maximum_frames += units * (8 if turn.isascii() else 4)
    if not minimum_frames:
        raise ValueError("Podcast text must contain a non-empty turn")
    # Match antoc's timely_rain bounds in 12.5 Hz latent frames, then floor both limits to Flow steps.
    maximum_frames = max(8, min(600, maximum_frames))
    minimum_frames = min(maximum_frames, max(8, minimum_frames))
    minimum = minimum_frames // FLOW_PATCH_SIZE
    maximum = maximum_frames // FLOW_PATCH_SIZE
    if max_decode_steps is not None:
        maximum = min(maximum, max_decode_steps)
    if maximum < minimum:
        raise ValueError(f"Podcast maximum decode steps {maximum} is smaller than text-derived minimum {minimum}")
    return minimum, maximum


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


@lru_cache(maxsize=8)
def validate_ming3_tokenizer(tokenizer: Any) -> dict[str, int]:
    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in EXPECTED_TOKENS}
    mismatches = {
        token: (expected, actual[token]) for token, expected in EXPECTED_TOKENS.items() if actual[token] != expected
    }
    if mismatches:
        details = ", ".join(f"{token}: expected {want}, got {got}" for token, (want, got) in mismatches.items())
        raise ValueError("Tokenizer IDs do not match Ming-3.0-TTS training: " + details)
    for token, token_id in EXPECTED_TOKENS.items():
        encoded = _encode(tokenizer, token)
        if encoded != [token_id]:
            raise ValueError(f"{token!r} must encode to [{token_id}], got {encoded}")
    eos_id = int(tokenizer.eos_token_id)
    if eos_id != TEXT_EOS_TOKEN_ID and not (eos_id == 156895 and _encode(tokenizer, "<|role_end|>") == [eos_id]):
        raise ValueError(f"Unsupported Ming3 tokenizer EOS: {eos_id}")
    return actual


def build_ming3_prompt_token_ids(
    tokenizer: Any,
    *,
    target_text: str,
    prompt_text: str | None = None,
    prompt_latent_steps: int = 0,
    instruction: str | None = None,
    speaker_count: int = 1,
    target_language: str | None = None,
) -> tuple[list[int], int | None, int | None]:
    if not target_text or not target_text.strip():
        raise ValueError("target_text must not be empty")
    if prompt_text is not None and not prompt_text.strip():
        raise ValueError("prompt_text must be non-empty when provided")
    if prompt_latent_steps and prompt_text is None:
        raise ValueError("prompt_text is required when prompt latents are present")
    validate_ming3_tokenizer(tokenizer)
    target_language = normalize_target_language(target_language)
    if speaker_count < 0:
        raise ValueError("speaker_count must be non-negative")
    prompt = USER_PREFIX.removeprefix("<role>HUMAN</role>").removesuffix("  speaker_1:")
    if target_language is not None:
        prompt += f"Target language: {target_language}.\n"
    ids = build_moe_prompt_token_ids(
        tokenizer,
        prompt=prompt,
        text=target_text,
        instruction=instruction,
        prompt_text=prompt_text,
        speaker_count=speaker_count,
        prompt_patch_count=prompt_latent_steps,
    )
    if instruction:
        # Ming3 terminates assistant-side controls with the checkpoint's EOS.
        instruction_end = len(ids) - prompt_latent_steps - 2
        ids[instruction_end] = int(tokenizer.eos_token_id)
    speaker_position = ids.index(SPEAKER_START_TOKEN_ID) + 1 if speaker_count else None
    missing_text_prefix = all(
        marker in target_text for marker in ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")
    )
    if missing_text_prefix or (prompt_text is not None and not prompt_latent_steps):
        if speaker_count:
            last_speaker_end = 0
            for _ in range(speaker_count):
                last_speaker_end = ids.index(SPEAKER_END_TOKEN_ID, last_speaker_end) + 1
            text_position = last_speaker_end + len(_encode(tokenizer, "\n"))
        else:
            text_position = len(_encode(tokenizer, "<role>HUMAN</role>")) + len(_encode(tokenizer, prompt))
        text_prefix = _encode(tokenizer, " Text input:\n")
        if missing_text_prefix:
            ids[text_position:text_position] = text_prefix
        if prompt_text is not None and not prompt_latent_steps:
            text_position += len(text_prefix)
            ids[text_position:text_position] = _encode(tokenizer, prompt_text)
    prompt_position = len(ids) - prompt_latent_steps if prompt_latent_steps else None
    return ids, speaker_position, prompt_position


def build_ming3_prompt(
    tokenizer: Any,
    *,
    target_text: str,
    reference_waveform: torch.Tensor | list[torch.Tensor] | None,
    reference_sample_rate: int | list[int] | None,
    prompt_text: str | None = None,
    speaker_embedding: Any = None,
    max_decode_steps: int | None = None,
    instruction: str | None = None,
    target_language: str | None = None,
    speaker_slot: bool = True,
) -> dict[str, Any]:
    if not speaker_slot and (reference_waveform is not None or reference_sample_rate is not None):
        raise ValueError("No-speaker-slot conditioning cannot include reference audio")
    mute_speaker = speaker_slot and reference_waveform is None
    if not speaker_slot and (prompt_text is not None or speaker_embedding is not None):
        raise ValueError("No-speaker-slot conditioning cannot include reference text or speaker embeddings")
    if mute_speaker and (prompt_text is not None or speaker_embedding is not None):
        raise ValueError("Masked speaker conditioning cannot include reference text or speaker embeddings")
    if mute_speaker and not instruction:
        raise ValueError("Masked speaker conditioning requires an instruction")
    waveforms = (
        []
        if mute_speaker or not speaker_slot
        else (reference_waveform if isinstance(reference_waveform, list) else [reference_waveform])
    )
    sample_rates = (
        []
        if mute_speaker or not speaker_slot
        else (reference_sample_rate if isinstance(reference_sample_rate, list) else [reference_sample_rate])
    )
    if (speaker_slot and not mute_speaker and not waveforms) or len(waveforms) != len(sample_rates):
        raise ValueError("Each reference waveform requires a sample rate")
    if any(waveform.shape[-1] == 0 for waveform in waveforms) or any(
        rate is None or rate <= 0 for rate in sample_rates
    ):
        raise ValueError("Reference waveforms and sample rates must be non-empty and positive")
    if len(waveforms) > 1 and not prompt_text:
        raise ValueError("Multi-speaker Ming3 prompts require a reference transcript")
    minimum_steps = None
    if len(waveforms) > 1:
        minimum_steps, max_decode_steps = resolve_podcast_decode_window(target_text, max_decode_steps=max_decode_steps)
    prompt_waveform = None
    prompt_steps = 0
    if prompt_text is not None:
        normalized_waveforms = [
            _normalize_prompt_waveform((waveform, rate), target_sr=44100)
            for waveform, rate in zip(waveforms, sample_rates, strict=True)
        ]
        prompt_waveform = pad_prompt_waveform(
            normalized_waveforms[0] if len(waveforms) == 1 else torch.cat(normalized_waveforms, dim=-1),
            patch_size=1,
            sample_rate=44100,
        )
        prompt_steps = count_prompt_waveform_patches(
            prompt_waveform,
            patch_size=1,
            frame_hop=882,
            vae_patch_size=4,
        )
    ids, _, prompt_position = build_ming3_prompt_token_ids(
        tokenizer,
        target_text=target_text,
        prompt_text=prompt_text,
        prompt_latent_steps=prompt_steps,
        instruction=instruction,
        speaker_count=max(1, len(waveforms)) if speaker_slot else 0,
        target_language=target_language,
    )
    info: dict[str, Any] = {KEY_MAX_DECODE_STEPS: int(max_decode_steps or MAX_DECODE_STEPS)}
    # Global positions survive scheduler slices, including a split between a
    # speaker marker and its embedding slot. Reference audio is the final span.
    info[KEY_SPEAKER_POSITIONS] = [index + 1 for index, token in enumerate(ids) if token == SPEAKER_START_TOKEN_ID]
    if prompt_position is not None:
        info[KEY_PROMPT_AUDIO_START] = prompt_position
    if minimum_steps is not None:
        info[KEY_MIN_DECODE_STEPS] = minimum_steps
    if not speaker_slot:
        info["no_speaker_slot"] = torch.tensor([True])
    elif mute_speaker:
        info["mute_speaker"] = torch.tensor([True])
    else:
        info.update(
            {
                KEY_SPEAKER_WAVEFORM: waveforms[0] if len(waveforms) == 1 else torch.cat(waveforms, dim=-1),
                KEY_SPEAKER_WAVEFORM_LENGTHS: torch.tensor(
                    [waveform.shape[-1] for waveform in waveforms], dtype=torch.int32
                ),
                KEY_SPEAKER_SAMPLE_RATES: torch.tensor(sample_rates, dtype=torch.int32),
            }
        )
    if speaker_embedding is not None:
        info[KEY_SPEAKER_EMBEDDING] = speaker_embedding
    if prompt_waveform is not None:
        info["prompt_waveform"] = prompt_waveform
    return {
        "prompt_token_ids": ids,
        "prompt": USER_PREFIX,
        "text": target_text,
        "additional_information": info,
    }


__all__ = [
    "ASSISTANT_PREFIX",
    "TEXT_PREFIX",
    "USER_PREFIX",
    "build_ming3_prompt",
    "build_ming3_prompt_token_ids",
    "validate_ming3_tokenizer",
]
