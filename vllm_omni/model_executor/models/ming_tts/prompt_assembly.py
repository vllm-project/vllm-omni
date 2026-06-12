# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

import torch

from .audio_prep import (
    coerce_speaker_embeddings,
    count_prompt_latent_patches,
    count_prompt_waveform_patches,
    pad_prompt_waveform,
)
from .config_ming_tts import (
    AUDIO_FRAME_HOP,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_PROMPT_LATENTS,
    KEY_REQUEST_ID,
    KEY_SPEAKER_EMBEDDING,
    LATENT_DIM,
    PATCH_SIZE,
    SAMPLE_RATE,
    VAE_PATCH_SIZE,
)

DEFAULT_PROMPT = "Please generate speech based on the following description.\n"
BASE_CAPTION_TEMPLATE: dict[str, Any] = {
    "audio_sequence": [
        {
            "序号": 1,
            "说话人": "speaker_1",
            "方言": None,
            "风格": None,
            "语速": None,
            "基频": None,
            "音量": None,
            "情感": None,
            "BGM": {
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None,
                "ENV": None,
                "SNR": None,
            },
            "IP": None,
        }
    ]
}
_DURATION_SECONDS_RE = re.compile(r"Duration:\s*([0-9]+(?:\.[0-9]+)?)\s*s\b", re.IGNORECASE)


def create_instruction(user_input: Any) -> str | None:
    if user_input is None:
        return None
    if isinstance(user_input, str):
        return user_input
    if not isinstance(user_input, dict):
        raise ValueError(f"Ming instruction must be str, dict, or None; got {type(user_input).__name__}")
    caption = copy.deepcopy(BASE_CAPTION_TEMPLATE)
    item = caption["audio_sequence"][0]
    for key, value in user_input.items():
        if key in item:
            item[key] = value
    return json.dumps(caption, ensure_ascii=False)


def parse_duration_seconds(text: str | None) -> float | None:
    if not isinstance(text, str):
        return None
    match = _DURATION_SECONDS_RE.search(text)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if value <= 0.0:
        return None
    return value


def estimate_decode_steps_for_duration(
    duration_seconds: float,
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_hop: int = AUDIO_FRAME_HOP,
    patch_size: int = PATCH_SIZE,
    vae_patch_size: int = VAE_PATCH_SIZE,
) -> int:
    if duration_seconds <= 0.0:
        return 0
    samples_per_decode_step = int(frame_hop) * int(patch_size) * int(vae_patch_size)
    required_samples = float(duration_seconds) * float(sample_rate)
    return max(1, int(math.ceil(required_samples / float(samples_per_decode_step))))


def estimate_decode_step_window_for_duration(duration_seconds: float) -> tuple[int, int]:
    target_steps = estimate_decode_steps_for_duration(duration_seconds)
    min_steps = max(1, target_steps - 3)
    max_steps = max(min_steps, target_steps + 3)
    return min_steps, max_steps


def resolve_effective_runtime_controls(
    *,
    text: str,
    runtime_controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    controls = {} if runtime_controls is None else dict(runtime_controls)
    has_explicit_min = KEY_MIN_DECODE_STEPS in controls and controls[KEY_MIN_DECODE_STEPS] is not None
    has_explicit_max = KEY_MAX_DECODE_STEPS in controls and controls[KEY_MAX_DECODE_STEPS] is not None
    if has_explicit_min or has_explicit_max:
        return controls
    duration_seconds = parse_duration_seconds(text)
    if duration_seconds is None:
        return controls
    min_decode_steps, max_decode_steps = estimate_decode_step_window_for_duration(duration_seconds)
    controls[KEY_MIN_DECODE_STEPS] = min_decode_steps
    controls[KEY_MAX_DECODE_STEPS] = max_decode_steps
    return controls


def build_dense_prompt_token_ids(
    tokenizer: Any,
    *,
    prompt: str,
    text: str,
    instruction: str | None = None,
    prompt_text: str | None = None,
    speaker_count: int = 0,
    prompt_patch_count: int = 0,
) -> list[int]:
    speaker_prompt = []
    for idx in range(int(speaker_count)):
        speaker_prompt.extend(
            tokenizer.encode(f"  speaker_{idx + 1}:")
            + tokenizer.encode("<|vision_start|>")
            + tokenizer.encode("<|vision_pad|>")
            + tokenizer.encode("<|vision_end|>\n")
        )
    instruction_prompt = (
        tokenizer.encode(instruction) + tokenizer.encode("<|endoftext|>") if instruction is not None else []
    )
    prompt_text_tokens = (
        tokenizer.encode(prompt_text) if int(prompt_patch_count) > 0 and prompt_text is not None else []
    )
    audio_patch_token_id = tokenizer.convert_tokens_to_ids("<audioPatch>")
    if audio_patch_token_id == tokenizer.unk_token_id:
        raise ValueError("Ming tokenizer is missing required <audioPatch> token.")
    prompt_latent_tokens = [audio_patch_token_id] * int(prompt_patch_count)
    text_input_prefix = (
        []
        if all(token in text for token in ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "))
        else tokenizer.encode(" Text input:\n")
    )
    return (
        tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        + tokenizer.encode("<|im_start|>user\n")
        + tokenizer.encode(prompt)
        + speaker_prompt
        + text_input_prefix
        + prompt_text_tokens
        + tokenizer.encode(text)
        + tokenizer.encode("<|im_end|>\n")
        + tokenizer.encode("<|im_start|>assistant\n")
        + instruction_prompt
        + tokenizer.encode("<audio>")
        + prompt_latent_tokens
    )


def build_ming_dense_prompt(
    tokenizer: Any,
    *,
    prompt: str,
    text: str,
    runtime_controls: dict[str, Any] | None = None,
    instruction: Any = None,
    prompt_text: str | None = None,
    prompt_waveform: Any = None,
    prompt_latents: Any = None,
    speaker_embedding: Any = None,
    use_zero_spk_emb: bool = False,
    request_id: str | None = None,
) -> dict[str, Any]:
    instruction_text = create_instruction(instruction)
    speaker_embeddings = coerce_speaker_embeddings(speaker_embedding, use_zero_spk_emb=use_zero_spk_emb)
    effective_runtime_controls = resolve_effective_runtime_controls(text=text, runtime_controls=runtime_controls)

    prompt_waveform_tensor = None
    prompt_patch_count = 0
    if prompt_waveform is not None:
        prompt_waveform_tensor = pad_prompt_waveform(prompt_waveform)
        prompt_patch_count = count_prompt_waveform_patches(prompt_waveform_tensor)
    if prompt_waveform_tensor is not None and prompt_latents is not None:
        raise ValueError(
            "Ming waveform cloning request provided both raw prompt_waveform and explicit prompt_latents. "
            "Choose exactly one source of truth."
        )

    prompt_latent_value = None
    if prompt_waveform_tensor is not None and prompt_text is None:
        raise ValueError(
            "Ming prompt_waveform requires prompt_text for prompt-latent conditioning. "
            "Use speaker_embedding for reference-audio-only speaker conditioning."
        )
    if prompt_latents is not None:
        prompt_latent_value = torch.as_tensor(prompt_latents)
        prompt_patch_count = count_prompt_latent_patches(
            prompt_latent_value, patch_size=PATCH_SIZE, latent_dim=LATENT_DIM
        )

    prompt_token_ids = build_dense_prompt_token_ids(
        tokenizer,
        prompt=prompt,
        text=text,
        instruction=instruction_text,
        prompt_text=prompt_text if prompt_patch_count > 0 else None,
        speaker_count=0 if speaker_embeddings is None else len(speaker_embeddings),
        prompt_patch_count=prompt_patch_count,
    )

    additional_information = {}
    for key, value in effective_runtime_controls.items():
        if isinstance(value, torch.Tensor):
            additional_information[key] = value
        elif key in (KEY_MIN_DECODE_STEPS, KEY_MAX_DECODE_STEPS):
            additional_information[key] = torch.tensor(int(value), dtype=torch.int32)
        else:
            additional_information[key] = torch.tensor(float(value), dtype=torch.float32)
    if request_id is not None:
        additional_information[KEY_REQUEST_ID] = request_id
    if instruction_text is not None:
        additional_information["instruction"] = instruction_text
    if prompt_text is not None:
        additional_information["prompt_text"] = prompt_text
    if prompt_waveform_tensor is not None:
        additional_information["prompt_waveform"] = prompt_waveform_tensor
        additional_information["prompt_waveform_length"] = torch.tensor(
            [int(prompt_waveform_tensor.shape[-1])], dtype=torch.int32
        )
    if prompt_latent_value is not None:
        additional_information[KEY_PROMPT_LATENTS] = prompt_latent_value
    if speaker_embeddings is not None:
        additional_information[KEY_SPEAKER_EMBEDDING] = (
            speaker_embeddings[0] if len(speaker_embeddings) == 1 else torch.stack(speaker_embeddings, dim=0)
        )
    if use_zero_spk_emb:
        additional_information["use_zero_spk_emb"] = True
    return {
        "prompt": prompt,
        "text": text,
        "prompt_token_ids": prompt_token_ids,
        "additional_information": additional_information,
    }


__all__ = [
    "DEFAULT_PROMPT",
    "build_dense_prompt_token_ids",
    "build_ming_dense_prompt",
    "create_instruction",
    "estimate_decode_step_window_for_duration",
    "estimate_decode_steps_for_duration",
    "parse_duration_seconds",
    "resolve_effective_runtime_controls",
]
