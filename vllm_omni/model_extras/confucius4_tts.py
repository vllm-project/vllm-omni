# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline request and output helpers for Confucius4-TTS."""

from __future__ import annotations

from typing import Any

import torch
from vllm import SamplingParams
from vllm.inputs import tokens_input

from vllm_omni.model_executor.models.confucius4_tts.configuration_confucius4_tts import (
    DEFAULT_MAX_TEXT_TOKENS,
    MAX_GENERATED_SEMANTIC_TOKENS,
    OUTPUT_SAMPLE_RATE,
    SEMANTIC_EOS_TOKEN_ID,
)
from vllm_omni.model_executor.models.confucius4_tts.prompt_utils import (
    tokenize_confucius4_prompt,
)

SAMPLE_RATE = OUTPUT_SAMPLE_RATE
MAX_TEXT_TOKENS = DEFAULT_MAX_TEXT_TOKENS
MAX_SEMANTIC_TOKENS = MAX_GENERATED_SEMANTIC_TOKENS


def confucius4_tts_prompt(
    model_path: str,
    text: str,
    ref_audio: str | tuple | list,
    *,
    lang: str = "zh",
) -> dict[str, Any]:
    """Build a Confucius4-TTS prompt for ``Omni.generate``."""
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")
    if ref_audio is None:
        raise ValueError("ref_audio is required")

    token_ids = tokenize_confucius4_prompt(model_path, text, lang)
    if len(token_ids) > MAX_TEXT_TOKENS:
        raise ValueError(f"Confucius4-TTS formatted prompt exceeds {MAX_TEXT_TOKENS} tokens")

    prompt = tokens_input(prompt_token_ids=[0] * (len(token_ids) + 2))
    prompt["additional_information"] = {
        "text": [text],
        "lang": [lang],
        "text_token_ids": [token_ids],
        "voice": [ref_audio],
    }
    return prompt


def confucius4_tts_sampling_params(
    *,
    max_new_tokens: int = MAX_SEMANTIC_TOKENS,
    seed: int | None = None,
) -> list[SamplingParams]:
    """Build the two-stage sampling parameters for Confucius4-TTS."""
    if not 1 <= max_new_tokens <= MAX_SEMANTIC_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_SEMANTIC_TOKENS}")
    return [
        SamplingParams(
            temperature=0.8,
            top_p=0.8,
            top_k=30,
            repetition_penalty=10.0,
            max_tokens=max_new_tokens,
            stop_token_ids=[SEMANTIC_EOS_TOKEN_ID],
            detokenize=False,
            seed=seed,
        ),
        SamplingParams(temperature=0.0, max_tokens=1, seed=seed),
    ]


def confucius4_tts_audio(output: Any) -> torch.Tensor:
    """Return one generated waveform as a flat CPU float tensor."""
    multimodal = getattr(output, "multimodal_output", None)
    if not multimodal and getattr(output, "outputs", None):
        multimodal = output.outputs[0].multimodal_output
    if not multimodal:
        raise ValueError("Confucius4-TTS returned no multimodal output")

    audio = multimodal.get("audio")
    if audio is None:
        audio = multimodal.get("model_outputs")
    if audio is None:
        raise ValueError(f"Confucius4-TTS returned no audio: {sorted(multimodal)}")
    if isinstance(audio, list):
        audio = torch.cat([torch.as_tensor(chunk).reshape(-1) for chunk in audio])
    return torch.as_tensor(audio).float().cpu().reshape(-1)


__all__ = [
    "MAX_SEMANTIC_TOKENS",
    "MAX_TEXT_TOKENS",
    "SAMPLE_RATE",
    "confucius4_tts_audio",
    "confucius4_tts_prompt",
    "confucius4_tts_sampling_params",
]
