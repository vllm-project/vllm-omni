# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plain-text TTS prompt builder + scope validators for higgs-audio v2.

The vllm-omni v1 scope is plain text -> 24 kHz speech only. The upstream
HiggsAudio chat template supports voice cloning, multi-speaker, ChatML rich
messages, and music / sound-event prompts; those are intentionally rejected
here with model-specific errors so the request validator in
``serving_speech.py`` can return a deterministic 4xx instead of silently
falling into an unsupported code path.

The prompt token ID sequence emitted by :func:`build_plain_text_prompt`
matches the upstream ``HiggsAudioV2Processor.apply_chat_template`` output for
the equivalent conversation, so that AC-1 (input-token parity) holds.
"""

from __future__ import annotations

import re
from typing import Any

import torch

__all__ = [
    "UnsupportedInputError",
    "MULTI_SPEAKER_TAG_PATTERN",
    "REJECTED_REQUEST_FIELDS",
    "validate_plain_text_request",
    "validate_plain_text_input",
    "build_plain_text_conversation",
    "build_plain_text_prompt",
]


class UnsupportedInputError(ValueError):
    """Raised when a request asks for an out-of-scope higgs_audio_v2 feature."""


# Matches the upstream multi-speaker SPEAKERn tag, e.g. [SPEAKER0], [SPEAKER12].
MULTI_SPEAKER_TAG_PATTERN = re.compile(r"\[SPEAKER\d+\]", re.IGNORECASE)

# Request fields that the v1 validator must reject with a 4xx. These cover
# voice cloning (ref_audio / ref_text / voice_prompt / reference_audio /
# speaker_audio), rich chat input (messages), and multi-speaker tags.
REJECTED_REQUEST_FIELDS: tuple[str, ...] = (
    "reference_audio",
    "ref_audio",
    "voice_prompt",
    "speaker_audio",
    "ref_text",
    "speakers",
    "messages",
)


def validate_plain_text_input(text: str) -> None:
    """Reject multi-speaker tags inside the user text body.

    Phase-1 explicitly does NOT support multi-speaker dialogue. Catching the
    pattern here means the rejection happens at the tokenizer boundary and is
    visible to both offline (`pipeline.py`) and online (`serving_speech.py`)
    code paths.
    """
    if not isinstance(text, str):
        raise UnsupportedInputError(
            f"higgs_audio_v2 expects plain text input; got {type(text).__name__}"
        )
    if MULTI_SPEAKER_TAG_PATTERN.search(text):
        raise UnsupportedInputError(
            "higgs_audio_v2 v1 does not support multi-speaker [SPEAKERn] tags; "
            "received text contains a speaker tag"
        )


def validate_plain_text_request(request_payload: dict[str, Any]) -> None:
    """Walk through a request dict and reject any out-of-scope field.

    The serving layer calls this BEFORE building the prompt so the 4xx error
    message can name the model and the offending field. Anything still
    present in :data:`REJECTED_REQUEST_FIELDS` after the validator is treated
    as a hard reject regardless of value.
    """
    for field in REJECTED_REQUEST_FIELDS:
        if field in request_payload and request_payload[field] not in (None, "", [], {}):
            raise UnsupportedInputError(
                f"higgs_audio_v2 v1 does not support the request field "
                f"{field!r}; supply plain text via the 'input' field instead"
            )

    text = request_payload.get("input")
    if text is None:
        raise UnsupportedInputError(
            "higgs_audio_v2 requires plain text in the 'input' field"
        )
    validate_plain_text_input(text)


def build_plain_text_conversation(text: str) -> list[dict[str, Any]]:
    """Build the canonical single-speaker plain-text conversation.

    Mirrors upstream's ``zero_shot`` input sample under
    ``examples/serve_engine/input_samples.py``: the system prompt contains
    a ``<|scene_desc_start|>SPEAKER0: ...<|scene_desc_end|>`` block, which
    is what conditions the model to produce coherent natural speech. The
    earlier no-scene-block variant produced repetitive babbling once the
    vLLM attention-backend NaN was unblocked (R14 / FLEX_ATTENTION).
    """
    validate_plain_text_input(text)
    system_prompt = (
        "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: british accent\n"
        "<|scene_desc_end|>"
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    ]


def build_plain_text_prompt(
    processor: Any,
    text: str,
    *,
    sampling_rate: int = 24000,
    return_tensors: str | None = "pt",
) -> dict[str, Any]:
    """Run the upstream processor's chat template on a plain-text input.

    Returns the processor output dict (``input_ids`` plus any auxiliary tensors
    such as ``attention_mask``). The serving layer passes ``input_ids`` to
    Stage 0 as ``prompt_token_ids`` after a ``.tolist()``.

    Using the upstream processor verbatim (no system-prompt rewriting) is what
    gives AC-1 input-token parity its meaning. DEC-2 (use upstream verbatim vs.
    a normalized template) is currently set to verbatim, in line with the
    Claude-recommended default in the plan.
    """
    conversation = build_plain_text_conversation(text)
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        sampling_rate=sampling_rate,
        return_tensors=return_tensors,
    )
    if "input_ids" not in inputs:
        raise RuntimeError(
            "HiggsAudioV2 processor returned no input_ids; got keys "
            f"{list(inputs.keys())!r}"
        )
    return inputs


def input_ids_to_python_list(inputs: dict[str, Any]) -> list[int]:
    """Convenience: pull a flat ``list[int]`` of token IDs from a processor output."""
    ids = inputs["input_ids"]
    if isinstance(ids, torch.Tensor):
        if ids.ndim == 2 and int(ids.shape[0]) != 1:
            raise ValueError(
                f"expected batch=1 prompt; got input_ids shape {tuple(ids.shape)}"
            )
        return ids.reshape(-1).tolist()
    return list(ids)
