# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline single-speaker prompt, matching the native HF chat template."""

from collections.abc import Sequence
from typing import Any

AUDIO_BOS = "<|vision_start|>"
AUDIO_EOS = "<|vision_end|>"
AUDIO_TOKEN = "<|vision_pad|>"
AUDIO_TOKEN_ID = 151654


def render_vibevoice_prompt(text: str, *, reference: bool = True) -> str:
    """Render plain text for speaker 0, with or without a reference voice.

    With ``reference=True``, pass the result as ``prompt`` and the reference
    waveform/sample-rate pair as ``multi_modal_data={"audio": (waveform,
    sample_rate)}`` to offline Omni; the multimodal processor expands the single
    placeholder. With ``reference=False``, the native no-voice prompt is
    rendered and no audio may be passed. As in HF, there is no default voice;
    the voice is sampled per request from the diffusion noise.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("VibeVoice requires non-empty text.")
    if "<|" in text:
        raise ValueError("VibeVoice text must not contain special-token literals.")
    voice = f" Voice input:\n Speaker 0:{AUDIO_BOS}{AUDIO_TOKEN}{AUDIO_EOS}\n" if reference else ""
    return (
        " Transform the text provided by various speakers into speech output, "
        "utilizing the distinct voice of each respective speaker.\n"
        f"{voice}"
        " Text input:\n"
        f" Speaker 0: {text}\n"
        f" Speech output:\n{AUDIO_BOS}"
    )


def check_reference_prompt(prompt: Any, sampling_params_list: Sequence[Any]) -> Any:
    """Reject a reference placeholder without audio (stage-0 prompt transform).

    vLLM skips the multimodal processor when no audio is passed, so its
    placeholder check never runs and the placeholder would be silently embedded
    as text. Audio without a placeholder is rejected by the processor.
    """
    fields = {"prompt": prompt} if isinstance(prompt, str) else prompt
    if isinstance(fields, dict) and not fields.get("multi_modal_data"):
        text = fields.get("prompt")
        token_ids = fields.get("prompt_token_ids") or ()
        if (isinstance(text, str) and AUDIO_TOKEN in text) or AUDIO_TOKEN_ID in token_ids:
            raise ValueError(
                "VibeVoice prompt has a reference-audio placeholder but no audio; pass "
                "multi_modal_data={'audio': ...} or use render_vibevoice_prompt(text, reference=False)."
            )
    return prompt
