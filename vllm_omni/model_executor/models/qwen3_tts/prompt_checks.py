# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 prompt checks for Qwen3-TTS.

The Talker cannot serve a ``Base`` (voice clone) request without a voice
reference: it raises inside the engine core, which takes the engine down, and
an in-context request without a transcript silently drops to x-vector-only mode
and runs away. Reject both in the caller's process with the speech API's
messages, after the text check.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from vllm_omni.config.stage_config import on_streaming_updates
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_REF_IDS_KEY,
    Qwen3TTSPromptEmbedsBuilder,
    check_text_prompt,
    check_text_update,
    coerce_token_ids,
    first_value,
    resolve_x_vector_only,
)

_MISSING_REFERENCE = (
    "Base task requires 'ref_audio' or 'speaker_embedding' for voice cloning: pass "
    "additional_information['ref_audio'] as a list (e.g. ['ref.wav']), or 'voice_clone_prompt' with "
    "'ref_spk_embedding' (and 'ref_code' unless 'x_vector_only_mode' is enabled)"
)
_MISSING_TRANSCRIPT = (
    "Base task requires non-empty 'ref_text' (transcript of the reference audio) unless 'x_vector_only_mode' is enabled"
)


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_token_ids(value: object) -> bool:
    # Whether the Talker can turn this into non-empty ``ref_ids``: empty ids fall back to x-vector mode,
    # and arrays torch.from_numpy rejects (str/object dtype, negative strides) raise in the engine core.
    try:
        return coerce_token_ids(value, device=torch.device("cpu")) is not None
    except (TypeError, ValueError):
        return False


def validate_voice_clone_request(info: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if a Base request lacks what the Talker needs."""
    if first_value(info.get("task_type"), "CustomVoice") != "Base":
        return
    ref_audio = info.get("ref_audio")
    has_ref_audio = isinstance(ref_audio, list) and bool(ref_audio) and ref_audio[0] is not None
    if has_ref_audio and isinstance(ref_audio[0], str):
        has_ref_audio = _is_text(ref_audio[0])
    prompt = Qwen3TTSPromptEmbedsBuilder._voice_clone_prompt_dict(info.get("voice_clone_prompt"))
    speaker = info.get("speaker")
    if not has_ref_audio and prompt is None and isinstance(speaker, list) and speaker and _is_text(speaker[0]):
        # A stored voice: the Talker resolves its profile from a cache this process cannot see.
        return
    prompt = prompt or {}

    xvec = resolve_x_vector_only(dict(info))
    spk = prompt.get("ref_spk_embedding")
    has_spk = isinstance(spk, torch.Tensor) or (isinstance(spk, (list, np.ndarray)) and len(spk) > 0)
    ref_code = prompt.get("ref_code")
    # The Talker only reads a voice_clone_prompt ref_code that is a tensor or an array.
    has_ref_code = Qwen3TTSPromptEmbedsBuilder._has_ref_code_like(ref_code) and isinstance(
        first_value(ref_code), (torch.Tensor, np.ndarray)
    )
    if not (has_ref_audio or (has_spk and (xvec or has_ref_code))):
        raise ValueError(_MISSING_REFERENCE)

    if xvec:
        return
    if any(_is_token_ids(v) for v in (info.get(PRECOMPUTED_REF_IDS_KEY), info.get("ref_ids"))):
        return
    if any(_is_token_ids(prompt.get(k)) for k in ("ref_ids", "ref_id")):
        return
    if _is_text(first_value(info.get("ref_text"))) or _is_text(first_value(prompt.get("ref_text"))):
        return
    raise ValueError(_MISSING_TRANSCRIPT)


def check_voice_clone_prompt(prompt: Any, sampling_params_list: Sequence[Any]) -> Any:
    """Stage-0 ``prompt_transform_func``: validate Base prompts, return them unchanged."""
    for item in prompt if isinstance(prompt, list) else (prompt,):
        info = item.get("additional_information") if isinstance(item, dict) else None
        if isinstance(info, Mapping):
            validate_voice_clone_request(info)
    return prompt


@on_streaming_updates(check_text_update)
def check_tts_prompt(prompt: Any, sampling_params_list: Sequence[Any]) -> Any:
    """Stage-0 ``prompt_transform_func`` of both Qwen3-TTS pipelines: text, then voice clone."""
    check_text_prompt(prompt, sampling_params_list)
    return check_voice_clone_prompt(prompt, sampling_params_list)
