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
    PRECOMPUTED_REF_CODE_KEY,
    PRECOMPUTED_REF_IDS_KEY,
    REF_AUDIO_CACHE_KEY,
    Qwen3TTSPromptEmbedsBuilder,
    check_text_prompt,
    check_text_update,
    coerce_ref_code_tensor,
    coerce_token_ids,
    first_value,
    resolve_x_vector_only,
)

_MISSING_REFERENCE = (
    "Base task requires 'ref_audio' or 'speaker_embedding' for voice cloning: pass "
    "additional_information['ref_audio'] as a list (e.g. ['ref.wav']), or 'voice_clone_prompt' with "
    "'ref_spk_embedding' and either 'x_vector_only_mode' enabled or codes['precomputed_ref']"
)
_MISSING_TRANSCRIPT = (
    "Base task requires non-empty 'ref_text' (transcript of the reference audio) unless 'x_vector_only_mode' is enabled"
)
_NOT_TRANSPORTABLE = (
    "Qwen3-TTS {field}[{key!r}] holds a NumPy array or a tensor in a form that does not reach the engine "
    "intact; pass {fix}"
)
_TRANSPORTABLE_FORMS = {
    "ref_audio": "a path/URL or the waveform as plain lists, e.g. ['ref.wav'] or [[waveform.tolist(), sr]]",
    "voice_clone_prompt": "its values as plain Python lists",
    "codes": "the precomputed_ref code as plain lists, e.g. [code.tolist()]",
}


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_array(value: object) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return True
    if isinstance(value, dict):
        return any(_has_array(v) for v in value.values())
    # A list starting with a plain number is taken as all numbers (e.g. a waveform) rather than walked.
    if isinstance(value, (list, tuple)) and not (value and isinstance(value[0], (int, float))):
        return any(_has_array(v) for v in value)
    return False


def _reaches_engine(value: object) -> bool:
    # The engine-core msgpack round trip keeps only a top-level tensor (or one directly under a
    # nested key such as `codes`); an ndarray, or a tensor inside a list or another dict, arrives
    # as a raw [dtype, shape, buffer] list the Talker cannot read.
    return isinstance(value, torch.Tensor) or not _has_array(value)


def _is_token_ids(value: object) -> bool:
    # Whether the Talker can read a transcript from these ids: it takes ids[:, 3:-2] of one request.
    ids = coerce_token_ids(value, device=torch.device("cpu"))
    return ids is not None and ids.shape[0] == 1 and ids.shape[1] > 5


def _is_base(info: Mapping[str, Any]) -> bool:
    return first_value(info.get("task_type"), "CustomVoice") == "Base"


def _reference_values(info: Mapping[str, Any]) -> dict[str, Any]:
    codes = info.get("codes")
    values = {key: info.get(key) for key in ("ref_audio", "voice_clone_prompt", "ref_ids", PRECOMPUTED_REF_IDS_KEY)}
    return {**values, "codes": codes.get(PRECOMPUTED_REF_CODE_KEY) if isinstance(codes, Mapping) else None}


def validate_voice_clone_request(info: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if a Base request lacks what the Talker needs."""
    if not _is_base(info):
        return
    values = _reference_values(info)
    for key, value in values.items():
        if not _reaches_engine(value):
            fix = _TRANSPORTABLE_FORMS.get(key, "a top-level tensor or plain Python lists")
            raise ValueError(_NOT_TRANSPORTABLE.format(field="additional_information", key=key, fix=fix))
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
    has_spk = isinstance(spk, list) and bool(spk)
    # A voice_clone_prompt arrives as plain lists, but the Talker reads its ref_code only as a tensor
    # or array; codes['precomputed_ref'] works as a tensor or as [code.tolist()].
    has_ref_code = coerce_ref_code_tensor(values["codes"], device=torch.device("cpu")) is not None
    # Reference artifacts cached in the engine under this key; a miss raises there, as on main.
    has_cached_ref = _is_text(first_value(info.get(REF_AUDIO_CACHE_KEY)))
    if not (has_ref_audio or has_cached_ref or (has_spk and (xvec or has_ref_code))):
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


def _overlay(base: Mapping[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    # How the runners merge the two payloads: dict values one level deep, other values replaced.
    view = dict(base)
    for key, value in top.items():
        section = view.get(key)
        view[key] = {**section, **value} if isinstance(value, dict) and isinstance(section, dict) else value
    return view


def check_voice_clone_prompt(prompt: Any, sampling_params_list: Sequence[Any]) -> Any:
    """Stage-0 ``prompt_transform_func``: validate Base prompts, return them unchanged.

    With a ``model_intermediate_buffer``, each Base view must pass: Model Runner V2
    lays the buffer over ``additional_information`` (``OmniIntermediateBuffer.add_request``),
    the V1 runner the reverse, and a plain streaming update replaces the buffer, so
    the reference must not live in the buffer alone. The buffer travels as a plain
    dict, so no tensor or array in it reaches the engine intact.
    """
    for item in prompt if isinstance(prompt, list) else (prompt,):
        info = item.get("additional_information") if isinstance(item, dict) else None
        buffer = item.get("model_intermediate_buffer") if isinstance(item, dict) else None
        info = info if isinstance(info, Mapping) else {}
        views = (
            (info, _overlay(info, buffer), _overlay(buffer, info)) if isinstance(buffer, dict) and buffer else (info,)
        )
        if len(views) > 1 and any(_is_base(view) for view in views):
            for key, value in _reference_values(buffer).items():
                if _has_array(value):
                    fix = _TRANSPORTABLE_FORMS.get(key, "plain Python lists")
                    raise ValueError(_NOT_TRANSPORTABLE.format(field="model_intermediate_buffer", key=key, fix=fix))
        for view in views:
            validate_voice_clone_request(view)
    return prompt


@on_streaming_updates(check_text_update)
def check_tts_prompt(prompt: Any, sampling_params_list: Sequence[Any]) -> Any:
    """Stage-0 ``prompt_transform_func`` of both Qwen3-TTS pipelines: text, then voice clone."""
    check_text_prompt(prompt, sampling_params_list)
    return check_voice_clone_prompt(prompt, sampling_params_list)
