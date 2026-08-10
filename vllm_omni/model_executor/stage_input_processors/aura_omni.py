# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage processors for the AURA Omni pipeline."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)

logger = init_logger(__name__)

DEFAULT_AURA_SYSTEM_PROMPT = (
    "You are receiving a live video stream where the final frame is the present moment. "
    "Respond only when a response is needed based on the user's message or the visual context. "
    "Otherwise, output '<|silent|>' to signify silence. Respond in Chinese."
)

SILENT_TEXT = "<|silent|>"
QWEN_IM_START_ID = 151644
QWEN_IM_END_ID = 151645
QWEN_ASSISTANT_ID = 77091
QWEN_NEWLINE_ID = 198
QWEN_ASSISTANT_PREFIX_IDS = [QWEN_IM_START_ID, QWEN_ASSISTANT_ID, QWEN_NEWLINE_ID]
QWEN_ASSISTANT_SUFFIX_IDS = [
    QWEN_IM_END_ID,
    QWEN_NEWLINE_ID,
    QWEN_IM_START_ID,
    QWEN_ASSISTANT_ID,
    QWEN_NEWLINE_ID,
]
AURA_SILENT_TOKEN_IDS = [151669]
QWEN_TEXT_SILENT_TOKEN_IDS = [27, 91, 68658, 91, 29]
QWEN_TEXT_SILENT_PREFIX_TOKEN_IDS = [
    [27],
    [27, 91],
    [27, 91, 34804],
    [27, 91, 34804, 91],
]
DEFAULT_QWEN3_TTS_REF_AUDIO = "vllm-omni/tests/assets/qwen3_tts/clone_2.wav"
DEFAULT_QWEN3_TTS_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
)


def default_qwen3_tts_ref_audio_path() -> str:
    """Return absolute path to the bundled ``clone_2.wav`` reference asset."""
    bundled = Path(__file__).resolve().parents[3] / "tests" / "assets" / "qwen3_tts" / "clone_2.wav"
    if bundled.is_file():
        return str(bundled)
    return DEFAULT_QWEN3_TTS_REF_AUDIO


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _as_prompt_dict(prompt_item: Any) -> dict[str, Any]:
    return prompt_item if isinstance(prompt_item, dict) else {}


def _first_value(value: Any, default: Any = None) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _first_bool(value: Any, default: bool = False) -> bool:
    value = value[0] if isinstance(value, list) and value else value or default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _first_str(value: Any, default: str = "") -> str:
    value = _first_value(value, default)
    return value if isinstance(value, str) else default


def _normalize_qwen3_tts_speaker(speaker: Any) -> Any:
    if not isinstance(speaker, str):
        return speaker
    speaker = speaker.strip()
    if not speaker:
        return speaker
    if "_" in speaker:
        return speaker
    return speaker[0].upper() + speaker[1:].lower()


def _extract_output(source_output: Any) -> Any:
    outputs = getattr(source_output, "outputs", None)
    if isinstance(outputs, list) and outputs:
        return outputs[0]
    return source_output


def _is_finished(source_output: Any) -> bool:
    return bool(getattr(source_output, "finished", False))


def _extract_text(source_output: Any) -> str:
    output = _extract_output(source_output)
    cumulative_text = getattr(output, "cumulative_text", None)
    if isinstance(cumulative_text, str) and cumulative_text:
        return cumulative_text
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text
    mm = getattr(output, "multimodal_output", None)
    if isinstance(mm, dict):
        for key in ("text", "transcript", "asr_text"):
            value = mm.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value and isinstance(value[0], str):
                return value[0]
    return ""


def _extract_token_ids(source_output: Any) -> list[int]:
    output = _extract_output(source_output)
    token_ids = getattr(output, "cumulative_token_ids", None)
    if isinstance(token_ids, list):
        return [int(token_id) for token_id in token_ids if isinstance(token_id, int)]
    return []


def _trim_aura_response_token_ids(token_ids: list[int]) -> list[int]:
    ids = list(token_ids)
    if ids[: len(QWEN_ASSISTANT_PREFIX_IDS)] == QWEN_ASSISTANT_PREFIX_IDS:
        ids = ids[len(QWEN_ASSISTANT_PREFIX_IDS) :]
    if QWEN_IM_END_ID in ids:
        ids = ids[: ids.index(QWEN_IM_END_ID)]
    while ids and ids[-1] in {QWEN_IM_START_ID, QWEN_IM_END_ID, QWEN_NEWLINE_ID}:
        ids.pop()
    return ids


def _qwen3_tts_assistant_token_ids_from_aura(source_output: Any) -> list[int]:
    content_ids = _trim_aura_response_token_ids(_extract_token_ids(source_output))
    if not content_ids:
        return []
    return QWEN_ASSISTANT_PREFIX_IDS + content_ids + QWEN_ASSISTANT_SUFFIX_IDS


def _ensure_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    return [int(token_id) for token_id in list(value) if isinstance(token_id, int)]


def _is_silent_token_prefix(content_ids: list[int]) -> bool:
    if not content_ids:
        return False
    candidates = [
        AURA_SILENT_TOKEN_IDS,
        QWEN_TEXT_SILENT_TOKEN_IDS,
        *QWEN_TEXT_SILENT_PREFIX_TOKEN_IDS,
    ]
    return any(candidate[: len(content_ids)] == content_ids for candidate in candidates)


def _request_additional_info(request: Any) -> dict[str, Any]:
    def decode_info(raw_info: Any) -> dict[str, Any]:
        if isinstance(raw_info, dict):
            return raw_info
        info = deserialize_additional_information(raw_info)
        return info if isinstance(info, dict) else {}

    info = decode_info(getattr(request, "omni_stage_payload", None))
    current_info = decode_info(getattr(request, "additional_information", None))
    if current_info:
        info = {**info, **current_info}

    nested_info = info.get("additional_information") if isinstance(info, dict) else None
    if nested_info is not None:
        nested_info = decode_info(nested_info)
        if isinstance(nested_info, dict):
            # Connector payloads wrap the original OpenAI request metadata under
            # `additional_information`; expose those tts_* keys at the level
            # consumed by asr2aura/aura2tts while preserving explicit top-level
            # overrides.
            info = {**nested_info, **info}

    return info


def _request_output_text(request: Any) -> str:
    output_text = getattr(request, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    if isinstance(output_text, list) and output_text and isinstance(output_text[0], str):
        return output_text[0]
    return ""


def _clean_asr_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text
    for marker in (
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
        "<|image_pad|>",
    ):
        cleaned = cleaned.replace(marker, "")
    return " ".join(cleaned.split()).strip()


def _clean_tts_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def _aura_system_prompt(additional_info: dict[str, Any]) -> str:
    return (
        _first_str(additional_info.get("aura_system_prompt"), DEFAULT_AURA_SYSTEM_PROMPT) or DEFAULT_AURA_SYSTEM_PROMPT
    )


def _merged_vision_multimodal_data(*sources: Any) -> dict[str, Any]:
    multi_modal_data: dict[str, Any] = {}
    for source in sources:
        if isinstance(source, dict):
            multi_modal_data.update(source)
    return _vision_multimodal_data(multi_modal_data)


def _tts_additional_info(additional_info: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in additional_info.items() if isinstance(key, str) and key.startswith("tts_")}


def _build_aura_input_payload(
    *,
    transcript: str,
    additional_info: dict[str, Any],
    multi_modal_data: dict[str, Any],
    requires_multimodal_data: bool,
    mm_processor_kwargs: Any = None,
    tokenizer: Any | None = None,
    include_empty_multimodal_data: bool = False,
) -> dict[str, Any]:
    prompt = _aura_prompt(_aura_system_prompt(additional_info), transcript, multi_modal_data)
    next_input: dict[str, Any] = {"prompt": prompt}
    if tokenizer is not None:
        prompt_token_ids = tokenizer.encode(prompt)
        next_input["prompt_token_ids"] = prompt_token_ids
        next_input["ids"] = {"prompt": prompt_token_ids}
    if requires_multimodal_data and (multi_modal_data or include_empty_multimodal_data):
        next_input["multi_modal_data"] = multi_modal_data
    if mm_processor_kwargs is not None:
        next_input["mm_processor_kwargs"] = mm_processor_kwargs
    tts_info = _tts_additional_info(additional_info)
    if tts_info:
        next_input["additional_information"] = tts_info
    return next_input


def _tts_info_and_prompt_len(
    additional_info: dict[str, Any],
    *,
    assistant_token_ids: list[int],
    text: str,
    prompt_len_token_ids: list[int] | None = None,
    default_language: str = "Chinese",
    allow_default_base_refs: bool = True,
) -> tuple[dict[str, Any], int]:
    task_type = _first_value(additional_info.get("tts_task_type"), "Base")
    language = _first_value(additional_info.get("tts_language"), default_language)
    instruct = _first_value(additional_info.get("tts_instruct"), "")
    max_new_tokens = _first_value(additional_info.get("tts_max_new_tokens"), 2048)
    tts_info: dict[str, Any] = {
        "task_type": [task_type],
        "language": [language],
        "instruct": [instruct],
        "max_new_tokens": [int(max_new_tokens)],
    }
    if assistant_token_ids:
        tts_info[PRECOMPUTED_TEXT_IDS_KEY] = [assistant_token_ids]
        prompt_len = _estimate_tts_prompt_len_from_token_ids(
            assistant_token_ids,
            task_type=str(task_type),
            language=str(language),
            instruct=str(instruct),
        )
    else:
        text = _clean_tts_text(text)
        tts_info["text"] = [text]
        text_token_count_proxy = max(1, len(text)) if isinstance(text, str) else 1
        token_ids_for_len = (
            prompt_len_token_ids
            if prompt_len_token_ids
            else [0] * (text_token_count_proxy + len(QWEN_ASSISTANT_PREFIX_IDS) + len(QWEN_ASSISTANT_SUFFIX_IDS))
        )
        prompt_len = _estimate_tts_prompt_len_from_token_ids(
            token_ids_for_len,
            task_type=str(task_type),
            language=str(language),
            instruct=str(instruct),
        )

    if task_type == "Base":
        ref_audio = _first_value(additional_info.get("tts_ref_audio"), None)
        ref_text = _first_value(additional_info.get("tts_ref_text"), None)
        if allow_default_base_refs:
            ref_audio = ref_audio or default_qwen3_tts_ref_audio_path()
            ref_text = ref_text or DEFAULT_QWEN3_TTS_REF_TEXT
        x_vector_only_mode = _first_bool(additional_info.get("tts_x_vector_only_mode"), False)
        ref_code_len_value = _first_value(additional_info.get("tts_ref_code_length"), None)
        ref_code_len = int(ref_code_len_value) if isinstance(ref_code_len_value, int) else None
        if not x_vector_only_mode and ref_code_len is None:
            ref_code_len = _estimate_ref_code_len_from_ref_audio(ref_audio)
        if ref_code_len is not None:
            tts_info["ref_code_length"] = [int(ref_code_len)]
            prompt_len = _estimate_tts_prompt_len_from_token_ids(
                assistant_token_ids if assistant_token_ids else [0] * max(1, len(text)),
                task_type="Base",
                language=str(language),
                instruct=str(instruct),
                x_vector_only_mode=x_vector_only_mode,
                ref_code_len=ref_code_len,
            )
        if ref_audio:
            tts_info["ref_audio"] = [ref_audio]
        if ref_text:
            tts_info["ref_text"] = [ref_text]
        tts_info["x_vector_only_mode"] = [x_vector_only_mode]

    elif task_type == "CustomVoice":
        speaker = _first_value(additional_info.get("tts_speaker"), "Vivian")
        tts_info["speaker"] = [_normalize_qwen3_tts_speaker(speaker)]

    return tts_info, prompt_len


def _source_prompt_by_request_id(source_outputs: list[Any], prompt: Any) -> dict[str, dict[str, Any]]:
    prompts = _as_list(prompt)
    return {
        str(getattr(source_output, "request_id", idx)): _as_prompt_dict(prompt_item)
        for idx, (source_output, prompt_item) in enumerate(zip(source_outputs, prompts))
    }


def _vision_placeholder(multi_modal_data: dict[str, Any]) -> str:
    if "video" in multi_modal_data:
        return "<|vision_start|><|video_pad|><|vision_end|>"
    if "image" in multi_modal_data:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    return ""


def _vision_multimodal_data(multi_modal_data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in multi_modal_data.items() if key in {"image", "video"}}


def _aura_prompt(system_prompt: str, transcript: str, multi_modal_data: dict[str, Any]) -> str:
    vision = _vision_placeholder(multi_modal_data)
    query = transcript.strip()
    user_body = f"{vision}{query}" if query else vision

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|><|im_start|>user\n{user_body}<|im_end|><|im_start|>assistant\n"
    )


def asr2aura(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = True,
) -> list[dict[str, Any]]:
    """Build AURA Qwen3-VL prompts from ASR transcripts and original video payloads."""
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[dict[str, Any]] = []
    for idx, source_output in enumerate(source_outputs):
        src_prompt = prompt_by_request_id.get(str(getattr(source_output, "request_id", idx)), {})
        additional_info = src_prompt.get("additional_information") or {}
        transcript = _extract_text(source_output)
        multi_modal_data = _merged_vision_multimodal_data(
            src_prompt.get("multi_modal_data") or {},
            additional_info.get("deferred_multi_modal_data") or {},
        )

        next_inputs.append(
            _build_aura_input_payload(
                transcript=transcript,
                additional_info=additional_info,
                multi_modal_data=multi_modal_data,
                requires_multimodal_data=requires_multimodal_data,
                mm_processor_kwargs=src_prompt.get("mm_processor_kwargs"),
                include_empty_multimodal_data=True,
            )
        )
    return next_inputs


def asr2aura_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any | None = None,
    request: Any | None = None,
    is_finished: bool = False,
    **_: Any,
) -> dict[str, Any] | None:
    """Accumulate ASR text chunks and emit one complete AURA input at ASR finish."""
    del multimodal_output
    if request is None:
        raise ValueError("asr2aura_async_chunk requires request.")

    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    finished = bool(is_finished or request.is_finished())
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    state = request_payload.setdefault(str(request_id), {})
    if not isinstance(state, dict):
        state = {}
        request_payload[str(request_id)] = state

    output_text = _request_output_text(request)
    if output_text:
        previous_text = str(state.get("asr_text", ""))
        cleaned_output_text = _clean_asr_text(output_text)
        state["asr_text"] = (
            cleaned_output_text
            if cleaned_output_text.startswith(previous_text)
            else _clean_asr_text(previous_text + output_text)
        )

    if not finished:
        return None

    tokenizer = cached_tokenizer_from_config(transfer_manager.config)
    if not state.get("asr_text"):
        token_ids = _ensure_int_list(getattr(request, "output_token_ids", []) or [])
        if token_ids:
            state["asr_text"] = _clean_asr_text(tokenizer.decode(token_ids))

    additional_info = _request_additional_info(request)
    multi_modal_data = _merged_vision_multimodal_data(
        getattr(request, "multi_modal_data", None) or {},
        additional_info.get("deferred_multi_modal_data") or {},
    )
    mm_processor_kwargs = getattr(request, "mm_processor_kwargs", None)

    payload = _build_aura_input_payload(
        transcript=str(state.get("asr_text", "")),
        additional_info=additional_info,
        multi_modal_data=multi_modal_data,
        requires_multimodal_data=True,
        mm_processor_kwargs=mm_processor_kwargs,
        tokenizer=tokenizer,
    )
    return payload


def _estimate_ref_code_len_from_ref_audio(ref_audio: Any) -> int | None:
    """Estimate Qwen3-TTS ref_code length from a ref-audio payload.

    For Qwen3-TTS 12Hz models, code length is approximately:
        ceil(duration_seconds * 12.5)
    i.e. one codec frame per 1920 samples at 24kHz.
    """

    codec_frame_rate = 24000.0 / 1920.0

    # Unwrap common list wrappers.
    item = ref_audio
    while isinstance(item, list) and item:
        item = item[0]

    # Accept tuple/list like (wav, sr).
    if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)):
        wav, sr = item
        sr_i = int(sr)
        if sr_i <= 0:
            return None
        if hasattr(wav, "__len__"):
            n_samples = len(wav)
        elif hasattr(wav, "shape"):
            shape = getattr(wav, "shape", None)
            if not shape:
                return None
            n_samples = shape[-1] if len(shape) > 1 else shape[0]
        else:
            return None
        if n_samples <= 0:
            return None
        return max(1, int(math.ceil((float(n_samples) / float(sr_i)) * codec_frame_rate)))

    # Accept file path (wav only).
    if isinstance(item, str) and item:
        audio_path = item
        if not os.path.isfile(audio_path) or not audio_path.lower().endswith(".wav"):
            return None
        try:
            info = sf.info(audio_path)
            n_frames = int(info.frames)
            sr = int(info.samplerate)
            if n_frames <= 0 or sr <= 0:
                return None
            return max(1, int(math.ceil((float(n_frames) / float(sr)) * codec_frame_rate)))
        except Exception:
            return None

    return None


def _estimate_tts_prompt_len_from_token_ids(
    token_ids: list[int],
    *,
    task_type: str = "Base",
    language: str = "Chinese",
    instruct: str = "",
    x_vector_only_mode: bool = False,
    non_streaming_mode: bool | None = None,
    ref_code_len: int | None = None,
) -> int:
    """Estimate Talker prefill length from prompt structure.

    This mirrors Qwen3-TTS prompt assembly at length level:
      prompt_len = instruct_len + role_len + codec_prefix_len + text/icl term
    """

    # Official defaults: Base -> streaming, others -> non-streaming.
    if non_streaming_mode is None:
        non_streaming_mode = task_type in ("CustomVoice", "VoiceDesign")

    # We do not have tokenizer here; use char length as a monotonic proxy.
    instruct_len = len(instruct.strip()) if isinstance(instruct, str) else 0
    assistant_len = max(0, len(token_ids))

    # role_len = 3; codec_prefix_len = (prefill_len + speaker_len + 2) - 1
    # prefill_len = 4 when language_id exists else 3. Use non-auto language as
    # the language-id-present proxy.
    has_language_id = isinstance(language, str) and language.strip().lower() != "auto"
    prefill_len = 4 if has_language_id else 3
    speaker_len = 1 if task_type in ("CustomVoice", "Base") else 0
    base_len = instruct_len + 3 + (prefill_len + speaker_len + 2 - 1)
    if task_type in ("CustomVoice", "VoiceDesign"):
        if non_streaming_mode:
            prompt_len = base_len + max(0, assistant_len - 6)
        else:
            prompt_len = base_len + 1
        return int(prompt_len)

    if task_type == "Base":
        in_context_mode = not bool(x_vector_only_mode)
        if in_context_mode and ref_code_len is not None:
            codec_lens = 1 + int(ref_code_len)
            if non_streaming_mode:
                # Exact non-streaming ICL needs ref_ids token length; unavailable
                # in this processor. Keep a conservative upper estimate.
                prompt_len = base_len + codec_lens + max(0, assistant_len - 8) + 1
            else:
                # Streaming ICL exact length term: 1 + ref_code_len
                prompt_len = base_len + codec_lens
        else:
            # Base x-vector-only (or missing ref_code length) follows CV shape.
            if non_streaming_mode:
                prompt_len = base_len + max(0, assistant_len - 6)
            else:
                prompt_len = base_len + 1
        return int(prompt_len)

    # Defensive fallback for unknown task types.
    return int(base_len + max(assistant_len, 1))


def aura2tts(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Convert AURA text output into Qwen3-TTS Talker requests."""
    del requires_multimodal_data
    del streaming_context
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[OmniTokensPrompt] = []
    for idx, source_output in enumerate(source_outputs):
        text = _extract_text(source_output).strip()
        if not _is_finished(source_output) and text and SILENT_TEXT.startswith(text):
            # AURA may stream the special silent marker token-by-token. Hold
            # these prefixes until the marker is complete so TTS never speaks it.
            continue
        if not text or text == SILENT_TEXT:
            continue

        src_prompt = prompt_by_request_id.get(str(getattr(source_output, "request_id", idx)), {})
        additional_info = src_prompt.get("additional_information") or {}
        assistant_token_ids_for_len = _qwen3_tts_assistant_token_ids_from_aura(source_output)
        pass_token_ids = _first_bool(additional_info.get("tts_pass_token_ids"), False)
        use_token_ids = pass_token_ids and bool(assistant_token_ids_for_len)
        tts_info, prompt_len = _tts_info_and_prompt_len(
            additional_info,
            assistant_token_ids=assistant_token_ids_for_len if use_token_ids else [],
            text=text,
            prompt_len_token_ids=assistant_token_ids_for_len if not use_token_ids else None,
            default_language="English",
            allow_default_base_refs=False,
        )

        task_type = _first_value(additional_info.get("tts_task_type"), "Base")
        if task_type == "Base":
            if not tts_info.get("ref_audio") or not tts_info.get("ref_text"):
                raise ValueError("AURA Base TTS requires tts_ref_audio and tts_ref_text.")
        next_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=tts_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return next_inputs


def aura2tts_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any | None = None,
    request: Any | None = None,
    is_finished: bool = False,
    **_: Any,
) -> dict[str, Any] | None:
    """Accumulate AURA output and emit one Qwen3-TTS Talker input at finish."""
    del multimodal_output
    if request is None:
        raise ValueError("aura2tts_async_chunk requires request.")

    content_ids = _trim_aura_response_token_ids(_ensure_int_list(getattr(request, "output_token_ids", []) or []))
    finished = bool(is_finished or request.is_finished())
    if content_ids and _is_silent_token_prefix(content_ids):
        return None

    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    state = request_payload.setdefault(str(request_id), {})
    if not isinstance(state, dict):
        state = {}
        request_payload[str(request_id)] = state

    request_text = _clean_tts_text(_request_output_text(request))
    if content_ids:
        state["aura2tts_content_ids"] = content_ids
    if request_text:
        previous_text = str(state.get("aura2tts_text", ""))
        state["aura2tts_text"] = (
            request_text if request_text.startswith(previous_text) else _clean_tts_text(previous_text + request_text)
        )

    additional_info = _request_additional_info(request)
    tts_metadata = _tts_additional_info(additional_info)
    if tts_metadata:
        state["aura2tts_tts_metadata"] = dict(tts_metadata)

    if not finished:
        return None

    content_ids = list(state.get("aura2tts_content_ids", content_ids) or [])
    if not content_ids:
        return None
    if _is_silent_token_prefix(content_ids):
        return None

    cached_tts_metadata = state.get("aura2tts_tts_metadata")
    if isinstance(cached_tts_metadata, dict):
        additional_info = {**cached_tts_metadata, **additional_info}
    pass_token_ids = _first_bool(additional_info.get("tts_pass_token_ids"), False)
    request_text = _clean_tts_text(str(state.get("aura2tts_text", ""))) or request_text
    if not request_text and not pass_token_ids:
        try:
            tokenizer = cached_tokenizer_from_config(transfer_manager.config)
            request_text = _clean_tts_text(tokenizer.decode(content_ids))
        except Exception:
            logger.exception(
                "[aura2tts_async_chunk] req=%s failed to decode AURA token ids; falling back to token ids",
                getattr(request, "request_id", None),
            )

    assistant_token_ids = QWEN_ASSISTANT_PREFIX_IDS + content_ids + QWEN_ASSISTANT_SUFFIX_IDS
    use_token_ids = pass_token_ids
    if not use_token_ids and not request_text:
        use_token_ids = True
    tts_info, prompt_len = _tts_info_and_prompt_len(
        additional_info,
        assistant_token_ids=assistant_token_ids if use_token_ids else [],
        text=request_text,
        prompt_len_token_ids=assistant_token_ids if not use_token_ids else None,
    )

    payload = {
        **tts_info,
        "prompt_token_ids": [0] * prompt_len,
    }
    return payload
