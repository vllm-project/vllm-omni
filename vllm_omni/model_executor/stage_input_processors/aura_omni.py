# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage processors for the AURA Omni pipeline."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    DEFAULT_AURA_SYSTEM_PROMPT,
    SILENT_TEXT,
    SessionHistory,
    get_session_history,
    is_effectively_silent,
)

QWEN_IM_START_ID = 151644
QWEN_IM_END_ID = 151645
QWEN_ASSISTANT_ID = 77091

logger = init_logger(__name__)

_TURN_TRANSCRIPTS_BY_REQUEST: dict[str, str] = {}


def _aura_log_turn_prompt_enabled() -> bool:
    return os.environ.get("VLLM_AURA_LOG_TURN_PROMPT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


QWEN_NEWLINE_ID = 198
QWEN_ASSISTANT_PREFIX_IDS = [QWEN_IM_START_ID, QWEN_ASSISTANT_ID, QWEN_NEWLINE_ID]
QWEN_ASSISTANT_SUFFIX_IDS = [
    QWEN_IM_END_ID,
    QWEN_NEWLINE_ID,
    QWEN_IM_START_ID,
    QWEN_ASSISTANT_ID,
    QWEN_NEWLINE_ID,
]
DEFAULT_QWEN3_TTS_REF_AUDIO = "vllm-omni/tests/assets/qwen3_tts/clone_2.wav"
DEFAULT_QWEN3_TTS_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
)

_AURA_TTS_INFO_KEYS = (
    "tts_task_type",
    "tts_language",
    "tts_instruct",
    "tts_max_new_tokens",
    "tts_ref_audio",
    "tts_ref_text",
    "tts_x_vector_only_mode",
    "tts_speaker",
    "tts_non_streaming_mode",
    "tts_ref_code_length",
    "tts_pass_token_ids",
)


def default_qwen3_tts_ref_audio_path() -> str:
    """Return absolute path to the bundled ``clone_2.wav`` reference asset."""
    bundled = Path(__file__).resolve().parents[3] / "tests" / "assets" / "qwen3_tts" / "clone_2.wav"
    if bundled.is_file():
        return str(bundled)
    return DEFAULT_QWEN3_TTS_REF_AUDIO


def default_aura_tts_additional_information() -> dict[str, Any]:
    """Default Qwen3-TTS fields for AURA ``additional_information``.

    Matches Base checkpoint deployments (e.g. ``aura_omni_gpu23.yaml``).
    For CustomVoice checkpoints, set ``tts_task_type`` / ``tts_speaker`` explicitly.
    """
    return {
        "tts_task_type": "Base",
        "tts_language": "Chinese",
        "tts_instruct": "",
        "tts_ref_audio": default_qwen3_tts_ref_audio_path(),
        "tts_ref_text": DEFAULT_QWEN3_TTS_REF_TEXT,
        "tts_pass_token_ids": False,
    }


def aura_tts_additional_information_from_session(
    *,
    task_type: str | None = None,
    language: str | None = None,
    speaker: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
    pass_token_ids: bool | None = None,
) -> dict[str, Any]:
    """Merge WebSocket ``session.config`` TTS fields into ``additional_information``."""
    info = default_aura_tts_additional_information()
    if isinstance(task_type, str) and task_type.strip():
        info["tts_task_type"] = task_type.strip()
    if isinstance(language, str) and language.strip():
        info["tts_language"] = language.strip()
    if isinstance(instruct, str):
        info["tts_instruct"] = instruct
    if isinstance(speaker, str) and speaker.strip():
        info["tts_speaker"] = _normalize_qwen3_tts_speaker(speaker.strip())
    if isinstance(ref_audio, str) and ref_audio.strip():
        info["tts_ref_audio"] = ref_audio.strip()
    if isinstance(ref_text, str) and ref_text.strip():
        info["tts_ref_text"] = ref_text.strip()
    if pass_token_ids is not None:
        info["tts_pass_token_ids"] = bool(pass_token_ids)
    if info.get("tts_task_type") == "CustomVoice":
        info.pop("tts_ref_audio", None)
        info.pop("tts_ref_text", None)
    return info


def frames_to_video_tuple(
    frames: list[np.ndarray],
    *,
    fps: float,
    max_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stack per-turn frames into a ``(ndarray, metadata)`` video tuple for AURA."""
    if not frames:
        raise ValueError("At least one frame is required to build video_tuple")

    selected = list(frames[-max_frames:])
    if len(selected) == 1:
        all_frames = np.stack([selected[0], selected[0]], axis=0)
    else:
        all_frames = np.stack(selected, axis=0)

    if all_frames.shape[0] < 2:
        all_frames = np.concatenate([all_frames, all_frames], axis=0)[:2]
    elif all_frames.shape[0] > max_frames:
        all_frames = all_frames[-max_frames:]

    metadata = {
        "fps": fps,
        "duration": all_frames.shape[0] / fps,
        "total_num_frames": int(all_frames.shape[0]),
        "frames_indices": list(range(all_frames.shape[0])),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return all_frames, metadata


def build_aura_streaming_turn_additional_information(
    *,
    session_id: str,
    video_array: np.ndarray,
    video_metadata: dict[str, Any],
    system_prompt: str,
    skip_asr: bool,
    include_tts: bool,
    tts_task_type: str | None = None,
    tts_language: str | None = None,
    tts_speaker: str | None = None,
    tts_ref_audio: str | None = None,
    tts_ref_text: str | None = None,
    tts_instruct: str | None = None,
    tts_pass_token_ids: bool | None = None,
) -> dict[str, Any]:
    """Build ``additional_information`` for one AURA streaming inference turn."""
    additional_information: dict[str, Any] = {
        "aura_session_id": session_id,
        "deferred_multi_modal_data": {
            "video": [(video_array, video_metadata)],
        },
        "aura_system_prompt": [system_prompt],
        "omni_skip_stages": [0] if skip_asr else [],
    }
    if include_tts:
        additional_information.update(
            aura_tts_additional_information_from_session(
                task_type=tts_task_type,
                language=tts_language,
                speaker=tts_speaker,
                ref_audio=tts_ref_audio,
                ref_text=tts_ref_text,
                instruct=tts_instruct,
                pass_token_ids=tts_pass_token_ids,
            )
        )
    return additional_information


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
    value = _first_value(value, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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


def _clean_asr_transcript(text: str) -> str:
    """Strip Qwen3-ASR wrappers like ``language Chinese<asr_text>...``."""
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if "<asr_text>" in cleaned:
        cleaned = cleaned.split("<asr_text>", 1)[-1]
    cleaned = re.sub(r"^language\s+[\w-]+\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _normalize_request_id_for_transcript(request_id: str) -> str:
    """Map AsyncOmni internal ids (``{external}-{uuid8}``) to the external id.

    ``AsyncOmni.generate()`` rewrites ``request_id`` to an internal orchestrator id
    while the streaming handler keeps the external id for ``on_turn_complete``.
    Transcript storage must use the external id so ``pop_turn_transcript`` works.
    """
    rid = str(request_id).strip()
    if not rid:
        return rid
    head, tail = rid.rsplit("-", 1)
    if len(tail) == 8 and all(ch in "0123456789abcdefABCDEF" for ch in tail):
        return head
    return rid


def record_turn_transcript(request_id: str, transcript: str) -> None:
    _TURN_TRANSCRIPTS_BY_REQUEST[_normalize_request_id_for_transcript(request_id)] = transcript


def pop_turn_transcript(request_id: str | None) -> str:
    if not request_id:
        return ""
    return _TURN_TRANSCRIPTS_BY_REQUEST.pop(_normalize_request_id_for_transcript(str(request_id)), "")


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
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _asr2aura_single_turn(
    source_output: Any,
    src_prompt: dict[str, Any],
    *,
    requires_multimodal_data: bool,
) -> dict[str, Any]:
    """Stateless ASR→AURA prompt when no session fields are present on the request."""
    additional_info = src_prompt.get("additional_information") or {}
    system_prompt = _first_value(additional_info.get("aura_system_prompt"), DEFAULT_AURA_SYSTEM_PROMPT)
    transcript = _clean_asr_transcript(_extract_text(source_output))
    multi_modal_data: dict[str, Any] = {}
    source_multi_modal_data = src_prompt.get("multi_modal_data") or {}
    if isinstance(source_multi_modal_data, dict):
        multi_modal_data.update(source_multi_modal_data)
    deferred_multi_modal_data = additional_info.get("deferred_multi_modal_data") or {}
    if isinstance(deferred_multi_modal_data, dict):
        multi_modal_data.update(deferred_multi_modal_data)
    multi_modal_data = _vision_multimodal_data(multi_modal_data)

    next_input: dict[str, Any] = {
        "prompt": _aura_prompt(str(system_prompt), transcript, multi_modal_data),
    }
    if requires_multimodal_data:
        next_input["multi_modal_data"] = multi_modal_data
    if src_prompt.get("mm_processor_kwargs") is not None:
        next_input["mm_processor_kwargs"] = src_prompt.get("mm_processor_kwargs")
    return next_input


def asr2aura(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = True,
) -> list[dict[str, Any]]:
    """Build AURA Qwen3-VL prompts from ASR output and optional SessionHistory."""
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[dict[str, Any]] = []

    for idx, source_output in enumerate(source_outputs):
        src_prompt = prompt_by_request_id.get(str(getattr(source_output, "request_id", idx)), {})
        additional_info = src_prompt.get("additional_information") or {}
        session_id = additional_info.get("aura_session_id")
        history = get_session_history(str(session_id)) if session_id else None
        if session_id and history is None:
            logger.warning(
                "AURA session_id=%s not found in server-side store; "
                "falling back to aura_session_state or single-turn prompt",
                session_id,
            )

        if history is None and additional_info.get("aura_session_state") is None:
            next_inputs.append(
                _asr2aura_single_turn(
                    source_output,
                    src_prompt,
                    requires_multimodal_data=requires_multimodal_data,
                )
            )
            continue

        request_id = str(getattr(source_output, "request_id", idx))
        transcript = _clean_asr_transcript(_extract_text(source_output))
        record_turn_transcript(request_id, transcript)
        video_tuple = video_tuple_from_additional_info(additional_info)
        if history is None:
            history = SessionHistory.from_dict(additional_info["aura_session_state"])
            history.add_user_message(transcript, video_tuple=video_tuple)
            vllm_inputs = history.get_vllm_inputs()
        else:
            vllm_inputs = history.preview_vllm_inputs(transcript, video_tuple=video_tuple)

        if _aura_log_turn_prompt_enabled():
            logger.info(
                "AURA turn prompt request_id=%s transcript=%r: %s",
                request_id,
                transcript,
                _summarize_vllm_inputs(vllm_inputs),
            )

        next_input = {
            "prompt": vllm_inputs["prompt"],
            "additional_information": _copy_aura_tts_fields(additional_info),
        }
        system_prompt = _first_value(additional_info.get("aura_system_prompt"), DEFAULT_AURA_SYSTEM_PROMPT)
        next_input["additional_information"]["aura_system_prompt"] = [str(system_prompt)]

        if requires_multimodal_data:
            next_input["multi_modal_data"] = vllm_inputs.get("multi_modal_data", {})
        if src_prompt.get("mm_processor_kwargs") is not None:
            next_input["mm_processor_kwargs"] = src_prompt.get("mm_processor_kwargs")
        next_inputs.append(next_input)

    return next_inputs


def _normalize_video_tuple(
    frames: Any,
    metadata: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Return (uint8 ndarray [T,H,W,C], metadata) with at least two frames."""
    if frames is None:
        return None
    video_array = np.asarray(frames, dtype=np.uint8)
    if video_array.ndim != 4:
        return None
    meta = dict(metadata or {})
    if video_array.shape[0] < 2:
        video_array = np.concatenate([video_array, video_array], axis=0)[:2]
        meta = dict(meta)
        meta["total_num_frames"] = 2
        meta["duration"] = 2 / float(meta.get("fps", 2.0))
    return video_array, meta


def video_tuple_from_aura_turn_video(aura_turn_video: Any) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Legacy JSON-serializable turn video: ``{frames: list, metadata: dict}``."""
    if not isinstance(aura_turn_video, dict):
        return None
    return _normalize_video_tuple(aura_turn_video.get("frames"), aura_turn_video.get("metadata"))


def video_tuple_from_deferred_multi_modal(deferred: Any) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Read the first ``(ndarray, metadata)`` video entry from deferred multimodal data."""
    if not isinstance(deferred, dict):
        return None
    videos = deferred.get("video")
    if not videos:
        return None
    first = videos[0] if isinstance(videos, list) else videos
    if isinstance(first, (tuple, list)) and len(first) == 2:
        return _normalize_video_tuple(first[0], first[1] if isinstance(first[1], dict) else {})
    return None


def video_tuple_from_additional_info(additional_info: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Resolve per-turn video from ``deferred_multi_modal_data`` or legacy ``aura_turn_video``."""
    video_tuple = video_tuple_from_deferred_multi_modal(additional_info.get("deferred_multi_modal_data"))
    if video_tuple is not None:
        return video_tuple
    return video_tuple_from_aura_turn_video(additional_info.get("aura_turn_video"))


def _copy_aura_tts_fields(additional_info: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key in _AURA_TTS_INFO_KEYS:
        if key in additional_info:
            copied[key] = additional_info[key]
    return copied


def _summarize_vllm_inputs(vllm_inputs: dict[str, Any]) -> str:
    """JSON summary of AURA stage-1 prompt (text skeleton + video metadata, no pixels)."""
    videos = vllm_inputs.get("multi_modal_data", {}).get("video", [])
    video_info: list[dict[str, Any]] = []
    for vt in videos:
        arr, meta = vt
        video_info.append(
            {
                "frames": int(arr.shape[0]),
                "shape": list(arr.shape),
                "fps": meta.get("fps"),
                "duration": meta.get("duration"),
            }
        )
    return json.dumps(
        {
            "prompt_text": vllm_inputs.get("prompt", ""),
            "videos": video_info,
        },
        ensure_ascii=False,
    )


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
) -> list[OmniTokensPrompt]:
    """Convert AURA text output into Qwen3-TTS Talker requests."""
    del requires_multimodal_data
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[OmniTokensPrompt] = []
    for idx, source_output in enumerate(source_outputs):
        text = _extract_text(source_output).strip()
        if is_effectively_silent(text):
            continue

        src_prompt = prompt_by_request_id.get(str(getattr(source_output, "request_id", idx)), {})
        additional_info = src_prompt.get("additional_information") or {}
        task_type = _first_value(additional_info.get("tts_task_type"), "Base")
        language = _first_value(additional_info.get("tts_language"), "Chinese")
        instruct = _first_value(additional_info.get("tts_instruct"), "")
        x_vector_only_mode = _first_bool(additional_info.get("tts_x_vector_only_mode"), False)
        non_streaming_mode_raw = _first_value(additional_info.get("tts_non_streaming_mode"), None)
        non_streaming_mode = non_streaming_mode_raw if isinstance(non_streaming_mode_raw, bool) else None
        ref_code_len_raw = _first_value(additional_info.get("tts_ref_code_length"), None)
        ref_code_len = int(ref_code_len_raw) if isinstance(ref_code_len_raw, int) else None
        ref_audio = None
        ref_text = None
        if task_type == "Base" and not x_vector_only_mode and ref_code_len is None:
            ref_audio = _first_value(additional_info.get("tts_ref_audio"), None)
            ref_code_len = _estimate_ref_code_len_from_ref_audio(ref_audio)

        assistant_token_ids_for_len = _qwen3_tts_assistant_token_ids_from_aura(source_output)
        pass_token_ids = _first_bool(additional_info.get("tts_pass_token_ids"), False)
        tts_info = {
            "task_type": [task_type],
            "language": [language],
            "instruct": [instruct],
            "max_new_tokens": [int(_first_value(additional_info.get("tts_max_new_tokens"), 2048))],
        }
        if pass_token_ids and assistant_token_ids_for_len:
            tts_info[PRECOMPUTED_TEXT_IDS_KEY] = [assistant_token_ids_for_len]
        else:
            tts_info["text"] = [text]
        if ref_code_len is not None:
            tts_info["ref_code_length"] = [int(ref_code_len)]
        prompt_len = _estimate_tts_prompt_len_from_token_ids(
            assistant_token_ids_for_len if assistant_token_ids_for_len else [0] * max(0, len(text)),
            task_type=str(task_type),
            language=str(language),
            instruct=str(instruct),
            x_vector_only_mode=x_vector_only_mode,
            non_streaming_mode=non_streaming_mode,
            ref_code_len=ref_code_len,
        )

        if task_type == "Base":
            ref_audio = ref_audio or _first_value(additional_info.get("tts_ref_audio"), None)
            ref_text = _first_value(additional_info.get("tts_ref_text"), None)
            if not ref_audio:
                ref_audio = default_qwen3_tts_ref_audio_path()
            if not ref_text:
                ref_text = DEFAULT_QWEN3_TTS_REF_TEXT
            x_vector_only_mode = _first_bool(additional_info.get("tts_x_vector_only_mode"), False)
            tts_info["ref_audio"] = [ref_audio]
            tts_info["ref_text"] = [ref_text]
            tts_info["x_vector_only_mode"] = [x_vector_only_mode]
        elif task_type == "CustomVoice":
            tts_info["speaker"] = [
                _normalize_qwen3_tts_speaker(_first_value(additional_info.get("tts_speaker"), "Vivian"))
            ]
        next_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=tts_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return next_inputs
