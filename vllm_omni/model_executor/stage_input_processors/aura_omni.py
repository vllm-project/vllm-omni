# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage processors for the AURA Omni pipeline."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import regex as re
import soundfile as sf
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)

DEFAULT_AURA_SYSTEM_PROMPT = (
    "You are receiving a live video stream where the final frame is the present moment. "
    "Respond only when a response is needed based on the user's message or the visual context. "
    "Otherwise, output `<|silent|>` to signify silence."
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
DEFAULT_QWEN3_TTS_REF_AUDIO = "vllm-omni/tests/assets/qwen3_tts/clone_2.wav"
DEFAULT_QWEN3_TTS_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
)
# Used only to size Talker prompt_token_ids placeholders (must match build_prompt_embeds).
DEFAULT_QWEN3_TTS_TOKENIZER = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

logger = init_logger(__name__)

# Lazy cache: (path, tokenizer, codec_language_id, spk_is_dialect)
_qwen3_tts_prompt_len_cache: dict[str, Any] | None = None


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
    value = _first_value(value, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_qwen3_tts_tokenizer_path(additional_info: dict[str, Any] | None = None) -> str:
    """Resolve Qwen3-TTS tokenizer / config path for prompt_len parity."""
    if additional_info:
        for key in ("tts_tokenizer", "tts_model", "qwen3_tts_model"):
            raw = _first_value(additional_info.get(key), None)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    for env_key in ("VLLM_AURA_TTS_TOKENIZER", "VLLM_AURA_TTS_MODEL"):
        env = os.environ.get(env_key, "").strip()
        if env:
            return env
    for candidate in (
        "/workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "/workspace/models/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe",
        str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    ):
        path = Path(candidate)
        if path.is_dir() and (path / "tokenizer_config.json").is_file():
            return candidate
        snaps = path / "snapshots"
        if snaps.is_dir():
            for snap in sorted(snaps.iterdir()):
                if (snap / "tokenizer_config.json").is_file():
                    return str(snap)
    return DEFAULT_QWEN3_TTS_TOKENIZER


def _load_qwen3_tts_prompt_len_tools(
    additional_info: dict[str, Any] | None = None,
) -> tuple[Any, Mapping[str, int] | None, Mapping[str, object] | None] | None:
    """Load tokenizer + talker dialect maps for official prompt_len estimate."""
    global _qwen3_tts_prompt_len_cache
    path = _resolve_qwen3_tts_tokenizer_path(additional_info)
    if (
        isinstance(_qwen3_tts_prompt_len_cache, dict)
        and _qwen3_tts_prompt_len_cache.get("path") == path
        and _qwen3_tts_prompt_len_cache.get("tokenizer") is not None
    ):
        return (
            _qwen3_tts_prompt_len_cache["tokenizer"],
            _qwen3_tts_prompt_len_cache.get("codec_language_id"),
            _qwen3_tts_prompt_len_cache.get("spk_is_dialect"),
        )
    try:
        from transformers import AutoConfig, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            padding_side="left",
        )
        codec_language_id = None
        spk_is_dialect = None
        try:
            hf_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            talker_config = getattr(hf_config, "talker_config", None) or hf_config
            codec_language_id = getattr(talker_config, "codec_language_id", None)
            spk_is_dialect = getattr(talker_config, "spk_is_dialect", None)
        except Exception as cfg_err:
            cfg_path = Path(path) / "config.json"
            if cfg_path.is_file():
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                talker = raw.get("talker_config") if isinstance(raw.get("talker_config"), dict) else raw
                codec_language_id = talker.get("codec_language_id")
                spk_is_dialect = talker.get("spk_is_dialect")
            else:
                logger.warning("Qwen3-TTS talker_config unavailable for prompt_len (%s): %s", path, cfg_err)
        _qwen3_tts_prompt_len_cache = {
            "path": path,
            "tokenizer": tokenizer,
            "codec_language_id": codec_language_id,
            "spk_is_dialect": spk_is_dialect,
        }
        return tokenizer, codec_language_id, spk_is_dialect
    except Exception as e:
        logger.warning("Failed to load Qwen3-TTS tokenizer for prompt_len (%s): %s", path, e)
        return None


def _estimate_tts_prompt_len_official(
    tts_info: dict[str, Any],
    *,
    task_type: str,
    additional_info: dict[str, Any] | None = None,
) -> int | None:
    """Match standalone speech API: real BPE + estimate_prompt_len_from_additional_information."""
    tools = _load_qwen3_tts_prompt_len_tools(additional_info)
    if tools is None:
        return None
    tokenizer, codec_language_id, spk_is_dialect = tools
    try:
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
            Qwen3TTSPromptEmbedsBuilder,
        )

        return int(
            Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                additional_information=tts_info,
                task_type=task_type,
                tokenize_prompt=lambda t: tokenizer(t, padding=False)["input_ids"],
                codec_language_id=codec_language_id if isinstance(codec_language_id, dict) else None,
                spk_is_dialect=spk_is_dialect if isinstance(spk_is_dialect, dict) else None,
            )
        )
    except Exception as e:
        logger.warning("Official Qwen3-TTS prompt_len estimate failed; falling back to heuristic: %s", e)
        return None


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


def _one_video_item(video: Any) -> Any:
    """The current clip, if ``multi_modal_data['video']`` is one ``(array, meta)``."""
    if isinstance(video, list) and len(video) == 1 and isinstance(video[0], tuple):
        return video[0]
    return None


def _vision_placeholder(multi_modal_data: dict[str, Any]) -> str:
    if "video" in multi_modal_data:
        return "<|vision_start|><|video_pad|><|vision_end|>"
    if "image" in multi_modal_data:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    return ""


def _vision_multimodal_data(multi_modal_data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in multi_modal_data.items() if key in {"image", "video"}}


def _aura_prompt(
    system_prompt: str,
    transcript: str,
    multi_modal_data: dict[str, Any],
    history_prefix: str = "",
) -> str:
    vision = _vision_placeholder(multi_modal_data)
    query = transcript.strip()
    user_body = f"{vision}{query}" if query else vision
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"{history_prefix}"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _strip_assistant_text(text: str) -> str:
    """Remove think wrappers and ChatML specials before TTS / client text.

    Stage1 uses ``skip_special_tokens=False`` so ``<|silent|>`` / ``<|im_end|>``
    remain visible for silent detection; strip ChatML markers (not silent)
    before spoken TTS / transcript emission.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    # Drop an unclosed leading think block if the model is still inside it.
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("</think>", "")
    cleaned = re.sub(r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>", "", cleaned)
    return cleaned.strip()


def is_effectively_silent(text: str | None) -> bool:
    """True for empty / whitespace-only / exact ``<|silent|>`` Stage1 text."""
    if not isinstance(text, str):
        return False
    stripped = _strip_assistant_text(text)
    return not stripped or stripped == SILENT_TEXT


# Native AURA sentence boundaries for incremental Stage1→TTS handoff.
# Same rule as AURA_026 ``aura2tts_async_chunk``. Duplex applies it in the
# orchestrator (not SHM ``from_stage_1``, which clears Talker text).
_NATIVE_TTS_SENT_ENDS = frozenset("。！？；.!?;\n")
_NATIVE_TTS_COMMA_ENDS = frozenset("，,")
_NATIVE_TTS_MIN_CHARS = 10
_NATIVE_TTS_MIN_EMIT_CHARS = 30


def _sentence_tts_enabled() -> bool:
    """Emit TTS per sentence while Stage1 still generates.

    Default on. Disable with ``VLLM_AURA_SENTENCE_TTS=0``.
    """
    raw = (os.environ.get("VLLM_AURA_SENTENCE_TTS") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _sentence_tts_min_emit_chars() -> int:
    """Minimum content chars before a mid-generation TTS handoff."""
    raw = (os.environ.get("VLLM_AURA_SENTENCE_TTS_MIN_CHARS") or "").strip()
    if not raw:
        return _NATIVE_TTS_MIN_EMIT_CHARS
    try:
        return max(1, min(200, int(raw)))
    except ValueError:
        return _NATIVE_TTS_MIN_EMIT_CHARS


def _tts_content_char_count(text: str) -> int:
    """Count alphanumeric / CJK content chars (ignore punctuation/whitespace)."""
    return sum(1 for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _pop_native_tts_sentence(buf: str) -> tuple[str | None, str]:
    """Pop one sentence from ``buf``; return (sentence_or_None, rest)."""
    if not buf:
        return None, buf
    split_pos = -1
    for i, ch in enumerate(buf):
        if ch in _NATIVE_TTS_SENT_ENDS:
            split_pos = i + 1
            break
        if ch in _NATIVE_TTS_COMMA_ENDS and i + 1 >= _NATIVE_TTS_MIN_CHARS:
            split_pos = i + 1
            break
    if split_pos < 0:
        return None, buf
    sentence = buf[:split_pos]
    rest = buf[split_pos:]
    if not sentence.strip():
        return _pop_native_tts_sentence(rest)
    return sentence, rest


def _pop_emit_ready_tts_text(
    buf: str,
    min_chars: int | None = None,
) -> tuple[str | None, str]:
    """Pop sentences until content length >= ``min_chars``."""
    if min_chars is None:
        min_chars = _sentence_tts_min_emit_chars()
    parts: list[str] = []
    rest = buf
    while True:
        sentence, rest = _pop_native_tts_sentence(rest)
        if sentence is None:
            break
        parts.append(sentence)
        if _tts_content_char_count("".join(parts)) >= min_chars:
            return "".join(parts), rest
    if not parts:
        return None, buf
    return None, "".join(parts) + rest


def _tool_marker_pending(text: str) -> bool:
    lowered = text.lower()
    return "<tool_call" in lowered or "</tool_call" in lowered


def next_duplex_sentence_chunk(state: dict[str, Any], raw_text: str, *, finished: bool) -> str | None:
    """Return the next Talker sentence, or None when this output must not start TTS.

    ``state`` is per Stage1 request and is mutated. Stage1 finish flushes any
    leftover even under the min-char floor. Silent, an unclosed ``<think>``,
    and tool markers do not emit mid-generation. A silent finish returns None
    so the legacy full-text path can drop TTS.
    """
    if not _sentence_tts_enabled():
        return None
    raw = raw_text or ""
    already = int(state.get("emits", 0))
    if already == 0 and (is_silent_text_prefix(raw) or is_effectively_silent(raw)):
        return None
    think_open = "<think>" in raw.lower() and "</think>" not in raw.lower()
    tool_pending = _tool_marker_pending(raw)
    if (think_open or tool_pending) and not finished:
        return None

    text = _strip_assistant_text(raw)
    if is_effectively_silent(text) and already == 0:
        return None

    emitted_prefix = str(state.get("emitted_prefix", ""))
    pending = str(state.get("pending", ""))
    if text.startswith(emitted_prefix):
        new_tail = text[len(emitted_prefix) :]
    else:
        new_tail = text
        pending = ""
        emitted_prefix = ""
    if new_tail:
        pending = pending + new_tail
        emitted_prefix = text
        state["emitted_prefix"] = emitted_prefix

    if not finished:
        sentence, pending = _pop_emit_ready_tts_text(pending)
        state["pending"] = pending
        if sentence is None:
            return None
        state["emits"] = already + 1
        state["last"] = sentence.strip()
        return state["last"]

    state["pending"] = ""
    remainder = _strip_assistant_text(pending).strip()
    if not remainder or is_effectively_silent(remainder):
        return None
    state["emits"] = already + 1
    state["last"] = remainder
    return remainder


def is_silent_text_prefix(text: str | None) -> bool:
    """True while streamed text is still a prefix of ``<|silent|>``.

    Holds Stage1 transcript deltas until the turn finishes (or diverges), so
    partial ``<|sil`` fragments do not leak before a silent short-circuit.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return True
    return SILENT_TEXT.startswith(stripped)


def _normalize_asr_transcript(transcript: str) -> str:
    """Strip Qwen3-ASR markup wrappers so AURA sees plain user text.

    Observed Stage0 text looks like:
      ``language Chinese<asr_text>出现《古韵》这本书的时候，提醒我。``
    """
    text = (transcript or "").strip()
    if not text:
        return ""
    marker = "<asr_text>"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    # Drop a leading ``language <lang>`` line if still present.
    if text.lower().startswith("language "):
        parts = text.split(None, 2)
        if len(parts) >= 3:
            text = parts[2].strip()
        elif len(parts) == 2:
            text = ""
    return text


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
        system_prompt = _first_value(additional_info.get("aura_system_prompt"), DEFAULT_AURA_SYSTEM_PROMPT)
        transcript = _normalize_asr_transcript(_extract_text(source_output))
        # Vision-follow: placeholder zeros may ASR into noise; trust client is_speech.
        if additional_info.get("is_speech") is False:
            transcript = ""
        multi_modal_data = {}
        source_multi_modal_data = src_prompt.get("multi_modal_data") or {}
        if isinstance(source_multi_modal_data, dict):
            multi_modal_data.update(source_multi_modal_data)
        deferred_multi_modal_data = additional_info.get("deferred_multi_modal_data") or {}
        if isinstance(deferred_multi_modal_data, dict):
            multi_modal_data.update(deferred_multi_modal_data)
        multi_modal_data = _vision_multimodal_data(multi_modal_data)

        history_prefix = ""
        prompt_mm = multi_modal_data
        session_id = additional_info.get("session_id") or additional_info.get("aura_session_id")
        if additional_info.get("aura_duplex") and isinstance(session_id, str) and session_id:
            from vllm_omni.model_executor.models.aura_omni.duplex.history import (
                get_or_create_session_history,
            )

            history = get_or_create_session_history(session_id)
            current_video = _one_video_item(multi_modal_data.get("video"))
            history.begin_user_turn(transcript, video=current_video)
            history_prefix = history.render_prefix()
            prior_videos = history.retained_videos()
            # Native get_vllm_inputs: every retained clip is a <|video_pad|> in
            # order, then this turn. The placeholder on the current user message
            # is only this clip, so pad count matches multi_modal_data["video"].
            if prior_videos or current_video is not None:
                merged = list(prior_videos)
                if current_video is not None:
                    merged.append(current_video)
                multi_modal_data = dict(multi_modal_data)
                multi_modal_data["video"] = merged
            prompt_mm = dict(multi_modal_data)
            if prior_videos:
                if current_video is None:
                    prompt_mm.pop("video", None)
                else:
                    prompt_mm["video"] = [current_video]
            else:
                prompt_mm = multi_modal_data
            if not transcript:
                last_assistant = next(
                    (
                        message["content"]
                        for message in reversed(history.messages)
                        if message.get("role") == "assistant" and isinstance(message.get("content"), str)
                    ),
                    "",
                )
                logger.info(
                    "[asr2aura] vision-follow history_prefix_len=%d has_assistant=%s last_assistant_len=%d",
                    len(history_prefix),
                    bool(last_assistant),
                    len(last_assistant) if isinstance(last_assistant, str) else 0,
                )

        next_input: dict[str, Any] = {
            "prompt": _aura_prompt(
                str(system_prompt),
                transcript,
                prompt_mm,
                history_prefix=history_prefix,
            ),
        }
        if isinstance(session_id, str) and session_id:
            next_input.setdefault("additional_information", {})
            # Preserve session id for aura2tts history commit.
            info = dict(additional_info)
            info["session_id"] = session_id
            info["aura_duplex"] = bool(additional_info.get("aura_duplex"))
            next_input["additional_information"] = info
        if requires_multimodal_data:
            next_input["multi_modal_data"] = multi_modal_data
        if src_prompt.get("mm_processor_kwargs") is not None:
            next_input["mm_processor_kwargs"] = src_prompt.get("mm_processor_kwargs")
        next_inputs.append(next_input)
    return next_inputs


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


def _approx_qwen_token_count(text: str) -> int:
    """Rough Qwen BPE length without loading a tokenizer.

    CJK / CJK punctuation ≈ 1 token each; contiguous non-CJK (incl. spaces)
    ≈ 1 token per 4 chars. Do **not** count spaces as their own tokens — that
    inflated English ``tts_instruct`` (~144 chars) from real ~35 to ~62 and
    zero-padded Talker prefill by ~100 (leading garbage / early cut).
    """
    if not text:
        return 0
    n = 0
    i = 0
    while i < len(text):
        code = ord(text[i])
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF:
            n += 1
            i += 1
            continue
        j = i + 1
        while j < len(text):
            cj = ord(text[j])
            if 0x4E00 <= cj <= 0x9FFF or 0x3400 <= cj <= 0x4DBF or 0x3000 <= cj <= 0x303F or 0xFF00 <= cj <= 0xFFEF:
                break
            j += 1
        n += max(1, (j - i + 3) // 4)
        i = j
    return n


def _estimate_instruct_prompt_tokens(instruct: str) -> int:
    """Token length of ``build_instruct_text(instruct)`` without a tokenizer."""
    body = instruct.strip() if isinstance(instruct, str) else ""
    if not body:
        return 0
    return 5 + _approx_qwen_token_count(body)


def _estimate_assistant_prompt_tokens(text: str) -> int:
    """Token length of ``build_assistant_text(text)`` without a tokenizer."""
    body = text if isinstance(text, str) else ""
    return max(8, 8 + _approx_qwen_token_count(body))


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

    ``instruct_len`` must be a *token* estimate. Using ``len(instruct)`` chars
    (e.g. 144 for the demo style string vs real ~40 tokens) zero-pads Talker
    prefill by ~100 and causes leading garbage / truncated speech.
    """

    # Official defaults: Base -> streaming, others -> non-streaming.
    if non_streaming_mode is None:
        non_streaming_mode = task_type in ("CustomVoice", "VoiceDesign")

    instruct_len = _estimate_instruct_prompt_tokens(instruct) if isinstance(instruct, str) else 0
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
        raw_text = _extract_text(source_output).strip()
        text = _strip_assistant_text(raw_text)
        src_prompt = prompt_by_request_id.get(str(getattr(source_output, "request_id", idx)), {})
        additional_info = src_prompt.get("additional_information") or {}
        close_only = bool(additional_info.get("aura_tts_close_only"))
        if is_effectively_silent(text) and not close_only:
            continue
        if close_only:
            text = ""
        task_type = _first_value(additional_info.get("tts_task_type"), "Base")
        language = _first_value(additional_info.get("tts_language"), "English")
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
            "max_new_tokens": [1 if close_only else int(_first_value(additional_info.get("tts_max_new_tokens"), 2048))],
        }
        if pass_token_ids and assistant_token_ids_for_len:
            tts_info[PRECOMPUTED_TEXT_IDS_KEY] = [assistant_token_ids_for_len]
        else:
            tts_info["text"] = [text]
        if ref_code_len is not None:
            tts_info["ref_code_length"] = [int(ref_code_len)]
        if task_type == "Base":
            ref_audio = ref_audio or _first_value(additional_info.get("tts_ref_audio"), None)
            ref_text = _first_value(additional_info.get("tts_ref_text"), None)
            if not ref_audio or not ref_text:
                raise ValueError("AURA Base TTS requires tts_ref_audio and tts_ref_text.")
            x_vector_only_mode = _first_bool(additional_info.get("tts_x_vector_only_mode"), False)
            tts_info["ref_audio"] = [ref_audio]
            tts_info["ref_text"] = [ref_text]
            tts_info["x_vector_only_mode"] = [x_vector_only_mode]
        elif task_type == "CustomVoice":
            tts_info["speaker"] = [
                _normalize_qwen3_tts_speaker(_first_value(additional_info.get("tts_speaker"), "Vivian"))
            ]

        if close_only:
            prompt_len = 1
        else:
            prompt_len = None
            if not (pass_token_ids and assistant_token_ids_for_len):
                prompt_len = _estimate_tts_prompt_len_official(
                    tts_info,
                    task_type=str(task_type),
                    additional_info=additional_info if isinstance(additional_info, dict) else None,
                )
            if prompt_len is None:
                if pass_token_ids and assistant_token_ids_for_len:
                    length_token_ids = assistant_token_ids_for_len
                else:
                    length_token_ids = [0] * _estimate_assistant_prompt_tokens(text)
                prompt_len = _estimate_tts_prompt_len_from_token_ids(
                    length_token_ids,
                    task_type=str(task_type),
                    language=str(language),
                    instruct=str(instruct),
                    x_vector_only_mode=x_vector_only_mode,
                    non_streaming_mode=non_streaming_mode,
                    ref_code_len=ref_code_len,
                )

        logger.info(
            "[aura2tts] task=%s language=%s speaker=%s text_len=%d instruct_len=%d "
            "prompt_len=%d close_only=%s text_preview=%r",
            task_type,
            language,
            tts_info.get("speaker", [None])[0],
            len(text),
            len(str(instruct).strip()) if instruct else 0,
            prompt_len,
            close_only,
            text[:120],
        )
        next_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] if close_only else [0] * int(prompt_len or 1),
                additional_information=tts_info,
                # Prefer runner data-plane so Talker preprocess still sees
                # text if legacy additional_information is wiped (e.g. a
                # mistaken Stage2 chunk-receiver edge).
                model_intermediate_buffer=dict(tts_info),
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return next_inputs
