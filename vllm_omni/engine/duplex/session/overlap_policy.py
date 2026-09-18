# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""How one duplex session reacts to input that arrives while it is speaking.

Short acknowledgement, barge-in, or keep listening: the rules read the session
config and the incoming append, and nothing else. Extracted from
``DuplexSessionRunner`` because they were a decision function wearing a
method's clothes -- the only runner state they ever touched was ``session`` --
and because reaching them previously meant driving a whole append through the
runner.
"""

from __future__ import annotations

import binascii
from collections.abc import Mapping

import numpy as np
import pybase64 as base64

from vllm_omni.engine.duplex.config import DuplexOverlapPolicy
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession


def decide(
    session: DuplexEngineSession,
    event: dict[str, object],
    payload: dict[str, object],
    *,
    auto_responds: bool,
) -> dict[str, object]:
    duration_ms = input_audio_duration_ms(event, payload)
    is_speech = input_looks_like_speech(session, event, payload)
    if not session.capabilities.supports_barge_in and event_requests_barge_in(event):
        return defer_unsupported_barge_in(session, duration_ms=duration_ms, is_speech=is_speech)
    explicit = event.get("overlap_action") or event.get("overlap")
    if isinstance(explicit, str):
        normalized = explicit.strip().lower()
        if normalized in {"barge_in", "interrupt", "cancel"}:
            return {
                "action": "barge_in",
                "reason": "client_overlap_action",
                "duration_ms": duration_ms,
                "buffer_audio": True,
            }
        if normalized in {"listen", "continue", "continue_output", "ack"}:
            session.reset_overlap_speech()
            return {
                "action": "listen",
                "reason": "client_overlap_action",
                "duration_ms": duration_ms,
                "buffer_audio": (
                    normalized == "listen" and is_speech and duration_ms > session.config.overlap_short_ack_ms
                ),
                "defer_runtime_append": True,
            }
        if normalized in {"drop", "ignore", "silence"}:
            session.reset_overlap_speech()
            return {
                "action": "drop",
                "reason": "client_overlap_action",
                "duration_ms": duration_ms,
                "buffer_audio": False,
            }

    if bool(event.get("force_barge_in", False)):
        return {
            "action": "barge_in",
            "reason": "client_force_barge_in",
            "duration_ms": duration_ms,
            "buffer_audio": True,
        }
    if auto_responds:
        if is_speech:
            session.accumulate_overlap_speech(duration_ms)
        speech_started = vad_speech_started(event, payload)
        if (
            session.capabilities.supports_barge_in
            and is_speech
            and session.config.overlap_policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value
            and speech_started is not False
        ):
            return {
                "action": "barge_in",
                "reason": ("server_vad_speech_started" if speech_started is True else "barge_in_on_speech"),
                "cancel_reason": "turn_detected" if speech_started is True else "barge_in",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
            }
        return {
            "action": "listen",
            "reason": "auto_response_continuous",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": True,
            "defer_runtime_append": False,
            "force_listen": event.get("force_listen") is True or payload.get("force_listen") is True,
            "preserve_realtime_input": True,
        }
    if bool(event.get("force_listen", False)):
        session.reset_overlap_speech()
        return {
            "action": "listen",
            "reason": "client_force_listen",
            "duration_ms": duration_ms,
            "buffer_audio": is_speech,
            "defer_runtime_append": True,
        }

    policy = session.config.overlap_policy
    if not is_speech:
        if session.overlap_speech_ms <= 0:
            session.reset_overlap_speech()
        return {
            "action": "drop",
            "reason": "silence_or_noise",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": False,
        }

    if is_short_ack_transcript_hint(event, payload):
        session.reset_overlap_speech()
        return {
            "action": "listen",
            "reason": "short_ack_transcript",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": False,
            "defer_runtime_append": True,
        }

    if policy == DuplexOverlapPolicy.LISTEN_ONLY.value:
        session.accumulate_overlap_speech(duration_ms)
        return {
            "action": "listen",
            "reason": "policy_listen_only",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": True,
            "defer_runtime_append": True,
        }

    if policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value and not session.capabilities.supports_barge_in:
        return defer_unsupported_barge_in(session, duration_ms=duration_ms, is_speech=True)

    session.accumulate_overlap_speech(duration_ms)
    if policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value:
        speech_started = vad_speech_started(event, payload)
        if speech_started is False:
            return {
                "action": "listen",
                "reason": "server_vad_utterance_active",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
                "defer_runtime_append": False,
                "force_listen": True,
                "preserve_realtime_input": True,
            }
        return {
            "action": "barge_in",
            "reason": ("server_vad_speech_started" if speech_started is True else "policy_barge_in_on_speech"),
            "cancel_reason": "turn_detected" if speech_started is True else "barge_in",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": True,
        }

    if (
        duration_ms <= session.config.overlap_short_ack_ms
        and session.overlap_speech_ms <= session.config.overlap_short_ack_ms
    ):
        return {
            "action": "listen",
            "reason": "short_ack",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": True,
            "defer_runtime_append": True,
        }
    if session.overlap_speech_ms >= session.config.overlap_barge_in_ms:
        if not session.capabilities.supports_barge_in:
            return {
                "action": "listen",
                "reason": "barge_in_unsupported",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
                "defer_runtime_append": True,
            }
        return {
            "action": "barge_in",
            "reason": "long_overlap_speech",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": True,
        }
    return {
        "action": "listen",
        "reason": "accumulating_overlap_speech",
        "duration_ms": duration_ms,
        "overlap_speech_ms": session.overlap_speech_ms,
        "buffer_audio": True,
        "defer_runtime_append": True,
    }


def vad_speech_started(event: Mapping[str, object], payload: Mapping[str, object]) -> bool | None:
    for source in (event, payload):
        vad = source.get("vad")
        if isinstance(vad, Mapping) and isinstance(vad.get("speech_started"), bool):
            return bool(vad["speech_started"])
    return None


def event_requests_barge_in(event: Mapping[str, object]) -> bool:
    if event.get("force_barge_in") is True:
        return True
    explicit = event.get("overlap_action") or event.get("overlap")
    return isinstance(explicit, str) and explicit.strip().lower() in {"barge_in", "interrupt", "cancel"}


def defer_unsupported_barge_in(
    session: DuplexEngineSession,
    *,
    duration_ms: int,
    is_speech: bool,
) -> dict[str, object]:
    if is_speech:
        session.accumulate_overlap_speech(duration_ms)
    return {
        "action": "listen",
        "reason": "barge_in_unsupported",
        "duration_ms": duration_ms,
        "overlap_speech_ms": session.overlap_speech_ms,
        "buffer_audio": is_speech,
        "defer_runtime_append": True,
    }


def is_short_ack_transcript_hint(event: dict[str, object], payload: dict[str, object]) -> bool:
    raw_text = event.get("transcript") or event.get("text") or payload.get("transcript") or payload.get("text")
    if not isinstance(raw_text, str):
        return False
    normalized = raw_text.strip().lower()
    if not normalized:
        return False
    compact = "".join(ch for ch in normalized if ch.isalnum() or "一" <= ch <= "鿿")
    if compact in {
        "嗯",
        "嗯嗯",
        "对",
        "对的",
        "好",
        "好的",
        "继续",
        "继续说",
        "可以",
        "是的",
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "continue",
        "goon",
        "right",
    }:
        return True
    return normalized in {"go on", "keep going", "please continue"}


def input_audio_duration_ms(event: dict[str, object], payload: dict[str, object]) -> int:
    for key in ("duration_ms", "audio_duration_ms"):
        value = event.get(key)
        if isinstance(value, int | float):
            return max(0, int(value))
    fmt = payload.get("format")
    sample_rate_hz = payload.get("sample_rate_hz")
    audio = payload.get("audio")
    if fmt == "pcm_f32le" and isinstance(sample_rate_hz, int) and sample_rate_hz > 0 and isinstance(audio, str):
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return 0
        return int((len(raw) // 4) * 1000 / sample_rate_hz)
    return 0


def merge_audio_payloads(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    if first.get("format") != "pcm_f32le" or second.get("format") != "pcm_f32le":
        return second
    first_rate = first.get("sample_rate_hz")
    second_rate = second.get("sample_rate_hz")
    if not isinstance(first_rate, int) or not isinstance(second_rate, int) or first_rate != second_rate:
        return second
    first_audio = first.get("audio")
    second_audio = second.get("audio")
    if not isinstance(first_audio, str) or not isinstance(second_audio, str):
        return second
    try:
        first_raw = base64.b64decode(first_audio, validate=True)
        second_raw = base64.b64decode(second_audio, validate=True)
    except (binascii.Error, ValueError):
        return second
    merged = dict(second)
    merged["audio"] = base64.b64encode(first_raw + second_raw).decode("ascii")
    merged["sample_rate_hz"] = first_rate
    merged_frames = [
        frame
        for source in (first.get("video_frames"), second.get("video_frames"))
        if isinstance(source, list)
        for frame in source
        if isinstance(frame, str) and frame
    ]
    if merged_frames:
        merged["video_frames"] = merged_frames
    else:
        merged.pop("video_frames", None)
    merged["force_listen"] = bool(first.get("force_listen", False)) or bool(second.get("force_listen", False))
    merged.pop("force_speak", None)
    merged["is_speech"] = bool(first.get("is_speech", False)) or bool(second.get("is_speech", False))
    return merged


def should_force_listen_for_short_commit(
    session: DuplexEngineSession, event: dict[str, object], payload: dict[str, object]
) -> bool:
    if event.get("force_listen") is True or payload.get("force_listen") is True:
        return True
    if event.get("force_barge_in") is True:
        return False
    if event.get("response_create") is not True:
        return False
    duration_ms = input_audio_duration_ms(event, payload)
    return 0 < duration_ms <= session.config.overlap_short_ack_ms


def should_force_listen_for_auto_response_overlap(
    event: dict[str, object], payload: dict[str, object], *, auto_responds: bool
) -> bool:
    if not auto_responds:
        return False
    if event.get("force_barge_in") is True:
        return False
    return event.get("force_listen") is True or payload.get("force_listen") is True


def input_looks_like_speech(session: DuplexEngineSession, event: dict[str, object], payload: dict[str, object]) -> bool:
    for key in ("is_speech", "speech"):
        value = event.get(key)
        if isinstance(value, bool):
            return value
    vad = event.get("vad")
    if isinstance(vad, dict):
        value = vad.get("is_speech")
        if isinstance(value, bool):
            return value
        probability = vad.get("speech_probability", vad.get("probability"))
        if isinstance(probability, int | float):
            return float(probability) >= 0.5
    probability = event.get("speech_probability")
    if isinstance(probability, int | float):
        return float(probability) >= 0.5

    fmt = payload.get("format")
    audio = payload.get("audio")
    if fmt in {"pcm_f32le", "pcm16"} and isinstance(audio, str):
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return True
        if fmt == "pcm_f32le":
            if len(raw) < 4 or len(raw) % 4 != 0:
                return True
            samples = np.frombuffer(raw, dtype=np.float32)
        else:
            if len(raw) < 2 or len(raw) % 2 != 0:
                return True
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
        return rms >= session.config.overlap_silence_rms
    return True
