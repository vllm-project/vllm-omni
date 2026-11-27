# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Portions derived from https://github.com/step-out/Duplex-MPE
# (Copyright (c) 2026 MPeval authors, MIT License; see benchmarks/duplex_mpe/LICENSE).
"""Reactive /v1/realtime WebSocket session driver for Duplex-MPE evaluation on vLLM-Omni."""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import io
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Literal
from typing import Protocol as TypingProtocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import numpy as np
import soundfile
import websockets
from numpy.typing import NDArray
from pydantic import JsonValue

from benchmarks.duplex_mpe.dataset import (
    DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH,
    DuplexMpeScenarioRecord,
    compute_file_sha256,
    resolve_dataset_audio_path,
)
from benchmarks.duplex_mpe.protocol import (
    N4_PAUSE_SECONDS,
    ORDINARY_GAP_SECONDS,
    TARGET_MAX_RESPONSE_SECONDS,
    TARGET_ONSET_DEADLINE_SECONDS,
    ScenarioScheduleResult,
    ScenarioTurnInput,
    compute_protocol_fingerprint,
    iter_scenario_schedule,
    load_mono_waveform_at_rate,
    resolve_scenario_seed,
)
from benchmarks.duplex_mpe.vad import SileroActivityConfig, SileroVadModel

logger = logging.getLogger(__name__)

RealtimeProtocolMode = Literal["native", "turn_based", "openai_turn"]
RealtimeProtocolSelection = Literal["auto", "native", "turn_based", "openai_turn"]
InputAudioEncoding = Literal["pcm16", "pcm_f32le"]
RealtimeWireMode = Literal["duplex", "openai_turn"]

DEFAULT_INPUT_SAMPLE_RATE_HZ = 16000
DEFAULT_OUTPUT_SAMPLE_RATE_HZ = 24000
OPENAI_TURN_SAMPLE_RATE_HZ = 24000
PACKET_DURATION_MILLISECONDS = 80.0
MAX_WEBSOCKET_MESSAGE_BYTES = 16 * 1024 * 1024
ADMISSION_DENIED_STATUS = 503
ADMISSION_RETRIES = 3
ADMISSION_BACKOFF_S = 0.25
WEBSOCKET_SUPPORTS_PROXY: bool = "proxy" in inspect.signature(websockets.connect).parameters
OPENAI_TURN_TAIL_WAIT_SECONDS = 1.0
DEFAULT_SESSION_TIMEOUT_SECONDS = 420.0
DEFAULT_MAX_PACING_LAG_SECONDS = 0.5
HANDSHAKE_TIMEOUT_SECONDS = 190.0
CLOSE_DRAIN_TIMEOUT_SECONDS = 20.0
SCENARIO_DRAIN_MARGIN_SECONDS = 60.0
MODELS_PROBE_MAX_ATTEMPTS = ADMISSION_RETRIES + 1
RETRYABLE_HTTP_STATUS_CODES: frozenset[int] = frozenset({502, 503, 504})
SEED_POLICY_METADATA_ONLY = "reference_seed_metadata_only"
NEUTRAL_NEMOTRON_INSTRUCTIONS = (
    "You are Aria, a voice assistant in a multi-party room. "
    "Follow the spoken duty instructions at the beginning of the audio stream, "
    "and do not speak an unsolicited opening greeting."
)

DEFAULT_TURN_DETECTION: dict[str, JsonValue] = {
    "type": "server_vad",
    "silence_duration_ms": 500,
}

KNOWN_SAFE_ERROR_CODES: frozenset[str] = frozenset(
    {
        "admission_rejected",
        "bad_audio",
        "bad_event",
        "config_timeout",
        "input_backpressure",
        "input_capacity_exceeded",
        "internal_error",
        "invalid_event",
        "invalid_input_modality",
        "invalid_request_error",
        "invalid_session_config",
        "invalid_value",
        "item_not_found",
        "multi_api_duplex_unsupported",
        "output_backpressure",
        "playback_ack_too_late",
        "playback_item_mismatch",
        "playback_item_not_found",
        "rate_limit_exceeded",
        "ref_audio_required",
        "resource_exhausted",
        "response_cancel_not_active",
        "response_not_active",
        "runtime_append_failed",
        "runtime_data_plane_stream_failed",
        "server_error",
        "server_vad_unavailable",
        "session_expired",
        "timeout",
        "unsupported",
        "unsupported_audio_format",
        "unsupported_model",
        "unsupported_nemotron_duplex_mode",
        "unsupported_turn_detection",
    }
)


class ProtocolSessionError(RuntimeError):
    """Sanitized protocol-level session error safe to surface in diagnostics."""


class RetryableSessionAdmissionError(RuntimeError):
    """Raised when a session open is rejected due to retryable capacity exhaustion."""


class MonotonicClock(TypingProtocol):
    def __call__(self) -> float: ...


@dataclass
class AdmissionState:
    denials: int = 0


def reject_json_constant(constant: str) -> float:
    raise ValueError(f"Out of range float values are not JSON compliant: {constant}")


SERVER_EVENT_DECODER = json.JSONDecoder(parse_constant=reject_json_constant)


def resolve_realtime_bearer_headers(api_key_env: str | None) -> dict[str, str]:
    if not api_key_env:
        return {}
    raw_token = os.environ.get(api_key_env)
    if not raw_token:
        raise RuntimeError(f"Realtime API key environment variable is not set: {api_key_env}")
    if any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in raw_token):
        raise ValueError("Realtime API key environment variable contains invalid or whitespace characters")
    return {"Authorization": f"Bearer {raw_token}"}


@dataclass(frozen=True, kw_only=True)
class RealtimeSessionOptions:
    instructions: str | None = None
    voice: str | None = None
    ref_audio_data_url: str | None = None
    temperature: float | None = None


@dataclass(frozen=True, kw_only=True)
class RealtimeEndpointSpec:
    websocket_url: str
    models_url: str
    wire_mode: RealtimeWireMode
    protocol: RealtimeProtocolMode
    input_audio_encoding: InputAudioEncoding
    input_sample_rate_hz: int
    output_sample_rate_hz: int
    turn_detection: dict[str, JsonValue] | None
    reported_model: str | None = None
    requested_model: str | None = None
    max_sessions: int | None = None
    supports_multi_session: bool = True
    supports_barge_in: bool = False
    supports_playback_ack: bool = False
    supports_audio_truncate: bool = False
    supports_client_commit: bool = True
    chunk_period_ms: int | None = None
    requires_auto_response_on_open: bool = False
    session_options: RealtimeSessionOptions = RealtimeSessionOptions()
    api_key_env: str | None = None
    trust_env: bool = False


def encode_audio_file_as_data_url(audio_path: Path) -> str:
    expanded_path = audio_path.expanduser()
    if expanded_path.is_symlink():
        raise ValueError(f"Symlink reference audio path is not permitted: {audio_path}")
    resolved_path = expanded_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
    audio_info = soundfile.info(str(resolved_path))
    fmt_upper = audio_info.format.upper()
    if fmt_upper == "FLAC":
        mime_type = "audio/flac"
    elif fmt_upper == "WAV":
        mime_type = "audio/wav"
    else:
        raise ValueError(f"Unsupported reference audio format {audio_info.format!r}; expected WAV or FLAC")
    encoded_bytes = base64.b64encode(resolved_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_bytes}"


def validate_realtime_base_url(base_url: str) -> None:
    try:
        parsed = urlsplit(base_url)
        allowed_query_keys = {"duplex", "model", "autostart"}
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        is_valid = (
            parsed.scheme in ("http", "https", "ws", "wss")
            and bool(parsed.hostname)
            and (parsed.port is None or 1 <= parsed.port <= 65535)
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
            and all(key in allowed_query_keys for key, _ in query_pairs)
        )
    except ValueError:
        is_valid = False
    if not is_valid:
        raise ValueError(
            "Realtime base_url must be an HTTP(S) or WS(S) URL without userinfo, fragment, or unknown query keys"
        )


def sanitize_realtime_origin(endpoint_url: str) -> str:
    parsed = urlsplit(endpoint_url)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, "", "", ""))


def summarize_transport_exception(exc: BaseException) -> str:
    if isinstance(exc, ProtocolSessionError):
        return str(exc) or type(exc).__name__
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTPStatusError(status={exc.response.status_code})"
    if isinstance(exc, websockets.InvalidStatus):
        return f"InvalidStatus(status={exc.response.status_code})"
    return type(exc).__name__


def extract_error_code_and_retryable(error_event: dict[str, JsonValue]) -> tuple[str | None, bool]:
    error_body = error_event.get("error")
    code: str | None = None
    retryable = error_event.get("retryable") is True
    if isinstance(error_body, dict):
        if error_body.get("retryable") is True:
            retryable = True
        for field_name in ("code", "type"):
            candidate = error_body.get(field_name)
            if isinstance(candidate, str) and candidate:
                code = candidate
                break
    if code is None:
        top_code = error_event.get("code")
        if isinstance(top_code, str) and top_code:
            code = top_code
    return code, retryable


def is_retryable_session_rejection(error_event: dict[str, JsonValue]) -> bool:
    if error_event.get("type") != "error":
        return False
    code, retryable = extract_error_code_and_retryable(error_event)
    return retryable or code in ("resource_exhausted", "admission_rejected")


def format_server_error_event(error_event: dict[str, JsonValue]) -> str:
    code, _ = extract_error_code_and_retryable(error_event)
    if code is not None and code in KNOWN_SAFE_ERROR_CODES:
        return f"Server returned error event (code={code})"
    return "Server returned error event"


def derive_realtime_urls(
    base_url: str,
) -> tuple[str, str, str, Literal["auto", "duplex", "openai_turn"], str | None]:
    """Return (duplex_ws_url, openai_turn_ws_url, models_http_url, url_preference, url_model)."""
    validate_realtime_base_url(base_url)
    parsed = urlsplit(base_url.rstrip("/"))
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"

    ws_scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    http_scheme = "https" if parsed.scheme in ("https", "wss") else "http"
    raw_path = parsed.path.rstrip("/")
    query_map = dict(parse_qsl(parsed.query, keep_blank_values=True))
    duplex_query = query_map.get("duplex", "").strip().lower()

    url_preference: Literal["auto", "duplex", "openai_turn"] = "auto"
    if raw_path.endswith("/v1/duplex"):
        prefix_path = raw_path[: -len("/v1/duplex")]
        duplex_ws_path = f"{prefix_path}/v1/duplex"
        url_preference = "duplex"
    elif raw_path.endswith("/v1/realtime"):
        prefix_path = raw_path[: -len("/v1/realtime")]
        duplex_ws_path = f"{prefix_path}/v1/realtime"
        if duplex_query in ("1", "true", "yes", "on"):
            url_preference = "duplex"
        elif duplex_query in ("0", "false", "no", "off"):
            url_preference = "openai_turn"
    elif raw_path.endswith("/v1"):
        prefix_path = raw_path[: -len("/v1")]
        duplex_ws_path = f"{prefix_path}/v1/realtime"
    else:
        prefix_path = raw_path
        duplex_ws_path = f"{prefix_path}/v1/realtime"

    raw_url_model = query_map.get("model", "").strip()
    url_model: str | None = raw_url_model if raw_url_model else None
    duplex_query_params: dict[str, str] = {"autostart": "0"}
    if not duplex_ws_path.endswith("/v1/duplex"):
        duplex_query_params = {"duplex": "1", "autostart": "0"}
    if url_model is not None:
        duplex_query_params["model"] = url_model
    openai_query_params: dict[str, str] = {"model": url_model} if url_model is not None else {}

    duplex_ws_url = urlunsplit((ws_scheme, host, duplex_ws_path, urlencode(duplex_query_params), ""))
    openai_turn_ws_url = urlunsplit((ws_scheme, host, f"{prefix_path}/v1/realtime", urlencode(openai_query_params), ""))
    models_url = urlunsplit((http_scheme, host, f"{prefix_path}/v1/models", "", ""))
    return duplex_ws_url, openai_turn_ws_url, models_url, url_preference, url_model


def infer_model_audio_defaults(
    model_identifier: str | None,
    *,
    is_native_duplex: bool,
    chunk_period_ms: int | None,
    supports_client_commit: bool = True,
    requires_auto_response_on_open: bool = False,
) -> tuple[InputAudioEncoding, int, int]:
    lowered = (model_identifier or "").lower()
    if "personaplex" in lowered or "haru" in lowered or chunk_period_ms in (133, 134):
        return ("pcm_f32le", 24000, 24000)
    if "nemotron" in lowered or "voicechat" in lowered or requires_auto_response_on_open:
        return ("pcm_f32le", 16000, 22050)
    # PersonaPlex and Haru are the only native duplex plugins with supports_client_commit=False & chunk_period_ms=80!
    if is_native_duplex and not supports_client_commit:
        return ("pcm_f32le", 24000, 24000)
    if is_native_duplex and chunk_period_ms == 80:
        return ("pcm_f32le", 16000, 24000)
    return ("pcm_f32le", DEFAULT_INPUT_SAMPLE_RATE_HZ, DEFAULT_OUTPUT_SAMPLE_RATE_HZ)


def resolve_input_encoding(
    requested_encoding: InputAudioEncoding | None,
    *,
    wire_mode: RealtimeWireMode,
    input_sample_rate_hz: int,
    inferred_encoding: InputAudioEncoding,
) -> InputAudioEncoding:
    if wire_mode == "openai_turn":
        if requested_encoding not in (None, "pcm16"):
            raise ValueError("OpenAI turn-based /v1/realtime endpoints accept only pcm16 input audio")
        return "pcm16"
    if requested_encoding is not None:
        if requested_encoding == "pcm16" and input_sample_rate_hz != 16000:
            raise ValueError(
                f"vLLM-Omni duplex resamples pcm16 input to 16000 Hz; use --input-audio-format pcm_f32le "
                f"for {input_sample_rate_hz} Hz duplex models"
            )
        return requested_encoding
    if input_sample_rate_hz != 16000:
        return "pcm_f32le"
    return inferred_encoding


async def connect_realtime_websocket(
    url: str,
    *,
    admission_state: AdmissionState,
    max_retries: int = ADMISSION_RETRIES,
    backoff_seconds: float = ADMISSION_BACKOFF_S,
    open_timeout_seconds: float = HANDSHAKE_TIMEOUT_SECONDS,
    additional_headers: dict[str, str] | None = None,
    trust_env: bool | None = None,
) -> websockets.ClientConnection | None:
    """Connect to /v1/realtime while charging HTTP 503 denials against admission_state."""
    while True:
        try:
            if trust_env is not None and WEBSOCKET_SUPPORTS_PROXY:
                return await websockets.connect(
                    url,
                    max_size=MAX_WEBSOCKET_MESSAGE_BYTES,
                    compression=None,
                    ping_interval=None,
                    open_timeout=open_timeout_seconds,
                    additional_headers=additional_headers,
                    proxy=True if trust_env else None,
                )
            return await websockets.connect(
                url,
                max_size=MAX_WEBSOCKET_MESSAGE_BYTES,
                compression=None,
                ping_interval=None,
                open_timeout=open_timeout_seconds,
                additional_headers=additional_headers,
            )
        except websockets.InvalidStatus as exc:
            if exc.response.status_code != ADMISSION_DENIED_STATUS:
                raise
            admission_state.denials += 1
            if admission_state.denials > max_retries:
                return None
            await asyncio.sleep(backoff_seconds * admission_state.denials)


async def open_websocket_with_admission_state(
    websocket_url: str,
    *,
    admission_state: AdmissionState,
    open_timeout_seconds: float = HANDSHAKE_TIMEOUT_SECONDS,
    additional_headers: dict[str, str] | None = None,
    trust_env: bool = False,
) -> websockets.ClientConnection:
    origin = sanitize_realtime_origin(websocket_url)
    try:
        connection = await connect_realtime_websocket(
            websocket_url,
            admission_state=admission_state,
            max_retries=ADMISSION_RETRIES,
            backoff_seconds=ADMISSION_BACKOFF_S,
            open_timeout_seconds=open_timeout_seconds,
            additional_headers=additional_headers,
            trust_env=trust_env,
        )
    except websockets.InvalidStatus as exc:
        raise RuntimeError(f"WebSocket handshake at {origin} rejected: {summarize_transport_exception(exc)}") from None
    except (OSError, TimeoutError, asyncio.TimeoutError, websockets.WebSocketException) as exc:
        raise RuntimeError(f"WebSocket connection to {origin} failed: {summarize_transport_exception(exc)}") from None
    if connection is None:
        raise RuntimeError(
            f"WebSocket handshake at {origin} rejected after {ADMISSION_RETRIES} HTTP 503 admission retries"
        )
    return connection


async def query_first_served_model_id(
    models_url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 15.0,
    trust_env: bool = False,
) -> str | None:
    async with httpx.AsyncClient(timeout=timeout_seconds, trust_env=trust_env) as http_client:
        for attempt_index in range(MODELS_PROBE_MAX_ATTEMPTS):
            try:
                response = await http_client.get(models_url, headers=headers)
                if response.status_code in RETRYABLE_HTTP_STATUS_CODES:
                    await asyncio.sleep(ADMISSION_BACKOFF_S * (attempt_index + 1))
                    continue
                if response.status_code != 200:
                    return None
                payload = response.json()
                if not isinstance(payload, dict):
                    return None
                data_rows = payload.get("data")
                if not isinstance(data_rows, list) or not data_rows:
                    return None
                first_row = data_rows[0]
                if isinstance(first_row, dict) and isinstance(first_row.get("id"), str) and first_row["id"]:
                    return str(first_row["id"])
                return None
            except (httpx.HTTPError, ValueError):
                await asyncio.sleep(ADMISSION_BACKOFF_S * (attempt_index + 1))
    return None


def resolve_effective_instructions(
    *,
    model: str | None,
    requires_auto_response_on_open: bool,
    explicit_instructions: str | None,
) -> str | None:
    if explicit_instructions is not None:
        return explicit_instructions
    lowered = (model or "").lower()
    if requires_auto_response_on_open or "nemotron" in lowered or "voicechat" in lowered:
        # Nemotron VoiceChat's server default instructs the model to "Start the conversation by greeting the user",
        # which directly conflicts with Duplex-MPE's spoken duty preamble unless overridden with a neutral prompt.
        return NEUTRAL_NEMOTRON_INSTRUCTIONS
    return None


def build_duplex_session_payload(
    *,
    model: str | None,
    protocol: Literal["native", "turn_based"],
    input_audio_encoding: InputAudioEncoding,
    input_sample_rate_hz: int,
    output_sample_rate_hz: int,
    turn_detection: dict[str, JsonValue] | None,
    supports_barge_in: bool,
    supports_playback_ack: bool,
    requires_auto_response_on_open: bool = False,
    session_options: RealtimeSessionOptions,
) -> dict[str, JsonValue]:
    extra_body: dict[str, JsonValue] = {"silence_continuation": False}
    if protocol == "native":
        extra_body["auto_response"] = True
    # Omit flat `output_audio_format` so `apply_realtime_session_defaults` honors nested `audio.output`
    # (`format: {"type": "audio/pcm", "rate": ...}`, `sample_rate_hz: ...`) while still mapping to `pcm16`.
    payload: dict[str, JsonValue] = {
        "modalities": ["audio", "text"],
        "input_audio_format": input_audio_encoding,
        "sample_rate_hz": input_sample_rate_hz,
        "audio": {
            "input": {
                "format": input_audio_encoding,
                "sample_rate_hz": input_sample_rate_hz,
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": output_sample_rate_hz},
                "sample_rate_hz": output_sample_rate_hz,
            },
        },
        "turn_detection": None if protocol == "native" else turn_detection,
        "extra_body": extra_body,
    }
    if model is not None:
        payload["model"] = model
    if supports_playback_ack:
        payload["playback_commit_policy"] = "ack_only"
    if protocol == "native":
        if supports_barge_in:
            payload["overlap_policy"] = "listen_only"
    elif supports_barge_in:
        payload["overlap_policy"] = "barge_in_on_speech"
    effective_instructions = resolve_effective_instructions(
        model=model,
        requires_auto_response_on_open=requires_auto_response_on_open,
        explicit_instructions=session_options.instructions,
    )
    if effective_instructions is not None:
        payload["instructions"] = effective_instructions
    if session_options.voice is not None:
        payload["voice"] = session_options.voice
    if session_options.ref_audio_data_url is not None and protocol == "native":
        payload["ref_audio"] = session_options.ref_audio_data_url
    if session_options.temperature is not None:
        payload["temperature"] = session_options.temperature
    return payload


def build_openai_turn_session_payload(
    *,
    model: str | None,
    turn_detection: dict[str, JsonValue],
    session_options: RealtimeSessionOptions,
) -> dict[str, JsonValue]:
    if session_options.temperature is not None:
        raise ValueError(
            "vLLM-Omni OpenAI turn-based /v1/realtime sessions do not support --temperature; "
            "omit --temperature or use native/turn_based duplex"
        )
    audio_output: dict[str, JsonValue] = {"format": {"type": "audio/pcm", "rate": OPENAI_TURN_SAMPLE_RATE_HZ}}
    if session_options.voice is not None:
        audio_output["voice"] = session_options.voice
    payload: dict[str, JsonValue] = {
        "type": "realtime",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": OPENAI_TURN_SAMPLE_RATE_HZ},
                "turn_detection": turn_detection,
            },
            "output": audio_output,
        },
    }
    if model is not None:
        payload["model"] = model
    if session_options.instructions is not None:
        payload["instructions"] = session_options.instructions
    return payload


def decode_json_frame(raw_frame: str | bytes, context_label: str) -> dict[str, JsonValue]:
    frame_text = raw_frame if isinstance(raw_frame, str) else raw_frame.decode("utf-8")
    try:
        parsed_event = SERVER_EVENT_DECODER.decode(frame_text)
    except ValueError:
        raise ProtocolSessionError(f"Malformed JSON frame during {context_label}") from None
    if not isinstance(parsed_event, dict) or not isinstance(parsed_event.get("type"), str):
        raise ProtocolSessionError(f"Malformed server event object during {context_label}")
    return parsed_event


@dataclass(frozen=True, kw_only=True)
class DuplexCapabilitySummary:
    protocol_mode: Literal["native", "turn_based"]
    reported_model: str | None
    supports_barge_in: bool
    supports_playback_ack: bool
    supports_audio_truncate: bool
    supports_client_commit: bool
    supports_multi_session: bool
    max_sessions: int | None
    chunk_period_ms: int | None
    requires_auto_response_on_open: bool = False


def parse_duplex_capabilities_dict(
    session_payload: dict[str, JsonValue],
    *,
    requires_auto_response_on_open: bool = False,
) -> DuplexCapabilitySummary:
    raw_model = session_payload.get("model")
    reported_model = raw_model if isinstance(raw_model, str) and raw_model else None
    capabilities = session_payload.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("session.created lacks duplex capabilities dictionary")
    is_native = (
        capabilities.get("implementation_level") == "model_native_duplex"
        or capabilities.get("supports_model_native_turn_policy") is True
    )
    protocol_mode: Literal["native", "turn_based"] = "native" if is_native else "turn_based"
    supports_barge_in = capabilities.get("supports_barge_in") is True
    supports_playback_ack = capabilities.get("supports_playback_ack") is not False
    supports_audio_truncate = capabilities.get("supports_audio_truncate") is True
    supports_client_commit = capabilities.get("supports_client_commit") is not False
    supports_multi_session = capabilities.get("supports_multi_session") is not False
    raw_max_sessions = capabilities.get("max_sessions")
    if type(raw_max_sessions) is int and raw_max_sessions >= 1:
        max_sessions: int | None = int(raw_max_sessions)
    elif not supports_multi_session:
        max_sessions = 1
    else:
        max_sessions = None
    raw_chunk_period = capabilities.get("chunk_period_ms")
    chunk_period_ms = int(raw_chunk_period) if type(raw_chunk_period) is int and raw_chunk_period > 0 else None
    return DuplexCapabilitySummary(
        protocol_mode=protocol_mode,
        reported_model=reported_model,
        supports_barge_in=supports_barge_in,
        supports_playback_ack=supports_playback_ack,
        supports_audio_truncate=supports_audio_truncate,
        supports_client_commit=supports_client_commit,
        supports_multi_session=supports_multi_session,
        max_sessions=max_sessions,
        chunk_period_ms=chunk_period_ms,
        requires_auto_response_on_open=requires_auto_response_on_open,
    )


class ProbeSocket(TypingProtocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...


async def drain_probe_close(websocket: ProbeSocket, close_timeout_seconds: float) -> None:
    deadline = time.perf_counter() + close_timeout_seconds
    await websocket.send(json.dumps({"type": "session.close", "event_id": uuid.uuid4().hex}, allow_nan=False))
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            break
        closing_raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        closing_event = decode_json_frame(closing_raw, "duplex probe close")
        if closing_event.get("type") == "session.closed":
            break


async def probe_duplex_endpoint_once(
    duplex_ws_url: str,
    *,
    requested_model: str | None,
    session_options: RealtimeSessionOptions,
    additional_headers: dict[str, str] | None,
    timeout_seconds: float,
    trust_env: bool,
) -> DuplexCapabilitySummary | None:
    """Open a short probe session on /v1/realtime?duplex=1&autostart=0; return None if duplex is disabled."""
    admission_state = AdmissionState()
    while True:
        connection = await open_websocket_with_admission_state(
            duplex_ws_url,
            admission_state=admission_state,
            open_timeout_seconds=timeout_seconds,
            additional_headers=additional_headers,
            trust_env=trust_env,
        )
        should_retry = False
        async with connection as websocket:
            # Note: Do NOT attach `ref_audio` on the generic probe because turn-based plugins
            # (e.g., Qwen3-Omni) reject `ref_audio` with `invalid_session_config`.
            probe_session: dict[str, JsonValue] = {
                "modalities": ["audio", "text"],
                "extra_body": {"silence_continuation": False},
            }
            if requested_model is not None:
                probe_session["model"] = requested_model
            if session_options.voice is not None:
                probe_session["voice"] = session_options.voice
            if session_options.instructions is not None:
                probe_session["instructions"] = session_options.instructions

            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": uuid.uuid4().hex,
                        "session": probe_session,
                    },
                    allow_nan=False,
                )
            )
            raw_frame = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
            first_event = decode_json_frame(raw_frame, "duplex capability probe")
            event_type = first_event.get("type")
            if event_type == "error":
                code, _ = extract_error_code_and_retryable(first_event)
                if code == "unsupported":
                    return None
                if is_retryable_session_rejection(first_event):
                    admission_state.denials += 1
                    if admission_state.denials > ADMISSION_RETRIES:
                        raise RuntimeError(
                            f"Duplex capability probe rejected after {admission_state.denials} "
                            f"resource_exhausted retries"
                        )
                    should_retry = True
                elif code == "unsupported_nemotron_duplex_mode":
                    should_retry = False
                else:
                    raise ProtocolSessionError(format_server_error_event(first_event))
            elif event_type == "session.created":
                session_obj = first_event.get("session")
                if not isinstance(session_obj, dict) or not isinstance(session_obj.get("capabilities"), dict):
                    return None
                parsed_caps = parse_duplex_capabilities_dict(session_obj, requires_auto_response_on_open=False)
                try:
                    await drain_probe_close(websocket, min(5.0, timeout_seconds))
                except (OSError, TimeoutError, asyncio.TimeoutError, websockets.WebSocketException):
                    logger.warning("Duplex capability probe close timed out; pausing briefly for server lease cleanup")
                    await asyncio.sleep(0.5)
                return parsed_caps
            else:
                raise ProtocolSessionError(f"Unexpected event {event_type!r} during duplex capability probe")

        if should_retry:
            await asyncio.sleep(ADMISSION_BACKOFF_S * admission_state.denials)
            continue

        # Retry once with auto_response=True for plugins (such as Nemotron VoiceChat) that refuse
        # a bare session.update without extra_body.auto_response=True.
        retry_connection = await open_websocket_with_admission_state(
            duplex_ws_url,
            admission_state=admission_state,
            open_timeout_seconds=timeout_seconds,
            additional_headers=additional_headers,
            trust_env=trust_env,
        )
        async with retry_connection as websocket:
            auto_probe_session: dict[str, JsonValue] = {
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm_f32le",
                "sample_rate_hz": 16000,
                "extra_body": {"auto_response": True, "silence_continuation": False},
            }
            if requested_model is not None:
                auto_probe_session["model"] = requested_model
            if session_options.voice is not None:
                auto_probe_session["voice"] = session_options.voice
            if session_options.instructions is not None:
                auto_probe_session["instructions"] = session_options.instructions
            if session_options.ref_audio_data_url is not None:
                auto_probe_session["ref_audio"] = session_options.ref_audio_data_url
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": uuid.uuid4().hex,
                        "session": auto_probe_session,
                    },
                    allow_nan=False,
                )
            )
            raw_frame = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
            second_event = decode_json_frame(raw_frame, "duplex auto_response capability probe")
            if second_event.get("type") == "error":
                if is_retryable_session_rejection(second_event):
                    admission_state.denials += 1
                    if admission_state.denials <= ADMISSION_RETRIES:
                        await asyncio.sleep(ADMISSION_BACKOFF_S * admission_state.denials)
                        continue
                raise ProtocolSessionError(format_server_error_event(second_event))
            session_obj = second_event.get("session")
            if second_event.get("type") != "session.created" or not isinstance(session_obj, dict):
                raise ProtocolSessionError("Expected session.created during duplex auto_response probe")
            parsed_caps = parse_duplex_capabilities_dict(session_obj, requires_auto_response_on_open=True)
            try:
                await drain_probe_close(websocket, min(5.0, timeout_seconds))
            except (OSError, TimeoutError, asyncio.TimeoutError, websockets.WebSocketException):
                logger.warning("Duplex auto_response probe close timed out; pausing briefly for server lease cleanup")
                await asyncio.sleep(0.5)
            return parsed_caps


def append_model_query_param(websocket_url: str, model_id: str | None) -> str:
    if not model_id:
        return websocket_url
    parsed = urlsplit(websocket_url)
    query_pairs = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != "model"]
    query_pairs.append(("model", model_id))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query_pairs), parsed.fragment))


async def probe_openai_turn_endpoint_once(
    openai_turn_ws_url: str,
    *,
    requested_model: str | None,
    additional_headers: dict[str, str] | None,
    timeout_seconds: float,
    trust_env: bool,
) -> str | None:
    target_ws_url = append_model_query_param(openai_turn_ws_url, requested_model)
    admission_state = AdmissionState()
    connection = await open_websocket_with_admission_state(
        target_ws_url,
        admission_state=admission_state,
        open_timeout_seconds=timeout_seconds,
        additional_headers=additional_headers,
        trust_env=trust_env,
    )
    async with connection as websocket:
        raw_frame = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
        first_event = decode_json_frame(raw_frame, "OpenAI realtime probe")
        if first_event.get("type") == "error":
            raise ProtocolSessionError(format_server_error_event(first_event))
        session_obj = first_event.get("session")
        if first_event.get("type") != "session.created" or not isinstance(session_obj, dict):
            raise ProtocolSessionError("Expected session.created from /v1/realtime endpoint")
        raw_model = session_obj.get("model")
        reported_model = raw_model if isinstance(raw_model, str) and raw_model else None
        if requested_model is not None and reported_model is not None and reported_model != requested_model:
            raise ValueError(
                f"OpenAI /v1/realtime endpoint serves model {reported_model!r}, "
                f"which conflicts with requested model {requested_model!r}"
            )
        return reported_model


async def negotiate_realtime_endpoint(
    base_url: str,
    *,
    protocol_selection: RealtimeProtocolSelection = "auto",
    requested_model: str | None = None,
    input_audio_encoding: InputAudioEncoding | None = None,
    input_sample_rate_hz: int | None = None,
    output_sample_rate_hz: int | None = None,
    turn_detection: dict[str, JsonValue] | None = None,
    session_options: RealtimeSessionOptions | None = None,
    api_key_env: str | None = None,
    timeout_seconds: float = HANDSHAKE_TIMEOUT_SECONDS,
    trust_env: bool = False,
) -> RealtimeEndpointSpec:
    """Negotiate vLLM-Omni `/v1/realtime?duplex=1` or turn-based `/v1/realtime` session parameters."""
    effective_options = session_options if session_options is not None else RealtimeSessionOptions()
    auth_headers = resolve_realtime_bearer_headers(api_key_env)
    duplex_ws_url, openai_turn_ws_url, models_url, url_preference, url_model = derive_realtime_urls(base_url)
    origin = sanitize_realtime_origin(models_url)
    if url_model is not None and requested_model is not None and url_model != requested_model:
        raise ValueError(f"URL query model={url_model!r} conflicts with --model={requested_model!r}")

    explicit_model = requested_model or url_model
    discovered_model = explicit_model
    if discovered_model is None:
        discovered_model = await query_first_served_model_id(
            models_url,
            headers=auth_headers or None,
            timeout_seconds=min(15.0, timeout_seconds),
            trust_env=trust_env,
        )

    if protocol_selection == "openai_turn" or url_preference == "openai_turn":
        if protocol_selection in ("native", "turn_based"):
            raise ValueError(
                f"URL {origin} explicitly selects duplex=0, which conflicts with --protocol={protocol_selection}"
            )
        if turn_detection is not None and not isinstance(turn_detection, dict):
            raise ValueError("turn_detection must be a JSON object when provided")
        if effective_options.ref_audio_data_url is not None:
            raise ValueError("OpenAI turn-based /v1/realtime endpoints do not accept --ref-audio")
        if effective_options.temperature is not None:
            raise ValueError(
                "vLLM-Omni OpenAI turn-based /v1/realtime sessions do not support --temperature; "
                "omit --temperature or use native/turn_based duplex"
            )
        reported_model = await probe_openai_turn_endpoint_once(
            openai_turn_ws_url,
            requested_model=explicit_model,
            additional_headers=auth_headers or None,
            timeout_seconds=timeout_seconds,
            trust_env=trust_env,
        )
        effective_input_rate = input_sample_rate_hz or OPENAI_TURN_SAMPLE_RATE_HZ
        effective_output_rate = output_sample_rate_hz or OPENAI_TURN_SAMPLE_RATE_HZ
        if effective_input_rate != OPENAI_TURN_SAMPLE_RATE_HZ or effective_output_rate != OPENAI_TURN_SAMPLE_RATE_HZ:
            raise ValueError(
                f"vLLM-Omni OpenAI turn-based /v1/realtime requires 24000 Hz PCM16 "
                f"(got input={effective_input_rate}, output={effective_output_rate})"
            )
        resolved_encoding = resolve_input_encoding(
            input_audio_encoding,
            wire_mode="openai_turn",
            input_sample_rate_hz=effective_input_rate,
            inferred_encoding="pcm16",
        )
        resolved_openai_model = explicit_model or reported_model or discovered_model
        return RealtimeEndpointSpec(
            websocket_url=append_model_query_param(openai_turn_ws_url, explicit_model),
            models_url=models_url,
            wire_mode="openai_turn",
            protocol="openai_turn",
            input_audio_encoding=resolved_encoding,
            input_sample_rate_hz=effective_input_rate,
            output_sample_rate_hz=effective_output_rate,
            turn_detection=dict(turn_detection) if turn_detection is not None else dict(DEFAULT_TURN_DETECTION),
            reported_model=resolved_openai_model,
            requested_model=explicit_model,
            supports_barge_in=True,
            supports_playback_ack=False,
            supports_audio_truncate=True,
            supports_client_commit=True,
            session_options=effective_options,
            api_key_env=api_key_env,
            trust_env=trust_env,
        )

    duplex_probe = await probe_duplex_endpoint_once(
        duplex_ws_url,
        requested_model=discovered_model,
        session_options=effective_options,
        additional_headers=auth_headers or None,
        timeout_seconds=timeout_seconds,
        trust_env=trust_env,
    )
    if duplex_probe is None:
        if url_preference == "duplex" or protocol_selection in ("native", "turn_based"):
            raise RuntimeError(
                f"Duplex API is not enabled on {origin}; deploy with session_mode: duplex or use --protocol openai_turn"
            )
        if effective_options.ref_audio_data_url is not None:
            raise ValueError("OpenAI turn-based /v1/realtime endpoints do not accept --ref-audio")
        if effective_options.temperature is not None:
            raise ValueError(
                "vLLM-Omni OpenAI turn-based /v1/realtime sessions do not support --temperature; "
                "omit --temperature or use native/turn_based duplex"
            )
        reported_model = await probe_openai_turn_endpoint_once(
            openai_turn_ws_url,
            requested_model=explicit_model,
            additional_headers=auth_headers or None,
            timeout_seconds=timeout_seconds,
            trust_env=trust_env,
        )
        effective_input_rate = input_sample_rate_hz or OPENAI_TURN_SAMPLE_RATE_HZ
        effective_output_rate = output_sample_rate_hz or OPENAI_TURN_SAMPLE_RATE_HZ
        if effective_input_rate != OPENAI_TURN_SAMPLE_RATE_HZ or effective_output_rate != OPENAI_TURN_SAMPLE_RATE_HZ:
            raise ValueError(
                f"vLLM-Omni OpenAI turn-based /v1/realtime requires 24000 Hz PCM16 "
                f"(got input={effective_input_rate}, output={effective_output_rate})"
            )
        resolved_encoding = resolve_input_encoding(
            input_audio_encoding,
            wire_mode="openai_turn",
            input_sample_rate_hz=effective_input_rate,
            inferred_encoding="pcm16",
        )
        resolved_openai_model = explicit_model or reported_model or discovered_model
        return RealtimeEndpointSpec(
            websocket_url=append_model_query_param(openai_turn_ws_url, explicit_model),
            models_url=models_url,
            wire_mode="openai_turn",
            protocol="openai_turn",
            input_audio_encoding=resolved_encoding,
            input_sample_rate_hz=effective_input_rate,
            output_sample_rate_hz=effective_output_rate,
            turn_detection=dict(turn_detection) if turn_detection is not None else dict(DEFAULT_TURN_DETECTION),
            reported_model=resolved_openai_model,
            requested_model=explicit_model,
            supports_barge_in=True,
            supports_playback_ack=False,
            supports_audio_truncate=True,
            supports_client_commit=True,
            session_options=effective_options,
            api_key_env=api_key_env,
            trust_env=trust_env,
        )

    detected_protocol = duplex_probe.protocol_mode
    effective_model_name = duplex_probe.reported_model or discovered_model

    if protocol_selection != "auto" and protocol_selection != detected_protocol:
        raise ValueError(
            f"Requested --protocol={protocol_selection!r} conflicts with server duplex capability "
            f"{detected_protocol!r} at {origin}"
        )
    if detected_protocol != "native" and effective_options.ref_audio_data_url is not None:
        raise ValueError("Turn-based duplex plugins do not accept --ref-audio; use --voice or native duplex")

    inferred_encoding, inferred_input_rate, inferred_output_rate = infer_model_audio_defaults(
        effective_model_name,
        is_native_duplex=(detected_protocol == "native"),
        chunk_period_ms=duplex_probe.chunk_period_ms,
        supports_client_commit=duplex_probe.supports_client_commit,
        requires_auto_response_on_open=duplex_probe.requires_auto_response_on_open,
    )
    resolved_input_rate = input_sample_rate_hz if input_sample_rate_hz is not None else inferred_input_rate
    resolved_output_rate = output_sample_rate_hz if output_sample_rate_hz is not None else inferred_output_rate
    if resolved_input_rate <= 0 or resolved_output_rate <= 0:
        raise ValueError("Resolved input and output sample rates must be positive")

    resolved_encoding = resolve_input_encoding(
        input_audio_encoding,
        wire_mode="duplex",
        input_sample_rate_hz=resolved_input_rate,
        inferred_encoding=inferred_encoding,
    )
    resolved_turn_detection: dict[str, JsonValue] | None
    if detected_protocol == "native":
        if turn_detection is not None:
            raise ValueError("Native full-duplex endpoints reject turn_detection")
        resolved_turn_detection = None
    else:
        resolved_turn_detection = dict(turn_detection) if turn_detection is not None else dict(DEFAULT_TURN_DETECTION)

    return RealtimeEndpointSpec(
        websocket_url=duplex_ws_url,
        models_url=models_url,
        wire_mode="duplex",
        protocol=detected_protocol,
        input_audio_encoding=resolved_encoding,
        input_sample_rate_hz=resolved_input_rate,
        output_sample_rate_hz=resolved_output_rate,
        turn_detection=resolved_turn_detection,
        reported_model=effective_model_name,
        requested_model=explicit_model,
        max_sessions=duplex_probe.max_sessions,
        supports_multi_session=duplex_probe.supports_multi_session,
        supports_barge_in=duplex_probe.supports_barge_in,
        supports_playback_ack=duplex_probe.supports_playback_ack,
        supports_audio_truncate=duplex_probe.supports_audio_truncate,
        supports_client_commit=duplex_probe.supports_client_commit,
        chunk_period_ms=duplex_probe.chunk_period_ms,
        requires_auto_response_on_open=duplex_probe.requires_auto_response_on_open,
        session_options=effective_options,
        api_key_env=api_key_env,
        trust_env=trust_env,
    )


@dataclass(frozen=True, kw_only=True)
class QueuedAudioSegment:
    start_sample_index: int
    samples: NDArray[np.float32]
    response_id: str | None = None
    item_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class PlaybackTruncationRecord:
    item_id: str
    audio_end_milliseconds: int


@dataclass(frozen=True, kw_only=True)
class PlaybackAckRecord:
    response_id: str
    item_id: str
    played_milliseconds: int


class PlaybackTimelineBuffer:
    """Monotonic sample-accurate playout timeline with playback.ack and barge-in truncation."""

    def __init__(self, sample_rate_hz: int) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.sample_rate_hz: int = int(sample_rate_hz)
        self.queued_segments: list[QueuedAudioSegment] = []
        self.render_cursor_samples: int = 0
        self.tail_samples: int = 0
        self.cancelled_unplayed_samples: int = 0
        self.played_samples_by_item_id: dict[str, int] = {}
        self.acked_milliseconds_by_item_id: dict[str, int] = {}
        self.response_id_by_item_id: dict[str, str] = {}
        self.items_with_audio: set[str] = set()
        self.truncated_item_ids: set[str] = set()

    def append_audio(
        self,
        arrival_seconds: float,
        pcm_samples: NDArray[np.float32],
        *,
        response_id: str | None = None,
        item_id: str | None = None,
    ) -> None:
        flat_samples = np.asarray(pcm_samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(flat_samples).all():
            raise ValueError("Received non-finite PCM audio samples from server")
        if len(flat_samples) == 0:
            return
        if item_id is not None and item_id in self.truncated_item_ids:
            self.cancelled_unplayed_samples += len(flat_samples)
            return
        if item_id is not None:
            self.items_with_audio.add(item_id)
            self.played_samples_by_item_id.setdefault(item_id, 0)
            if response_id is not None:
                self.response_id_by_item_id[item_id] = response_id
        arrival_sample_index = int(round(arrival_seconds * self.sample_rate_hz))
        start_sample_index = max(self.render_cursor_samples, self.tail_samples, arrival_sample_index)
        self.queued_segments.append(
            QueuedAudioSegment(
                start_sample_index=start_sample_index,
                samples=flat_samples,
                response_id=response_id,
                item_id=item_id,
            )
        )
        self.tail_samples = start_sample_index + len(flat_samples)

    def cancel_unplayed(
        self,
        cutoff_seconds: float,
        *,
        target_response_id: str | None = None,
        target_item_id: str | None = None,
    ) -> list[PlaybackTruncationRecord]:
        cutoff_sample_index = max(
            self.render_cursor_samples,
            int(round(cutoff_seconds * self.sample_rate_hz)),
        )
        retained_segments: list[QueuedAudioSegment] = []
        affected_item_ids: list[str] = []
        dropped_samples_by_item_id: dict[str, int] = {}
        kept_ahead_samples_by_item_id: dict[str, int] = {}
        for segment in self.queued_segments:
            if target_response_id is None and target_item_id is None:
                matches_target = True
            elif target_response_id is not None and target_item_id is not None:
                matches_target = (segment.response_id == target_response_id) and (
                    segment.item_id is None or segment.item_id == target_item_id
                )
            elif target_response_id is not None:
                matches_target = segment.response_id == target_response_id
            else:
                matches_target = segment.item_id == target_item_id
            if not matches_target:
                retained_segments.append(segment)
                continue
            keep_count = max(0, min(len(segment.samples), cutoff_sample_index - segment.start_sample_index))
            dropped_count = len(segment.samples) - keep_count
            self.cancelled_unplayed_samples += dropped_count
            if segment.item_id is not None:
                if segment.item_id not in affected_item_ids:
                    affected_item_ids.append(segment.item_id)
                dropped_samples_by_item_id[segment.item_id] = (
                    dropped_samples_by_item_id.get(segment.item_id, 0) + dropped_count
                )
                if keep_count > 0:
                    kept_ahead_samples_by_item_id[segment.item_id] = (
                        kept_ahead_samples_by_item_id.get(segment.item_id, 0) + keep_count
                    )
            if keep_count > 0:
                retained_segments.append(
                    QueuedAudioSegment(
                        start_sample_index=segment.start_sample_index,
                        samples=segment.samples[:keep_count],
                        response_id=segment.response_id,
                        item_id=segment.item_id,
                    )
                )
        # vLLM-Omni's `resolve_truncate_item` rejects `conversation.item.truncate` with `bad_event`
        # if the item never produced audio; only emit truncations for items in `self.items_with_audio`.
        if (
            target_item_id is not None
            and target_item_id not in affected_item_ids
            and target_item_id in self.items_with_audio
        ):
            affected_item_ids.append(target_item_id)
        self.queued_segments = retained_segments
        self.tail_samples = max(
            [self.render_cursor_samples]
            + [segment.start_sample_index + len(segment.samples) for segment in retained_segments]
        )
        truncation_records: list[PlaybackTruncationRecord] = []
        for item_id in affected_item_ids:
            if item_id in self.truncated_item_ids or item_id not in self.items_with_audio:
                continue
            played_samples = self.played_samples_by_item_id.get(item_id, 0) + kept_ahead_samples_by_item_id.get(
                item_id, 0
            )
            had_dropped_samples = dropped_samples_by_item_id.get(item_id, 0) > 0
            if had_dropped_samples or target_item_id == item_id:
                self.truncated_item_ids.add(item_id)
                audio_end_milliseconds = max(0, int((played_samples * 1000) // self.sample_rate_hz))
                truncation_records.append(
                    PlaybackTruncationRecord(
                        item_id=item_id,
                        audio_end_milliseconds=audio_end_milliseconds,
                    )
                )
        return truncation_records

    def render_interval(
        self,
        start_sample_index: int,
        end_sample_index: int,
    ) -> tuple[NDArray[np.float32], list[PlaybackAckRecord]]:
        if start_sample_index != self.render_cursor_samples or end_sample_index < start_sample_index:
            raise ValueError("PlaybackTimelineBuffer render must advance monotonically")
        rendered = np.zeros(end_sample_index - start_sample_index, dtype=np.float32)
        remaining_segments: list[QueuedAudioSegment] = []
        advanced_item_ids: list[str] = []
        for segment in self.queued_segments:
            segment_offset = segment.start_sample_index
            segment_samples = segment.samples
            overlap_start = max(start_sample_index, segment_offset)
            overlap_end = min(end_sample_index, segment_offset + len(segment_samples))
            if overlap_end > overlap_start:
                overlap_count = overlap_end - overlap_start
                rendered[overlap_start - start_sample_index : overlap_end - start_sample_index] = segment_samples[
                    overlap_start - segment_offset : overlap_end - segment_offset
                ]
                if segment.item_id is not None:
                    self.played_samples_by_item_id[segment.item_id] = (
                        self.played_samples_by_item_id.get(segment.item_id, 0) + overlap_count
                    )
                    if segment.item_id not in advanced_item_ids:
                        advanced_item_ids.append(segment.item_id)
            if segment_offset + len(segment_samples) > end_sample_index:
                if overlap_end > segment_offset:
                    remaining_segments.append(
                        QueuedAudioSegment(
                            start_sample_index=overlap_end,
                            samples=segment_samples[overlap_end - segment_offset :],
                            response_id=segment.response_id,
                            item_id=segment.item_id,
                        )
                    )
                else:
                    remaining_segments.append(segment)
        self.queued_segments = remaining_segments
        self.render_cursor_samples = end_sample_index

        ack_records: list[PlaybackAckRecord] = []
        for item_id in advanced_item_ids:
            if item_id in self.truncated_item_ids:
                continue
            response_id = self.response_id_by_item_id.get(item_id)
            if response_id is None:
                continue
            played_ms = int((self.played_samples_by_item_id[item_id] * 1000) // self.sample_rate_hz)
            if played_ms > self.acked_milliseconds_by_item_id.get(item_id, 0):
                ack_records.append(
                    PlaybackAckRecord(
                        response_id=response_id,
                        item_id=item_id,
                        played_milliseconds=played_ms,
                    )
                )
        return rendered, ack_records

    def mark_ack_sent(self, item_id: str, played_milliseconds: int) -> None:
        if played_milliseconds > self.acked_milliseconds_by_item_id.get(item_id, 0):
            self.acked_milliseconds_by_item_id[item_id] = played_milliseconds


def encode_float32_to_pcm16_base64(samples: NDArray[np.float32]) -> str:
    flat_samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.isfinite(flat_samples).all():
        raise ValueError("Input audio samples must be finite")
    clipped = np.clip(np.round(flat_samples * 32768.0), -32768.0, 32767.0).astype("<i2")
    return base64.b64encode(clipped.tobytes()).decode("ascii")


def encode_float32_to_f32le_base64(samples: NDArray[np.float32]) -> str:
    flat_samples = np.ascontiguousarray(samples, dtype="<f4").reshape(-1)
    if not np.isfinite(flat_samples).all():
        raise ValueError("Input audio samples must be finite")
    return base64.b64encode(flat_samples.tobytes()).decode("ascii")


def encode_input_audio_chunk(samples: NDArray[np.float32], encoding: InputAudioEncoding) -> str:
    if encoding == "pcm_f32le":
        return encode_float32_to_f32le_base64(samples)
    return encode_float32_to_pcm16_base64(samples)


class PolyphaseChannelResampler:
    """Chunk-partition-invariant Kaiser-windowed polyphase FIR streaming resampler."""

    _HALF_FILTER_WIDTH = 10
    _KAISER_BETA = 5.0
    _MAX_POLYPHASE_FACTOR = 2048

    def __init__(self, source_rate_hz: int, target_rate_hz: int) -> None:
        if source_rate_hz <= 0 or target_rate_hz <= 0:
            raise ValueError("Audio sample rates must be positive")
        self.source_rate_hz: int = int(source_rate_hz)
        self.target_rate_hz: int = int(target_rate_hz)
        rate_gcd = gcd(self.source_rate_hz, self.target_rate_hz)
        self.up: int = self.target_rate_hz // rate_gcd
        self.down: int = self.source_rate_hz // rate_gcd
        if max(self.up, self.down) > self._MAX_POLYPHASE_FACTOR:
            raise ValueError(f"Unsupported wire audio resampling ratio {self.source_rate_hz} -> {self.target_rate_hz}")
        self.half_len, self.phase_kernels = self._design_polyphase_filter()
        self.history_len: int = int(self.phase_kernels.shape[1] - 1)
        self.history: NDArray[np.float32] = np.zeros(self.history_len, dtype=np.float32)
        self.input_samples_seen: int = 0
        self.output_samples_emitted: int = 0

    def _design_polyphase_filter(self) -> tuple[int, NDArray[np.float32]]:
        if self.up == self.down:
            return 0, np.ones((1, 1), dtype=np.float32)
        max_rate = max(self.up, self.down)
        half_len = self._HALF_FILTER_WIDTH * max_rate
        offsets = np.arange(-half_len, half_len + 1, dtype=np.float64)
        cutoff = 1.0 / max_rate
        taps = cutoff * np.sinc(cutoff * offsets)
        taps *= np.kaiser(taps.size, self._KAISER_BETA)
        taps /= np.sum(taps)
        taps *= self.up
        taps_f32 = np.ascontiguousarray(taps, dtype=np.float32)
        phases = [taps_f32[phase :: self.up] for phase in range(self.up)]
        phase_width = max(phase.size for phase in phases)
        kernels = np.zeros((self.up, phase_width), dtype=np.float32)
        for phase_idx, phase in enumerate(phases):
            kernels[phase_idx, -phase.size :] = phase[::-1]
        return half_len, kernels

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        return -(-numerator // denominator)

    def _render_outputs(
        self,
        combined: NDArray[np.float32],
        *,
        first_output: int,
        output_count: int,
        input_start: int,
    ) -> NDArray[np.float32]:
        if output_count <= 0:
            return np.zeros(0, dtype=np.float32)
        output_indexes = np.arange(first_output, first_output + output_count, dtype=np.int64)
        filter_indexes = output_indexes * self.down + self.half_len
        source_indexes = filter_indexes // self.up
        phases = filter_indexes % self.up
        windows = np.lib.stride_tricks.sliding_window_view(combined, self.phase_kernels.shape[1])
        selected_windows = windows[source_indexes - input_start]
        summed = np.sum(selected_windows * self.phase_kernels[phases], axis=1, dtype=np.float32)
        return np.asarray(summed, dtype=np.float32).reshape(-1)

    def push(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        flat = np.ascontiguousarray(chunk, dtype=np.float32).reshape(-1)
        if flat.size == 0 or self.up == self.down:
            return flat
        combined = np.concatenate((self.history, flat))
        next_input_samples = self.input_samples_seen + flat.size
        stable_output_samples = max(
            0,
            self._ceil_div(next_input_samples * self.up - self.half_len, self.down),
        )
        output = self._render_outputs(
            combined,
            first_output=self.output_samples_emitted,
            output_count=stable_output_samples - self.output_samples_emitted,
            input_start=self.input_samples_seen,
        )
        if self.history_len > 0:
            self.history = combined[-self.history_len :].copy()
        self.input_samples_seen = next_input_samples
        self.output_samples_emitted = stable_output_samples
        return output

    def flush(self) -> NDArray[np.float32]:
        if self.up == self.down or self.input_samples_seen == 0:
            return np.zeros(0, dtype=np.float32)
        total_output_samples = self._ceil_div(self.input_samples_seen * self.up, self.down)
        output_count = total_output_samples - self.output_samples_emitted
        if output_count > 0:
            last_output = total_output_samples - 1
            last_source_index = (last_output * self.down + self.half_len) // self.up
            right_padding = max(0, last_source_index - self.input_samples_seen + 1)
            combined = np.concatenate((self.history, np.zeros(right_padding, dtype=np.float32)))
            output = self._render_outputs(
                combined,
                first_output=self.output_samples_emitted,
                output_count=output_count,
                input_start=self.input_samples_seen,
            )
        else:
            output = np.zeros(0, dtype=np.float32)
        self.history = np.zeros(self.history_len, dtype=np.float32)
        self.input_samples_seen = 0
        self.output_samples_emitted = 0
        return output


class StatefulWireResampler:
    """Per-response stateful polyphase wire resampler with exact chunk-partition invariance."""

    def __init__(self, target_sample_rate_hz: int) -> None:
        self.target_sample_rate_hz: int = int(target_sample_rate_hz)
        self._channels: dict[str, PolyphaseChannelResampler] = {}
        self._stream_source_rates_hz: dict[str, int] = {}

    def push(
        self,
        samples: NDArray[np.float32],
        source_sample_rate_hz: int,
        *,
        stream_key: str = "default",
    ) -> NDArray[np.float32]:
        flat = np.asarray(samples, dtype=np.float32).reshape(-1)
        if source_sample_rate_hz <= 0:
            raise ProtocolSessionError("source_sample_rate_hz must be positive")
        recorded_rate = self._stream_source_rates_hz.get(stream_key)
        if recorded_rate is not None and recorded_rate != source_sample_rate_hz:
            raise ProtocolSessionError(
                f"Output audio stream {stream_key!r} changed sample rate mid-response "
                f"({recorded_rate} Hz -> {source_sample_rate_hz} Hz)"
            )
        if flat.size == 0:
            return flat
        self._stream_source_rates_hz[stream_key] = source_sample_rate_hz
        if source_sample_rate_hz == self.target_sample_rate_hz:
            return flat
        channel = self._channels.get(stream_key)
        if channel is None:
            try:
                channel = PolyphaseChannelResampler(source_sample_rate_hz, self.target_sample_rate_hz)
            except ValueError as exc:
                raise ProtocolSessionError(str(exc)) from None
            self._channels[stream_key] = channel
        return channel.push(flat)

    def flush(self, *, stream_key: str = "default") -> NDArray[np.float32]:
        self._stream_source_rates_hz.pop(stream_key, None)
        channel = self._channels.pop(stream_key, None)
        if channel is None:
            return np.zeros(0, dtype=np.float32)
        return channel.flush()

    def discard(self, *, stream_key: str = "default") -> None:
        self._stream_source_rates_hz.pop(stream_key, None)
        self._channels.pop(stream_key, None)

    def discard_all(self) -> None:
        self._stream_source_rates_hz.clear()
        self._channels.clear()


def decode_output_audio_delta_to_float32(
    encoded_audio: str,
    *,
    declared_format: str | None = None,
) -> tuple[NDArray[np.float32], int | None]:
    """Decode `response.output_audio.delta` supporting `pcm16`, `pcm_f32le`, and `wav` containers."""
    try:
        raw_bytes = base64.b64decode(encoded_audio, validate=True)
    except (binascii.Error, ValueError):
        raise ProtocolSessionError("Invalid base64 audio delta from server") from None
    if len(raw_bytes) == 0:
        return np.zeros(0, dtype=np.float32), None
    fmt_normalized = (declared_format or "pcm16").strip().lower()
    if fmt_normalized in ("audio/pcm", "pcm", "pcm16", "pcm_s16le", "s16le"):
        if len(raw_bytes) >= 44 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
            fmt_normalized = "wav"
        else:
            if len(raw_bytes) % 2 != 0:
                raise ProtocolSessionError("Server PCM16 delta has an odd byte length")
            return np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0, None
    if fmt_normalized in ("pcm_f32le", "f32le", "audio/pcm_f32le"):
        if len(raw_bytes) % 4 != 0:
            raise ProtocolSessionError("Server pcm_f32le delta byte length is not a multiple of 4")
        return np.frombuffer(raw_bytes, dtype="<f4").astype(np.float32, copy=True), None
    if fmt_normalized in ("wav", "audio/wav", "audio/x-wav"):
        try:
            decoded, wav_rate = soundfile.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
        except (OSError, RuntimeError, ValueError):
            raise ProtocolSessionError("Malformed or unsupported WAV container in audio delta") from None
        mono_samples = decoded[:, 0] if decoded.shape[1] == 1 else decoded.mean(axis=1)
        return np.ascontiguousarray(mono_samples, dtype=np.float32), int(wav_rate)
    raise ProtocolSessionError(f"Unsupported output audio delta format: {declared_format!r}")


def extract_turn_detection_from_session(session_payload: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    top_turn = session_payload.get("turn_detection")
    if isinstance(top_turn, dict):
        return top_turn
    audio_block = session_payload.get("audio")
    if isinstance(audio_block, dict):
        input_block = audio_block.get("input")
        if isinstance(input_block, dict):
            nested_turn = input_block.get("turn_detection")
            if isinstance(nested_turn, dict):
                return nested_turn
    return None


def validate_accepted_session(
    session_event: dict[str, JsonValue],
    endpoint_spec: RealtimeEndpointSpec,
) -> None:
    """Verify session.created / session.updated echoed the requested turn_detection and protocol."""
    session_payload = session_event.get("session")
    if not isinstance(session_payload, dict):
        raise RuntimeError("Session confirmation event lacks a session object")

    if endpoint_spec.requested_model is not None:
        echoed_model = session_payload.get("model")
        if isinstance(echoed_model, str) and echoed_model and echoed_model != endpoint_spec.requested_model:
            raise RuntimeError(
                f"Session echoed unexpected model={echoed_model!r} (expected {endpoint_spec.requested_model!r})"
            )

    echoed_in_fmt = session_payload.get("input_audio_format")
    if isinstance(echoed_in_fmt, str) and echoed_in_fmt:
        normalized_in = "pcm16" if echoed_in_fmt in ("pcm16", "audio/pcm") else echoed_in_fmt
        if normalized_in != endpoint_spec.input_audio_encoding:
            raise RuntimeError(
                f"Session echoed unexpected input_audio_format={echoed_in_fmt!r} "
                f"(expected {endpoint_spec.input_audio_encoding!r})"
            )
    echoed_turn = extract_turn_detection_from_session(session_payload)
    if endpoint_spec.protocol == "native":
        if echoed_turn is not None and echoed_turn.get("type") is not None:
            raise RuntimeError("Server enabled turn_detection on a native full-duplex session")
        capabilities = session_payload.get("capabilities")
        if isinstance(capabilities, dict) and (
            capabilities.get("implementation_level") != "model_native_duplex"
            and capabilities.get("supports_model_native_turn_policy") is not True
        ):
            raise RuntimeError("Server reported non-native capabilities on a native session")
    else:
        assert endpoint_spec.turn_detection is not None
        requested_detector = endpoint_spec.turn_detection.get("type")
        echoed_detector = echoed_turn.get("type") if echoed_turn is not None else None
        if echoed_detector != requested_detector:
            raise RuntimeError(
                f"Session echoed unexpected turn_detection.type={echoed_detector!r} (expected {requested_detector!r})"
            )
        for field_name, requested_value in endpoint_spec.turn_detection.items():
            if field_name == "type" or requested_value is None:
                continue
            echoed_value = echoed_turn.get(field_name) if echoed_turn is not None else None
            if type(requested_value) is bool:
                is_match = type(echoed_value) is bool and echoed_value == requested_value
            elif isinstance(requested_value, int | float):
                is_match = (
                    isinstance(echoed_value, int | float)
                    and type(echoed_value) is not bool
                    and math.isclose(float(echoed_value), float(requested_value), rel_tol=0.0, abs_tol=1e-9)
                )
            else:
                is_match = echoed_value == requested_value
            if not is_match:
                raise RuntimeError(f"Session echoed unexpected turn_detection.{field_name}")


def compute_effective_scenario_timeout_seconds(
    *,
    configured_timeout_seconds: float,
    maximum_schedule_duration_seconds: float,
) -> float:
    """Ensure long Duplex-MPE clips (e.g., duplex_mpe_1808 at 292.4s) never expire mid-exchange."""
    if not math.isfinite(configured_timeout_seconds) or configured_timeout_seconds <= 0.0:
        raise ValueError("session_timeout_seconds must be finite and positive")
    required_budget_seconds = maximum_schedule_duration_seconds + SCENARIO_DRAIN_MARGIN_SECONDS
    return max(configured_timeout_seconds, required_budget_seconds)


def prepare_scenario_schedule_sync(
    scenario: DuplexMpeScenarioRecord,
    *,
    dataset_root: Path,
    input_sample_rate_hz: int,
    output_sample_rate_hz: int,
    vad_config: SileroActivityConfig | None,
    vad_model: SileroVadModel | None,
    chunk_milliseconds: float,
) -> tuple[
    Generator[NDArray[np.float32], NDArray[np.float32], ScenarioScheduleResult],
    NDArray[np.float32] | None,
    ScenarioScheduleResult | None,
    float,
]:
    preamble_path = resolve_dataset_audio_path(dataset_root, DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH)
    preamble_samples = load_mono_waveform_at_rate(
        preamble_path,
        target_sample_rate_hz=input_sample_rate_hz,
    )
    turn_inputs = [
        ScenarioTurnInput(
            turn_index=turn.turn_index,
            event_type=turn.event_type,
            speaker=turn.speaker,
            waveform_samples=load_mono_waveform_at_rate(
                resolve_dataset_audio_path(dataset_root, turn.audio_path),
                target_sample_rate_hz=input_sample_rate_hz,
            ),
        )
        for turn in scenario.turns
    ]
    maximum_schedule_duration_seconds = 0.0
    if len(preamble_samples) > 0:
        maximum_schedule_duration_seconds += len(preamble_samples) / input_sample_rate_hz + ORDINARY_GAP_SECONDS
    for turn_input in turn_inputs:
        assert turn_input.waveform_samples is not None
        maximum_schedule_duration_seconds += len(turn_input.waveform_samples) / input_sample_rate_hz
        if turn_input.event_type == "T":
            maximum_schedule_duration_seconds += TARGET_ONSET_DEADLINE_SECONDS + TARGET_MAX_RESPONSE_SECONDS
        elif turn_input.event_type == "N4_Q":
            maximum_schedule_duration_seconds += N4_PAUSE_SECONDS
        else:
            maximum_schedule_duration_seconds += ORDINARY_GAP_SECONDS

    schedule = iter_scenario_schedule(
        turn_inputs,
        input_sample_rate_hz=input_sample_rate_hz,
        output_sample_rate_hz=output_sample_rate_hz,
        vad_config=vad_config,
        vad_model=vad_model,
        chunk_milliseconds=chunk_milliseconds,
        ordinary_gap_seconds=ORDINARY_GAP_SECONDS,
        n4_pause_seconds=N4_PAUSE_SECONDS,
        onset_deadline_seconds=TARGET_ONSET_DEADLINE_SECONDS,
        maximum_response_seconds=TARGET_MAX_RESPONSE_SECONDS,
        preamble_samples=preamble_samples,
        collect_audio=True,
    )
    try:
        first_chunk = next(schedule)
        return schedule, first_chunk, None, maximum_schedule_duration_seconds
    except StopIteration as finished_empty:
        return schedule, None, finished_empty.value, maximum_schedule_duration_seconds


def advance_schedule_step_sync(
    schedule: Generator[NDArray[np.float32], NDArray[np.float32], ScenarioScheduleResult],
    rendered_chunk: NDArray[np.float32],
) -> tuple[NDArray[np.float32] | None, ScenarioScheduleResult | None]:
    try:
        next_chunk = schedule.send(rendered_chunk)
        return next_chunk, None
    except StopIteration as finished_schedule:
        return None, finished_schedule.value


def persist_captured_waveform_sync(
    output_audio_path: Path,
    waveform_samples: NDArray[np.float32],
    sample_rate_hz: int,
) -> str:
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_wav = output_audio_path.with_name(f".{output_audio_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        soundfile.write(
            str(temporary_wav),
            waveform_samples,
            sample_rate_hz,
            format="WAV",
            subtype="FLOAT",
        )
        temporary_wav.replace(output_audio_path)
    finally:
        temporary_wav.unlink(missing_ok=True)
    return compute_file_sha256(output_audio_path)


async def run_realtime_scenario(
    scenario: DuplexMpeScenarioRecord,
    *,
    dataset_root: Path,
    endpoint_spec: RealtimeEndpointSpec,
    output_audio_path: Path,
    vad_config: SileroActivityConfig | None = None,
    vad_model: SileroVadModel | None = None,
    chunk_milliseconds: float = PACKET_DURATION_MILLISECONDS,
    session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
    seed_mapping: dict[str, int] | None = None,
    realtime_pacing: bool = True,
    max_pacing_lag_seconds: float | None = DEFAULT_MAX_PACING_LAG_SECONDS,
    trust_env: bool | None = None,
    clock_fn: MonotonicClock | None = None,
) -> dict[str, JsonValue]:
    """Stream one Duplex-MPE scenario over /v1/realtime(?duplex=1) and return its mpeval.unified.v3 report."""
    if not math.isfinite(chunk_milliseconds) or chunk_milliseconds <= 0.0:
        raise ValueError("chunk_milliseconds must be finite and positive")
    if max_pacing_lag_seconds is not None and (
        not math.isfinite(max_pacing_lag_seconds) or max_pacing_lag_seconds <= 0.0
    ):
        raise ValueError("max_pacing_lag_seconds must be finite and positive when set")
    monotonic_clock: MonotonicClock = clock_fn if clock_fn is not None else time.perf_counter
    scenario_seed = await asyncio.to_thread(resolve_scenario_seed, scenario.scenario_id, seed_mapping)
    (
        schedule,
        initial_input_chunk,
        immediate_result,
        maximum_schedule_duration_seconds,
    ) = await asyncio.to_thread(
        prepare_scenario_schedule_sync,
        scenario,
        dataset_root=dataset_root,
        input_sample_rate_hz=endpoint_spec.input_sample_rate_hz,
        output_sample_rate_hz=endpoint_spec.output_sample_rate_hz,
        vad_config=vad_config,
        vad_model=vad_model,
        chunk_milliseconds=chunk_milliseconds,
    )
    effective_timeout_seconds = compute_effective_scenario_timeout_seconds(
        configured_timeout_seconds=session_timeout_seconds,
        maximum_schedule_duration_seconds=maximum_schedule_duration_seconds,
    )
    effective_trust_env = endpoint_spec.trust_env if trust_env is None else trust_env

    async def execute_once_with_admission_state(
        admission_state: AdmissionState,
    ) -> tuple[
        ScenarioScheduleResult,
        PlaybackTimelineBuffer,
        float,
        float,
        NDArray[np.float32],
        int,
        int,
    ]:
        playback_buffer = PlaybackTimelineBuffer(endpoint_spec.output_sample_rate_hz)
        wire_resampler = StatefulWireResampler(endpoint_spec.output_sample_rate_hz)
        ws_headers = resolve_realtime_bearer_headers(endpoint_spec.api_key_env)
        connection = await open_websocket_with_admission_state(
            endpoint_spec.websocket_url,
            admission_state=admission_state,
            open_timeout_seconds=HANDSHAKE_TIMEOUT_SECONDS,
            additional_headers=ws_headers or None,
            trust_env=effective_trust_env,
        )
        async with connection as websocket:
            event_waiters: dict[str, asyncio.Event] = {}
            latest_events: dict[str, dict[str, JsonValue]] = {}
            receiver_failure: list[BaseException] = []
            receiver_done_event = asyncio.Event()
            control_done_event = asyncio.Event()
            handshake_completed = False
            awaiting_openai_turn_terminal_echo = False
            is_closing_expected = False
            close_requested = False
            stream_epoch_seconds: float | None = None
            total_input_samples = 0
            send_semaphore = asyncio.Semaphore(1)

            truncate_supported = endpoint_spec.supports_audio_truncate
            playback_ack_supported = endpoint_spec.supports_playback_ack
            truncations_sent = 0
            playback_acks_sent = 0
            active_response_ids: set[str] = set()
            cancelled_response_ids: set[str] = set()
            item_id_by_response_id: dict[str, str] = {}

            # Enable speech_started playback interruption on ALL turn-based protocols (`turn_based` + `openai_turn`),
            # so a completed-but-still-queued response stops playing immediately when the next speaker starts!
            # Native scored full-duplex (`protocol == "native"`) never imposes external VAD interruption.
            interrupt_on_speech_started = (
                endpoint_spec.protocol != "native"
                and endpoint_spec.turn_detection is not None
                and endpoint_spec.turn_detection.get("interrupt_response") is not False
            )
            control_queue: asyncio.Queue[tuple[str, dict[str, JsonValue], PlaybackAckRecord | None] | None] = (
                asyncio.Queue(maxsize=512)
            )

            async def send_envelope(envelope: dict[str, JsonValue]) -> bool:
                nonlocal close_requested, is_closing_expected
                event_type = str(envelope.get("type", ""))
                try:
                    async with send_semaphore:
                        if event_type in ("conversation.item.truncate", "playback.ack") and (
                            close_requested or is_closing_expected
                        ):
                            return False
                        if event_type == "session.close":
                            close_requested = True
                            is_closing_expected = True
                        await websocket.send(json.dumps(envelope, allow_nan=False))
                        return True
                except websockets.ConnectionClosed:
                    raise RuntimeError(f"WebSocket connection closed while sending {event_type}") from None

            async def pump_control_messages() -> None:
                nonlocal truncations_sent, playback_acks_sent
                try:
                    while True:
                        item = await control_queue.get()
                        try:
                            if item is None:
                                return
                            kind, envelope, ack_meta = item
                            if close_requested or is_closing_expected:
                                continue
                            if ack_meta is not None and ack_meta.item_id in playback_buffer.truncated_item_ids:
                                continue
                            sent = await send_envelope(envelope)
                            if sent:
                                if kind == "truncate":
                                    truncations_sent += 1
                                elif kind == "ack" and ack_meta is not None:
                                    playback_buffer.mark_ack_sent(ack_meta.item_id, ack_meta.played_milliseconds)
                                    playback_acks_sent += 1
                        finally:
                            control_queue.task_done()
                except asyncio.CancelledError:
                    raise
                except (
                    OSError,
                    ProtocolSessionError,
                    RuntimeError,
                    TimeoutError,
                    asyncio.TimeoutError,
                    ValueError,
                    websockets.WebSocketException,
                ) as exc:
                    receiver_failure.append(exc)
                finally:
                    control_done_event.set()

            def enqueue_playback_truncations(records: list[PlaybackTruncationRecord]) -> None:
                if not records or not truncate_supported or is_closing_expected or endpoint_spec.protocol == "native":
                    return
                for record in records:
                    envelope: dict[str, JsonValue] = {
                        "type": "conversation.item.truncate",
                        "event_id": uuid.uuid4().hex,
                        "item_id": record.item_id,
                        "content_index": 0,
                        "audio_end_ms": record.audio_end_milliseconds,
                    }
                    try:
                        control_queue.put_nowait(("truncate", envelope, None))
                    except asyncio.QueueFull:
                        raise ProtocolSessionError(
                            "Control outbound queue overflowed while sending truncations"
                        ) from None

            def enqueue_playback_acks(records: list[PlaybackAckRecord]) -> None:
                if not records or not playback_ack_supported or is_closing_expected:
                    return
                for record in records:
                    envelope: dict[str, JsonValue] = {
                        "type": "playback.ack",
                        "event_id": uuid.uuid4().hex,
                        "response_id": record.response_id,
                        "item_id": record.item_id,
                        "played_ms": record.played_milliseconds,
                        "committed_ms": record.played_milliseconds,
                    }
                    try:
                        control_queue.put_nowait(("ack", envelope, record))
                    except asyncio.QueueFull:
                        raise ProtocolSessionError(
                            "Control outbound queue overflowed while sending playback.ack"
                        ) from None

            def raise_if_unexpected_background_bug() -> None:
                for bg_task in (control_task, receiver_task):
                    if bg_task.done() and not bg_task.cancelled():
                        bg_exc = bg_task.exception()
                        if bg_exc is not None:
                            raise bg_exc

            async def wait_for_server_event(
                event_type: str,
                wait_timeout_seconds: float | None = None,
            ) -> None:
                raise_if_unexpected_background_bug()
                if receiver_failure:
                    first_err = receiver_failure[0]
                    if isinstance(first_err, RetryableSessionAdmissionError):
                        raise first_err
                    raise RuntimeError(
                        f"Realtime session failed before awaiting {event_type}: "
                        f"{summarize_transport_exception(first_err)}"
                    ) from None
                target_waiter = event_waiters.setdefault(event_type, asyncio.Event())
                if target_waiter.is_set():
                    return
                if receiver_done_event.is_set():
                    raise RuntimeError(f"WebSocket receiver exited before {event_type} arrived")
                event_task = asyncio.create_task(target_waiter.wait())
                done_task = asyncio.create_task(receiver_done_event.wait())
                ctrl_done_task = asyncio.create_task(control_done_event.wait())
                try:
                    completed, _ = await asyncio.wait(
                        {event_task, done_task, ctrl_done_task},
                        timeout=wait_timeout_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    event_task.cancel()
                    done_task.cancel()
                    ctrl_done_task.cancel()
                    await asyncio.gather(event_task, done_task, ctrl_done_task, return_exceptions=True)
                raise_if_unexpected_background_bug()
                if receiver_failure:
                    first_err = receiver_failure[0]
                    if isinstance(first_err, RetryableSessionAdmissionError):
                        raise first_err
                    raise RuntimeError(
                        f"Realtime session failed while awaiting {event_type}: "
                        f"{summarize_transport_exception(first_err)}"
                    ) from None
                if not completed:
                    raise TimeoutError(f"Timed out after {wait_timeout_seconds}s awaiting {event_type}")
                if not target_waiter.is_set():
                    raise RuntimeError(f"WebSocket connection closed before {event_type} arrived")

            async def receive_server_frames() -> None:
                nonlocal is_closing_expected
                try:
                    async for raw_frame in websocket:
                        received_perf_seconds = monotonic_clock()
                        if not realtime_pacing:
                            arrival_seconds = total_input_samples / endpoint_spec.input_sample_rate_hz
                        elif stream_epoch_seconds is None:
                            arrival_seconds = 0.0
                        else:
                            arrival_seconds = max(0.0, received_perf_seconds - stream_epoch_seconds)
                        parsed_event = decode_json_frame(raw_frame, "/v1/realtime stream")
                        event_type = str(parsed_event["type"])
                        latest_events[event_type] = parsed_event
                        if (event_type == "session.closed" and close_requested) or (
                            awaiting_openai_turn_terminal_echo and event_type == "session.updated"
                        ):
                            is_closing_expected = True

                        if event_type == "error":
                            if not handshake_completed and is_retryable_session_rejection(parsed_event):
                                is_closing_expected = True
                                raise RetryableSessionAdmissionError("resource_exhausted")
                            code, _ = extract_error_code_and_retryable(parsed_event)
                            if code in ("playback_ack_too_late", "tool_result_timeout"):
                                logger.debug(
                                    "Ignoring non-fatal duplex session warning",
                                    extra={"code": code, "scenario_id": scenario.scenario_id},
                                )
                                continue
                            raise ProtocolSessionError(format_server_error_event(parsed_event))

                        if event_type == "session.closed" and not close_requested:
                            is_closing_expected = True
                            raise ProtocolSessionError("Server closed session before scenario completion")
                        if event_type == "session.expired" and not close_requested:
                            is_closing_expected = True
                            raise ProtocolSessionError("Server expired session before scenario completion")
                        if event_type in ("session.replaced", "session.resync_required"):
                            is_closing_expected = True
                            raise ProtocolSessionError(f"Server terminated session attachment ({event_type})")

                        if event_type == "response.created":
                            created_response = parsed_event.get("response")
                            raw_top_rid = parsed_event.get("response_id")
                            created_id = (
                                raw_top_rid
                                if isinstance(raw_top_rid, str) and raw_top_rid
                                else (
                                    created_response.get("id")
                                    if isinstance(created_response, dict)
                                    and isinstance(created_response.get("id"), str)
                                    else None
                                )
                            )
                            if isinstance(created_id, str) and created_id:
                                active_response_ids.add(created_id)

                        elif event_type in ("response.output_audio.delta", "response.audio.delta"):
                            raw_delta = parsed_event.get("delta")
                            raw_response_id = parsed_event.get("response_id")
                            delta_response_id = (
                                raw_response_id if isinstance(raw_response_id, str) and raw_response_id else None
                            )
                            raw_item_id = parsed_event.get("item_id")
                            delta_item_id = raw_item_id if isinstance(raw_item_id, str) and raw_item_id else None
                            if delta_response_id is not None and delta_item_id is not None:
                                item_id_by_response_id[delta_response_id] = delta_item_id
                            if not isinstance(raw_delta, str):
                                raise ProtocolSessionError(f"Malformed {event_type} payload: delta must be a string")
                            if raw_delta:
                                raw_fmt = parsed_event.get("format")
                                delta_fmt = raw_fmt if isinstance(raw_fmt, str) else None
                                decoded_samples, container_sr = decode_output_audio_delta_to_float32(
                                    raw_delta,
                                    declared_format=delta_fmt,
                                )
                                raw_chunk_sr = parsed_event.get("sample_rate_hz")
                                source_sr = (
                                    int(raw_chunk_sr)
                                    if type(raw_chunk_sr) is int and raw_chunk_sr > 0
                                    else container_sr
                                )
                                resampler_key = delta_response_id or delta_item_id or "default"
                                is_delta_cancelled = (
                                    delta_response_id is not None and delta_response_id in cancelled_response_ids
                                ) or (delta_item_id is not None and delta_item_id in playback_buffer.truncated_item_ids)
                                if is_delta_cancelled:
                                    wire_resampler.discard(stream_key=resampler_key)
                                    expected_len = (
                                        int(
                                            round(
                                                len(decoded_samples) * endpoint_spec.output_sample_rate_hz / source_sr
                                            )
                                        )
                                        if source_sr is not None and source_sr > 0
                                        else len(decoded_samples)
                                    )
                                    playback_buffer.cancelled_unplayed_samples += expected_len
                                else:
                                    effective_source_sr = (
                                        source_sr if source_sr is not None else endpoint_spec.output_sample_rate_hz
                                    )
                                    decoded_samples = wire_resampler.push(
                                        decoded_samples,
                                        effective_source_sr,
                                        stream_key=resampler_key,
                                    )
                                    playback_buffer.append_audio(
                                        arrival_seconds,
                                        decoded_samples,
                                        response_id=delta_response_id,
                                        item_id=delta_item_id,
                                    )

                        elif event_type in ("response.output_audio.done", "response.audio.done"):
                            raw_done_audio_rid = parsed_event.get("response_id")
                            done_audio_rid = (
                                raw_done_audio_rid
                                if isinstance(raw_done_audio_rid, str) and raw_done_audio_rid
                                else None
                            )
                            raw_done_audio_iid = parsed_event.get("item_id")
                            done_audio_iid = (
                                raw_done_audio_iid
                                if isinstance(raw_done_audio_iid, str) and raw_done_audio_iid
                                else (
                                    item_id_by_response_id.get(done_audio_rid) if done_audio_rid is not None else None
                                )
                            )
                            resampler_key = done_audio_rid or done_audio_iid or "default"
                            is_audio_done_cancelled = (
                                done_audio_rid is not None and done_audio_rid in cancelled_response_ids
                            ) or (done_audio_iid is not None and done_audio_iid in playback_buffer.truncated_item_ids)
                            if is_audio_done_cancelled:
                                wire_resampler.discard(stream_key=resampler_key)
                            else:
                                flushed_tail = wire_resampler.flush(stream_key=resampler_key)
                                if len(flushed_tail) > 0:
                                    playback_buffer.append_audio(
                                        arrival_seconds,
                                        flushed_tail,
                                        response_id=done_audio_rid,
                                        item_id=done_audio_iid,
                                    )

                        elif event_type == "output_audio_buffer.cleared":
                            raw_cleared_rid = parsed_event.get("response_id")
                            cleared_rid = (
                                raw_cleared_rid if isinstance(raw_cleared_rid, str) and raw_cleared_rid else None
                            )
                            raw_cleared_iid = parsed_event.get("item_id")
                            cleared_iid = (
                                raw_cleared_iid
                                if isinstance(raw_cleared_iid, str) and raw_cleared_iid
                                else (item_id_by_response_id.get(cleared_rid) if cleared_rid is not None else None)
                            )
                            if cleared_rid is not None:
                                cancelled_response_ids.add(cleared_rid)
                                active_response_ids.discard(cleared_rid)
                                wire_resampler.discard(stream_key=cleared_rid)
                                if cleared_iid is not None:
                                    wire_resampler.discard(stream_key=cleared_iid)
                            elif cleared_iid is not None:
                                wire_resampler.discard(stream_key=cleared_iid)
                                playback_buffer.truncated_item_ids.add(cleared_iid)
                                for mapped_rid, mapped_iid in list(item_id_by_response_id.items()):
                                    if mapped_iid == cleared_iid:
                                        cancelled_response_ids.add(mapped_rid)
                                        active_response_ids.discard(mapped_rid)
                                        wire_resampler.discard(stream_key=mapped_rid)
                            else:
                                for open_rid in list(active_response_ids):
                                    cancelled_response_ids.add(open_rid)
                                active_response_ids.clear()
                                wire_resampler.discard_all()
                            truncation_records = playback_buffer.cancel_unplayed(
                                arrival_seconds,
                                target_response_id=cleared_rid,
                                target_item_id=cleared_iid,
                            )
                            enqueue_playback_truncations(truncation_records)

                        elif interrupt_on_speech_started and event_type == "input_audio_buffer.speech_started":
                            for open_rid in list(active_response_ids):
                                cancelled_response_ids.add(open_rid)
                            active_response_ids.clear()
                            wire_resampler.discard_all()
                            truncation_records = playback_buffer.cancel_unplayed(arrival_seconds)
                            enqueue_playback_truncations(truncation_records)

                        elif event_type == "response.done":
                            response_payload = parsed_event.get("response")
                            if not isinstance(response_payload, dict):
                                raise ProtocolSessionError(
                                    "Malformed response.done payload: response must be an object"
                                )
                            if response_payload.get("status") == "failed":
                                raise ProtocolSessionError("Server reported failed response.done")
                            raw_done_rid = parsed_event.get("response_id") or response_payload.get("id")
                            done_response_id = raw_done_rid if isinstance(raw_done_rid, str) and raw_done_rid else None
                            if done_response_id is not None:
                                active_response_ids.discard(done_response_id)
                            done_item_id: str | None = (
                                item_id_by_response_id.get(done_response_id) if done_response_id is not None else None
                            )
                            raw_output_items = response_payload.get("output")
                            if (
                                done_item_id is None
                                and isinstance(raw_output_items, list)
                                and raw_output_items
                                and isinstance(raw_output_items[0], dict)
                            ):
                                candidate_output_id = raw_output_items[0].get("id")
                                if isinstance(candidate_output_id, str) and candidate_output_id:
                                    done_item_id = candidate_output_id
                            done_resampler_key = done_response_id or done_item_id or "default"
                            is_done_cancelled = (
                                response_payload.get("status") == "cancelled"
                                or (done_response_id is not None and done_response_id in cancelled_response_ids)
                                or (done_item_id is not None and done_item_id in playback_buffer.truncated_item_ids)
                            )
                            if is_done_cancelled:
                                wire_resampler.discard(stream_key=done_resampler_key)
                                if done_response_id is not None:
                                    cancelled_response_ids.add(done_response_id)
                                truncation_records = playback_buffer.cancel_unplayed(
                                    arrival_seconds,
                                    target_response_id=done_response_id,
                                    target_item_id=done_item_id,
                                )
                                enqueue_playback_truncations(truncation_records)
                            else:
                                flushed_done_tail = wire_resampler.flush(stream_key=done_resampler_key)
                                if len(flushed_done_tail) > 0:
                                    playback_buffer.append_audio(
                                        arrival_seconds,
                                        flushed_done_tail,
                                        response_id=done_response_id,
                                        item_id=done_item_id,
                                    )

                        event_waiters.setdefault(event_type, asyncio.Event()).set()

                    if not is_closing_expected:
                        receiver_failure.append(
                            ProtocolSessionError("WebSocket connection closed before session completion")
                        )
                except asyncio.CancelledError:
                    raise
                except websockets.ConnectionClosed as exc:
                    if not is_closing_expected:
                        receiver_failure.append(
                            ProtocolSessionError(
                                f"WebSocket connection closed prematurely: {summarize_transport_exception(exc)}"
                            )
                        )
                except (
                    OSError,
                    ProtocolSessionError,
                    RetryableSessionAdmissionError,
                    RuntimeError,
                    TimeoutError,
                    asyncio.TimeoutError,
                    UnicodeDecodeError,
                    ValueError,
                    websockets.WebSocketException,
                ) as exc:
                    receiver_failure.append(exc)
                finally:
                    receiver_done_event.set()

            receiver_task = asyncio.create_task(receive_server_frames())
            control_task = asyncio.create_task(pump_control_messages())
            cancelled_externally = False
            try:
                if endpoint_spec.wire_mode == "duplex":
                    duplex_protocol: Literal["native", "turn_based"] = (
                        "native" if endpoint_spec.protocol == "native" else "turn_based"
                    )
                    opening_session = build_duplex_session_payload(
                        model=endpoint_spec.reported_model,
                        protocol=duplex_protocol,
                        input_audio_encoding=endpoint_spec.input_audio_encoding,
                        input_sample_rate_hz=endpoint_spec.input_sample_rate_hz,
                        output_sample_rate_hz=endpoint_spec.output_sample_rate_hz,
                        turn_detection=endpoint_spec.turn_detection,
                        supports_barge_in=endpoint_spec.supports_barge_in,
                        supports_playback_ack=endpoint_spec.supports_playback_ack,
                        requires_auto_response_on_open=endpoint_spec.requires_auto_response_on_open,
                        session_options=endpoint_spec.session_options,
                    )
                    await send_envelope(
                        {
                            "type": "session.update",
                            "event_id": uuid.uuid4().hex,
                            "session": opening_session,
                        }
                    )
                    await wait_for_server_event("session.created", HANDSHAKE_TIMEOUT_SECONDS)
                    await wait_for_server_event("session.updated", HANDSHAKE_TIMEOUT_SECONDS)
                    created_session = latest_events["session.created"].get("session")
                    if isinstance(created_session, dict):
                        caps = created_session.get("capabilities")
                        if isinstance(caps, dict):
                            if type(caps.get("supports_audio_truncate")) is bool:
                                truncate_supported = bool(caps["supports_audio_truncate"])
                            if type(caps.get("supports_playback_ack")) is bool:
                                playback_ack_supported = bool(caps["supports_playback_ack"])
                    validate_accepted_session(latest_events["session.updated"], endpoint_spec)
                else:
                    assert endpoint_spec.turn_detection is not None
                    await wait_for_server_event("session.created", HANDSHAKE_TIMEOUT_SECONDS)
                    event_waiters.setdefault("session.updated", asyncio.Event()).clear()
                    opening_turn_session = build_openai_turn_session_payload(
                        model=endpoint_spec.reported_model,
                        turn_detection=endpoint_spec.turn_detection,
                        session_options=endpoint_spec.session_options,
                    )
                    await send_envelope(
                        {
                            "type": "session.update",
                            "event_id": uuid.uuid4().hex,
                            "session": opening_turn_session,
                        }
                    )
                    await wait_for_server_event("session.updated", HANDSHAKE_TIMEOUT_SECONDS)
                    validate_accepted_session(latest_events["session.updated"], endpoint_spec)

                handshake_completed = True
                rendered_output_samples = 0
                packet_sequence = 0
                max_observed_pacing_lag_seconds = 0.0

                def record_and_validate_pacing_lag(nominal_deadline_seconds: float) -> None:
                    nonlocal max_observed_pacing_lag_seconds
                    observed_lag_seconds = max(0.0, monotonic_clock() - nominal_deadline_seconds)
                    max_observed_pacing_lag_seconds = max(max_observed_pacing_lag_seconds, observed_lag_seconds)
                    if max_pacing_lag_seconds is not None and observed_lag_seconds > max_pacing_lag_seconds:
                        raise RuntimeError(
                            f"Realtime input pacing lagged by {observed_lag_seconds:.3f}s "
                            f"(limit {max_pacing_lag_seconds:.3f}s)"
                        )

                if immediate_result is not None:
                    schedule_result = immediate_result
                else:
                    assert initial_input_chunk is not None
                    next_input_chunk: NDArray[np.float32] | None = initial_input_chunk
                    stream_epoch_seconds = monotonic_clock()

                    async def stream_measured_schedule() -> ScenarioScheduleResult:
                        nonlocal next_input_chunk, total_input_samples, packet_sequence, rendered_output_samples
                        completed_res: ScenarioScheduleResult | None = None
                        assert stream_epoch_seconds is not None
                        while next_input_chunk is not None:
                            raise_if_unexpected_background_bug()
                            if receiver_failure:
                                raise RuntimeError(
                                    f"Realtime receiver failed mid-scenario: "
                                    f"{summarize_transport_exception(receiver_failure[0])}"
                                ) from None
                            if receiver_done_event.is_set():
                                raise RuntimeError("WebSocket receiver exited unexpectedly during scenario streaming")
                            if control_done_event.is_set() and not receiver_failure:
                                raise RuntimeError("Control outbound sender exited unexpectedly during streaming")
                            if realtime_pacing and packet_sequence > 0:
                                record_and_validate_pacing_lag(
                                    stream_epoch_seconds + total_input_samples / endpoint_spec.input_sample_rate_hz
                                )
                            chunk_samples_count = len(next_input_chunk)
                            chunk_duration_ms = round(chunk_samples_count * 1000.0 / endpoint_spec.input_sample_rate_hz)
                            total_input_samples += chunk_samples_count
                            audio_end_ms = round(total_input_samples * 1000.0 / endpoint_spec.input_sample_rate_hz)
                            encoded_audio = encode_input_audio_chunk(
                                next_input_chunk,
                                endpoint_spec.input_audio_encoding,
                            )
                            append_envelope: dict[str, JsonValue] = {
                                "type": "input_audio_buffer.append",
                                "event_id": uuid.uuid4().hex,
                                "audio": encoded_audio,
                            }
                            if endpoint_spec.wire_mode == "duplex":
                                append_envelope["format"] = endpoint_spec.input_audio_encoding
                                append_envelope["sample_rate_hz"] = endpoint_spec.input_sample_rate_hz
                                append_envelope["duration_ms"] = chunk_duration_ms
                                append_envelope["audio_end_ms"] = audio_end_ms
                            await send_envelope(append_envelope)
                            packet_sequence += 1

                            target_perf_deadline = (
                                stream_epoch_seconds + total_input_samples / endpoint_spec.input_sample_rate_hz
                            )
                            if realtime_pacing:
                                await asyncio.sleep(max(0.0, target_perf_deadline - monotonic_clock()))
                                record_and_validate_pacing_lag(target_perf_deadline)
                            else:
                                await asyncio.sleep(0)

                            raise_if_unexpected_background_bug()
                            if receiver_failure:
                                raise RuntimeError(
                                    f"Realtime receiver failed while awaiting output: "
                                    f"{summarize_transport_exception(receiver_failure[0])}"
                                ) from None
                            if receiver_done_event.is_set():
                                raise RuntimeError("WebSocket receiver exited unexpectedly while awaiting output")

                            if control_done_event.is_set() and not receiver_failure:
                                raise RuntimeError("Control outbound sender exited unexpectedly while awaiting output")

                            target_output_samples = int(
                                round(
                                    total_input_samples
                                    * endpoint_spec.output_sample_rate_hz
                                    / endpoint_spec.input_sample_rate_hz
                                )
                            )
                            rendered_chunk, ack_records = playback_buffer.render_interval(
                                rendered_output_samples,
                                target_output_samples,
                            )
                            rendered_output_samples = target_output_samples
                            enqueue_playback_acks(ack_records)

                            if realtime_pacing:
                                next_input_chunk, completed_res = await asyncio.to_thread(
                                    advance_schedule_step_sync,
                                    schedule,
                                    rendered_chunk,
                                )
                            else:
                                next_input_chunk, completed_res = advance_schedule_step_sync(
                                    schedule,
                                    rendered_chunk,
                                )
                        if realtime_pacing:
                            record_and_validate_pacing_lag(
                                stream_epoch_seconds + total_input_samples / endpoint_spec.input_sample_rate_hz
                            )
                        assert completed_res is not None
                        return completed_res

                    # Bound the measured streaming schedule separately from cold handshake/admission!
                    schedule_result = await asyncio.wait_for(
                        stream_measured_schedule(),
                        timeout=effective_timeout_seconds,
                    )
                    raise_if_unexpected_background_bug()
                    if receiver_failure:
                        raise RuntimeError(
                            f"Realtime session failed before control drain: "
                            f"{summarize_transport_exception(receiver_failure[0])}"
                        ) from None
                    join_task = asyncio.create_task(control_queue.join())
                    ctrl_exit_task = asyncio.create_task(control_done_event.wait())
                    recv_exit_task = asyncio.create_task(receiver_done_event.wait())
                    drain_finished = False
                    try:
                        completed_drain, _ = await asyncio.wait(
                            {join_task, ctrl_exit_task, recv_exit_task},
                            timeout=min(5.0, CLOSE_DRAIN_TIMEOUT_SECONDS),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        drain_finished = join_task in completed_drain
                    finally:
                        join_task.cancel()
                        ctrl_exit_task.cancel()
                        recv_exit_task.cancel()
                        await asyncio.gather(join_task, ctrl_exit_task, recv_exit_task, return_exceptions=True)
                    raise_if_unexpected_background_bug()
                    if receiver_failure:
                        raise RuntimeError(
                            f"Realtime control sender failed during drain: "
                            f"{summarize_transport_exception(receiver_failure[0])}"
                        ) from None
                    if not completed_drain:
                        raise TimeoutError("Timed out draining outbound realtime control messages")
                    if not drain_finished:
                        raise RuntimeError("Realtime session terminated before outbound control messages drained")

                elapsed_wall_seconds = 0.0 if stream_epoch_seconds is None else monotonic_clock() - stream_epoch_seconds
                if endpoint_spec.wire_mode == "openai_turn":
                    if realtime_pacing:
                        await asyncio.sleep(OPENAI_TURN_TAIL_WAIT_SECONDS)
                    event_waiters.setdefault("session.updated", asyncio.Event()).clear()
                    awaiting_openai_turn_terminal_echo = True
                    await send_envelope(
                        {
                            "type": "session.update",
                            "event_id": uuid.uuid4().hex,
                            "session": {"type": "realtime"},
                        }
                    )
                    await wait_for_server_event("session.updated", CLOSE_DRAIN_TIMEOUT_SECONDS)
                    is_closing_expected = True
                else:
                    closed_waiter = event_waiters.setdefault("session.closed", asyncio.Event())
                    if not closed_waiter.is_set():
                        await send_envelope({"type": "session.close", "event_id": uuid.uuid4().hex})
                        await wait_for_server_event("session.closed", CLOSE_DRAIN_TIMEOUT_SECONDS)

                if playback_buffer.tail_samples > rendered_output_samples:
                    drain_tail_waveform, _ = playback_buffer.render_interval(
                        rendered_output_samples,
                        playback_buffer.tail_samples,
                    )
                else:
                    drain_tail_waveform = np.zeros(0, dtype=np.float32)

                return (
                    schedule_result,
                    playback_buffer,
                    elapsed_wall_seconds,
                    max_observed_pacing_lag_seconds,
                    drain_tail_waveform,
                    truncations_sent,
                    playback_acks_sent,
                )
            except asyncio.CancelledError:
                cancelled_externally = True
                raise
            finally:
                try:
                    if endpoint_spec.wire_mode == "duplex" and handshake_completed and not close_requested:
                        close_requested = True
                        is_closing_expected = True
                        try:
                            await asyncio.wait_for(
                                websocket.send(
                                    json.dumps({"type": "session.close", "event_id": uuid.uuid4().hex}, allow_nan=False)
                                ),
                                timeout=2.0,
                            )
                        except (OSError, TimeoutError, asyncio.TimeoutError, websockets.WebSocketException):
                            pass
                finally:
                    try:
                        control_queue.put_nowait(None)
                    except asyncio.QueueFull:
                        pass
                    control_task.cancel()
                    receiver_task.cancel()
                    await asyncio.gather(control_task, receiver_task, return_exceptions=True)
                    unexpected_task_exc: BaseException | None = None
                    for finished_task in (control_task, receiver_task):
                        if finished_task.cancelled():
                            continue
                        task_exc = finished_task.exception()
                        if task_exc is not None and unexpected_task_exc is None:
                            unexpected_task_exc = task_exc
                    if unexpected_task_exc is not None and not cancelled_externally:
                        raise unexpected_task_exc

    async def execute_with_admission_retries() -> tuple[
        ScenarioScheduleResult,
        PlaybackTimelineBuffer,
        float,
        float,
        NDArray[np.float32],
        int,
        int,
    ]:
        admission_state = AdmissionState()
        while True:
            try:
                return await execute_once_with_admission_state(admission_state)
            except RetryableSessionAdmissionError:
                admission_state.denials += 1
                if admission_state.denials > ADMISSION_RETRIES:
                    raise RuntimeError(
                        f"Session admission rejected after {admission_state.denials} "
                        f"combined admission denials (budget {ADMISSION_RETRIES} retries)"
                    ) from None
                # Re-prepare a fresh schedule generator ONLY if initial_input_chunk was already consumed;
                # because RetryableSessionAdmissionError occurs during handshake before any `schedule.send()`,
                # the generator is still positioned right after its first yielded chunk.
                await asyncio.sleep(ADMISSION_BACKOFF_S * admission_state.denials)

    overall_wall_deadline = HANDSHAKE_TIMEOUT_SECONDS + effective_timeout_seconds + CLOSE_DRAIN_TIMEOUT_SECONDS + 30.0
    try:
        (
            schedule_result,
            playback_buffer,
            elapsed_wall_seconds,
            max_observed_pacing_lag_seconds,
            drain_tail_waveform,
            truncations_sent_count,
            playback_acks_sent_count,
        ) = await asyncio.wait_for(execute_with_admission_retries(), timeout=overall_wall_deadline)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        if str(exc):
            raise TimeoutError(str(exc)) from None
        raise TimeoutError(f"Realtime session exceeded deadline of {effective_timeout_seconds:.1f}s") from None

    assert schedule_result.assistant_waveform is not None
    # Persist ONLY the exact scored timeline waveform so WAV length matches `total_duration_s` without unscored tail!
    audio_sha256 = await asyncio.to_thread(
        persist_captured_waveform_sync,
        output_audio_path,
        schedule_result.assistant_waveform,
        endpoint_spec.output_sample_rate_hz,
    )
    protocol_fingerprint = await asyncio.to_thread(compute_protocol_fingerprint)

    report = schedule_result.to_report_dict()
    report.update(
        {
            "scenario_id": scenario.scenario_id,
            "pair_id": scenario.pair_id,
            "addressing": scenario.addressing,
            "run_seed": scenario_seed,
            "seed_policy": SEED_POLICY_METADATA_ONLY,
            "server_seed_applied": False,
            "code_fingerprint": protocol_fingerprint,
            "assistant_wav_sr": endpoint_spec.output_sample_rate_hz,
            "input_audio_encoding": endpoint_spec.input_audio_encoding,
            "evaluation_clock": ("realtime_wall_clock" if realtime_pacing else "audio_samples_excluding_compute_time"),
            "realtime_protocol": endpoint_spec.protocol,
            "elapsed_wall_s": round(elapsed_wall_seconds, 4),
            "max_pacing_lag_ms": round(max_observed_pacing_lag_seconds * 1000.0, 3),
            "effective_session_timeout_s": round(effective_timeout_seconds, 4),
            "cancelled_unplayed_samples": playback_buffer.cancelled_unplayed_samples,
            "truncations_sent": truncations_sent_count,
            "playback_acks_sent": playback_acks_sent_count,
            "drain_tail_samples": int(len(drain_tail_waveform)),
            "assistant_audio_sha256": audio_sha256,
        }
    )
    return report
