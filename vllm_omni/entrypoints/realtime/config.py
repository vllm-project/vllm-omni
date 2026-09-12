# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session option validation shared by Realtime and duplex serving."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

# Canonical extra_body key for the per-session opt-in to a model-native
# duplex runtime, plus the deprecated model-prefixed spelling it replaced.
# Client payloads may still use the legacy key; it is folded into the
# canonical one at ingestion so everything downstream (and every echo in
# session.created/session.updated) sees only NATIVE_DUPLEX_KEY.
NATIVE_DUPLEX_KEY = "native_duplex"
LEGACY_NATIVE_DUPLEX_KEY = "minicpmo45_native_duplex"


def normalize_native_duplex_key(extra_body: dict[str, object]) -> dict[str, object]:
    """Fold the deprecated alias into the canonical key (canonical wins)."""
    if LEGACY_NATIVE_DUPLEX_KEY in extra_body:
        legacy = extra_body.pop(LEGACY_NATIVE_DUPLEX_KEY)
        extra_body.setdefault(NATIVE_DUPLEX_KEY, legacy)
    return extra_body


def native_duplex_opt_in(extra_body: Mapping[str, object]) -> object:
    """Read the opt-in flag, accepting the deprecated alias for raw dicts."""
    if NATIVE_DUPLEX_KEY in extra_body:
        return extra_body[NATIVE_DUPLEX_KEY]
    return extra_body.get(LEGACY_NATIVE_DUPLEX_KEY)


_SERVER_VAD_FIELDS = {
    "type",
    "threshold",
    "prefix_padding_ms",
    "silence_duration_ms",
    "create_response",
    "interrupt_response",
    "min_speech_duration_ms",
}


@dataclass(frozen=True, slots=True)
class ServerVADConfig:
    """Validated OpenAI-compatible ``server_vad`` session configuration."""

    type: Literal["server_vad"] = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True
    # Omission is resolved from runtime capabilities before the session is opened.
    interrupt_response: bool | None = None
    min_speech_duration_ms: int | None = None

    @classmethod
    def from_value(cls, value: object) -> ServerVADConfig:
        if not isinstance(value, dict):
            raise ValueError("turn_detection must be null or an object")
        unknown = sorted(set(value) - _SERVER_VAD_FIELDS)
        if unknown:
            raise ValueError(f"Unknown server_vad field(s): {', '.join(unknown)}")
        vad_type = value.get("type")
        if vad_type != "server_vad":
            raise ValueError("turn_detection.type must be 'server_vad'")

        threshold = value.get("threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0 <= threshold <= 1:
            raise ValueError("server_vad.threshold must be a number between 0 and 1")

        prefix_padding_ms = value.get("prefix_padding_ms", 300)
        if isinstance(prefix_padding_ms, bool) or not isinstance(prefix_padding_ms, int) or prefix_padding_ms < 0:
            raise ValueError("server_vad.prefix_padding_ms must be a non-negative integer")

        silence_duration_ms = value.get("silence_duration_ms", 500)
        if (
            isinstance(silence_duration_ms, bool)
            or not isinstance(silence_duration_ms, int)
            or silence_duration_ms <= 0
        ):
            raise ValueError("server_vad.silence_duration_ms must be a positive integer")

        create_response = value.get("create_response", True)
        if not isinstance(create_response, bool):
            raise ValueError("server_vad.create_response must be a boolean")

        interrupt_response = value.get("interrupt_response")
        if "interrupt_response" in value and not isinstance(interrupt_response, bool):
            raise ValueError("server_vad.interrupt_response must be a boolean")

        min_speech_duration_ms = value.get("min_speech_duration_ms")
        if min_speech_duration_ms is not None and (
            isinstance(min_speech_duration_ms, bool)
            or not isinstance(min_speech_duration_ms, int | float)
            or not np.isfinite(float(min_speech_duration_ms))
            or min_speech_duration_ms < 0
        ):
            raise ValueError("server_vad.min_speech_duration_ms must be a non-negative number")

        return cls(
            type=vad_type,
            threshold=float(threshold),
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
            create_response=create_response,
            interrupt_response=interrupt_response,
            min_speech_duration_ms=(int(min_speech_duration_ms) if min_speech_duration_ms is not None else None),
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": self.type,
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "create_response": self.create_response,
        }
        if self.interrupt_response is not None:
            result["interrupt_response"] = self.interrupt_response
        if self.min_speech_duration_ms is not None:
            result["min_speech_duration_ms"] = self.min_speech_duration_ms
        return result


def parse_session_turn_detection(
    payload: Mapping[str, object],
) -> tuple[bool, ServerVADConfig | None]:
    """Extract and validate OpenAI-compatible turn detection aliases."""
    configured_values: list[tuple[str, object]] = []
    if "turn_detection" in payload:
        configured_values.append(("turn_detection", payload["turn_detection"]))
    audio = payload.get("audio")
    if isinstance(audio, Mapping):
        audio_input = audio.get("input")
        if isinstance(audio_input, Mapping) and "turn_detection" in audio_input:
            configured_values.append(("audio.input.turn_detection", audio_input["turn_detection"]))
    if not configured_values:
        return False, None

    first_path, first_value = configured_values[0]
    first_config = None if first_value is None else ServerVADConfig.from_value(first_value)
    for field_path, value in configured_values[1:]:
        config = None if value is None else ServerVADConfig.from_value(value)
        if first_config is not None and config is not None:
            # An omitted option must not conflict with an explicit alias value.
            if first_config.interrupt_response is None:
                first_config = replace(first_config, interrupt_response=config.interrupt_response)
            if config.interrupt_response is None:
                config = replace(config, interrupt_response=first_config.interrupt_response)
        if config != first_config:
            raise ValueError(f"{first_path} and {field_path} must not specify conflicting values")
    return True, first_config
