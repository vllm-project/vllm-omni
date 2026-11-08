# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``RealtimeProtocolCapabilities``: one consumer's answers, the codec's checking.

This is the seam that lets a non-duplex Realtime surface validate a
``session.update`` without a ``DuplexModelPlugin``. What it has to get right is
narrow: the order the checks run in (formats before turn detection, because
that is the order clients already see), the error codes, and that a consumer
which declares a narrower format set is actually narrowed.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from vllm_omni.protocol.realtime.capabilities import (
    RealtimeProtocolCapabilities,
    RealtimeSessionRejection,
    validate_session_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEFAULT = RealtimeProtocolCapabilities()


def test_a_consumer_without_a_turn_detection_check_accepts_a_plain_session():
    assert (
        validate_session_payload({"model": "m", "audio": {"input": {"format": "pcm16"}}}, capabilities=_DEFAULT) is None
    )


def test_a_consumer_without_a_turn_detection_check_ignores_turn_detection():
    payload = {"turn_detection": {"type": "semantic_vad"}}

    assert validate_session_payload(payload, capabilities=_DEFAULT) is None


def test_an_unsupported_input_format_is_rejected_with_the_realtime_code():
    rejection = validate_session_payload({"input_audio_format": "opus"}, capabilities=_DEFAULT)

    assert rejection == RealtimeSessionRejection(
        code="unsupported_audio_format", message="Unsupported input_audio_format: opus"
    )


def test_a_consumer_may_declare_a_narrower_format_set():
    pcm_only = RealtimeProtocolCapabilities(input_audio_formats=frozenset({"pcm16"}))

    assert validate_session_payload({"input_audio_format": "pcm16"}, capabilities=pcm_only) is None
    rejection = validate_session_payload({"input_audio_format": "g711_ulaw"}, capabilities=pcm_only)
    assert rejection is not None
    assert rejection.code == "unsupported_audio_format"
    # The default capabilities still accept it: the narrowing is per consumer.
    assert validate_session_payload({"input_audio_format": "g711_ulaw"}, capabilities=_DEFAULT) is None


def test_the_turn_detection_check_runs_and_reports_its_own_code():
    def reject_everything(session_payload: Mapping[str, object]) -> str | None:
        del session_payload
        return "server_vad is not available"

    capabilities = RealtimeProtocolCapabilities(validate_turn_detection=reject_everything)

    rejection = validate_session_payload({"turn_detection": {"type": "server_vad"}}, capabilities=capabilities)

    assert rejection == RealtimeSessionRejection(
        code="unsupported_turn_detection", message="server_vad is not available", param="turn_detection"
    )


def test_a_bad_format_is_reported_before_a_bad_turn_detection():
    calls: list[Mapping[str, object]] = []

    def record(session_payload: Mapping[str, object]) -> str | None:
        calls.append(session_payload)
        return "turn detection is wrong too"

    capabilities = RealtimeProtocolCapabilities(validate_turn_detection=record)

    rejection = validate_session_payload(
        {"input_audio_format": "opus", "turn_detection": {"type": "semantic_vad"}},
        capabilities=capabilities,
    )

    assert rejection is not None
    assert rejection.code == "unsupported_audio_format"
    assert calls == []
