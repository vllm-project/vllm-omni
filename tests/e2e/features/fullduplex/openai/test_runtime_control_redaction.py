# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for duplex ``runtime_control`` payload redaction.

The redacted result is emitted on the WebSocket (for example
``session.created.runtime_control``), so it must always survive
``json.dumps``. Control results carry typed values straight from the engine --
``fence`` and ``accepted_fence`` are ``DuplexFence`` instances added by
``DuplexControlClient.execute`` -- which previously reached
``WebSocket.send_json`` unconverted and killed the session at handshake.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

import pytest

from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.openai.runtime_bridge import NativeRuntimeBridgeMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_redact = NativeRuntimeBridgeMixin._redact_runtime_control_result


def _control_result() -> dict[str, object]:
    """The shape ``DuplexControlClient.execute`` returns for an open."""
    return {
        "ok": True,
        "operation": "open",
        "session_id": "sess-1",
        "fence": DuplexFence("sess-1", epoch=0, turn_id=0, response_seq=0, incarnation=0),
        "accepted_fence": DuplexFence("sess-1", epoch=1, turn_id=2, response_seq=3, incarnation=4),
        "stage_results": [{"stage_id": -1, "replica_id": -1, "result": {"supported": True}}],
    }


def test_redacted_control_result_is_json_serializable() -> None:
    """The whole payload must serialize the way send_json does."""
    redacted = _redact(_control_result())
    # Must not raise.
    json.dumps({"type": "session.created", "runtime_control": redacted})


def test_fences_are_converted_to_plain_dicts() -> None:
    redacted = _redact(_control_result())
    assert redacted["fence"] == {
        "session_id": "sess-1",
        "epoch": 0,
        "turn_id": 0,
        "response_seq": 0,
        "incarnation": 0,
    }
    # Field values are preserved, not just the type.
    assert redacted["accepted_fence"]["epoch"] == 1
    assert redacted["accepted_fence"]["turn_id"] == 2
    assert redacted["accepted_fence"]["response_seq"] == 3
    assert redacted["accepted_fence"]["incarnation"] == 4


def test_nested_dataclasses_are_converted() -> None:
    """Fences can appear inside stage results, not only at the top level."""
    redacted = _redact(
        {
            "stage_results": [
                {"stage_id": 0, "result": {"fence": DuplexFence("sess-2", epoch=7)}},
            ]
        }
    )
    json.dumps(redacted)
    assert redacted["stage_results"][0]["result"]["fence"]["epoch"] == 7


def test_enums_are_reduced_to_values() -> None:
    class _Mode(str, Enum):
        APPEND = "append_audio_chunk"

    redacted = _redact({"mode": _Mode.APPEND})
    json.dumps(redacted)
    assert redacted["mode"] == "append_audio_chunk"


def test_redaction_still_drops_heavy_payload_keys() -> None:
    """The pre-existing contract must be preserved."""
    redacted = _redact(
        {
            "ok": True,
            "stage_handoff": object(),
            "tts_handoff": object(),
            "omni_payload": object(),
            "tts_hidden_states": object(),
            "tts_token_ids": object(),
            "traceback": "long trace",
        }
    )
    assert redacted == {"ok": True}


def test_plain_values_pass_through_unchanged() -> None:
    payload = {"ok": True, "count": 3, "name": "x", "missing": None, "items": [1, "two"]}
    assert _redact(payload) == payload


def test_dataclass_types_are_not_treated_as_instances() -> None:
    """A class object is not a dataclass instance and must not be asdict'd."""

    @dataclass
    class _Shape:
        x: int = 1

    assert _redact(_Shape) is _Shape
