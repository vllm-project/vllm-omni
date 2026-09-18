# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The overlap decision, reached directly.

These rules decide what happens when input arrives while the model is speaking.
They were unreachable except by driving a whole append through the runner, so
none of them had a test that named the rule it was checking.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session import overlap_policy
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _session(**config: object) -> DuplexEngineSession:
    values: dict[str, object] = {"model": "fake-model"}
    values.update(config)
    return DuplexEngineSession(session_id="sid", config=DuplexSessionConfig(**values))


def _pcm16(rms: float, samples: int = 1600) -> str:
    data = np.full(samples, int(rms * 32768), dtype=np.int16)
    return base64.b64encode(data.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- #
# Speech detection                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("flag", [True, False])
def test_an_explicit_is_speech_flag_wins_over_the_audio(flag: bool) -> None:
    session = _session()
    payload = {"format": "pcm16", "audio": _pcm16(0.9 if not flag else 0.0)}
    assert overlap_policy.input_looks_like_speech(session, {"is_speech": flag}, payload) is flag


def test_vad_probability_decides_when_no_flag_is_given() -> None:
    session = _session()
    payload = {"format": "pcm16", "audio": _pcm16(0.0)}
    assert overlap_policy.input_looks_like_speech(session, {"vad": {"speech_probability": 0.9}}, payload) is True
    assert overlap_policy.input_looks_like_speech(session, {"vad": {"speech_probability": 0.1}}, payload) is False


def test_quiet_audio_below_the_configured_rms_is_not_speech() -> None:
    session = _session(overlap_silence_rms=0.05)
    loud = {"format": "pcm16", "audio": _pcm16(0.5)}
    quiet = {"format": "pcm16", "audio": _pcm16(0.001)}
    assert overlap_policy.input_looks_like_speech(session, {}, loud) is True
    assert overlap_policy.input_looks_like_speech(session, {}, quiet) is False


def test_undecodable_audio_is_treated_as_speech() -> None:
    """Fail toward listening: dropping real speech is worse than a spurious turn."""
    session = _session()
    assert overlap_policy.input_looks_like_speech(session, {}, {"format": "pcm16", "audio": "!!not base64!!"}) is True


# --------------------------------------------------------------------------- #
# Barge-in                                                                    #
# --------------------------------------------------------------------------- #


def test_an_explicit_client_overlap_action_requests_barge_in() -> None:
    session = _session()
    payload = {"format": "pcm16", "audio": _pcm16(0.5)}
    decision = overlap_policy.decide(session, {"overlap_action": "barge_in"}, payload, auto_responds=True)
    assert decision["action"] == "barge_in"
    assert decision["reason"] == "client_overlap_action"


def test_barge_in_is_deferred_when_the_session_cannot_support_it() -> None:
    session = _session()
    session.capabilities = type(session.capabilities)(supports_barge_in=False)
    payload = {"format": "pcm16", "audio": _pcm16(0.5)}
    decision = overlap_policy.decide(session, {"force_barge_in": True}, payload, auto_responds=True)
    assert decision["action"] != "barge_in"


# --------------------------------------------------------------------------- #
# Force-listen                                                                #
# --------------------------------------------------------------------------- #


def test_a_short_response_create_is_forced_to_listen() -> None:
    """A sub-threshold ack should not take the turn from the model."""
    session = _session(overlap_short_ack_ms=700)
    short = {"response_create": True, "duration_ms": 200}
    long = {"response_create": True, "duration_ms": 1500}
    assert overlap_policy.should_force_listen_for_short_commit(session, short, {}) is True
    assert overlap_policy.should_force_listen_for_short_commit(session, long, {}) is False


def test_force_barge_in_overrides_the_short_ack_rule() -> None:
    session = _session(overlap_short_ack_ms=700)
    event = {"response_create": True, "duration_ms": 200, "force_barge_in": True}
    assert overlap_policy.should_force_listen_for_short_commit(session, event, {}) is False


def test_auto_response_overlap_forces_listen_only_when_auto_responding() -> None:
    event = {"force_listen": True}
    assert overlap_policy.should_force_listen_for_auto_response_overlap(event, {}, auto_responds=True) is True
    assert overlap_policy.should_force_listen_for_auto_response_overlap(event, {}, auto_responds=False) is False


# --------------------------------------------------------------------------- #
# Payload merging                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("speech_first", [True, False], ids=["speech-then-silence", "silence-then-speech"])
def test_merging_two_appends_keeps_speech_if_either_half_was_speech(speech_first: bool) -> None:
    """Matching rates and decodable audio, so the halves really are concatenated.

    Without a ``sample_rate_hz`` on both, ``merge_audio_payloads`` falls back
    to the newer chunk before merging, and the speech flag it returned was the
    second chunk's own -- the test passed without exercising the merge.
    """
    speech = {
        "format": "pcm_f32le",
        "audio": base64.b64encode(b"\x01" * 8).decode("ascii"),
        "sample_rate_hz": 16000,
        "is_speech": True,
    }
    silence = {
        "format": "pcm_f32le",
        "audio": base64.b64encode(b"\x00" * 8).decode("ascii"),
        "sample_rate_hz": 16000,
        "is_speech": False,
    }
    first, second = (speech, silence) if speech_first else (silence, speech)
    merged = overlap_policy.merge_audio_payloads(first, second)
    assert base64.b64decode(merged["audio"]) == base64.b64decode(first["audio"]) + base64.b64decode(second["audio"])
    assert merged["sample_rate_hz"] == 16000
    assert merged["is_speech"] is True


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"format": "pcm_f32le", "audio": "", "sample_rate_hz": 16000}, {"format": "pcm16", "audio": ""}),
        (
            {"format": "pcm_f32le", "audio": "", "sample_rate_hz": 16000},
            {"format": "pcm_f32le", "audio": "", "sample_rate_hz": 24000},
        ),
        (
            {"format": "pcm_f32le", "audio": "!!not base64!!", "sample_rate_hz": 16000},
            {"format": "pcm_f32le", "audio": "", "sample_rate_hz": 16000},
        ),
    ],
    ids=["format-mismatch", "rate-mismatch", "undecodable"],
)
def test_unmergeable_payloads_fall_back_to_the_newer_one(first: dict, second: dict) -> None:
    """Whatever cannot be concatenated keeps the newer chunk and drops the older.

    Worth pinning: the fallback discards ``first`` silently, so a bug that makes
    two mergeable payloads look mismatched loses buffered audio rather than
    raising.
    """
    assert overlap_policy.merge_audio_payloads(first, second) == second


def test_mergeable_payloads_are_concatenated_in_order() -> None:
    head = bytes([1, 2, 3, 4])
    tail = bytes([5, 6, 7, 8])
    first = {"format": "pcm_f32le", "audio": base64.b64encode(head).decode(), "sample_rate_hz": 16000}
    second = {"format": "pcm_f32le", "audio": base64.b64encode(tail).decode(), "sample_rate_hz": 16000}
    merged = overlap_policy.merge_audio_payloads(first, second)
    assert base64.b64decode(merged["audio"]) == head + tail
    assert merged["sample_rate_hz"] == 16000


# --------------------------------------------------------------------------- #
# Server VAD + barge_in_on_speech                                             #
# --------------------------------------------------------------------------- #


def test_server_vad_speech_started_reaches_a_barge_in_decision() -> None:
    """The configuration this PR introduces must decide, not raise.

    Both call sites read the VAD verdict through the module-level
    ``vad_speech_started`` helper while binding its result to a local of the
    same name, which made the name local for the whole scope and raised
    ``UnboundLocalError`` on the call itself. Every ``server_vad`` +
    ``barge_in_on_speech`` append hit it, and the client saw ``internal_error``
    instead of a barge-in. The MiniCPM-o preset is ``listen_only``, so no demo
    exercised this branch.
    """
    session = _session(overlap_policy="barge_in_on_speech")
    payload = {"format": "pcm16", "audio": _pcm16(0.5)}
    event = {"vad": {"speech_started": True}}

    decision = overlap_policy.decide(session, event, payload, auto_responds=True)

    assert decision["action"] == "barge_in"
    assert decision["reason"] == "server_vad_speech_started"
    assert decision["cancel_reason"] == "turn_detected"


def test_server_vad_mid_utterance_keeps_listening_instead_of_barging_in() -> None:
    """``speech_started`` false means the utterance is already running: not a new turn."""
    session = _session(overlap_policy="barge_in_on_speech")
    payload = {"format": "pcm16", "audio": _pcm16(0.5)}
    event = {"vad": {"speech_started": False}}

    decision = overlap_policy.decide(session, event, payload, auto_responds=True)

    assert decision["action"] != "barge_in"


def test_barge_in_on_speech_without_vad_still_barges_in_on_speech() -> None:
    """No VAD verdict: the policy falls back to its own speech classification."""
    session = _session(overlap_policy="barge_in_on_speech")
    payload = {"format": "pcm16", "audio": _pcm16(0.5)}

    decision = overlap_policy.decide(session, {}, payload, auto_responds=True)

    assert decision["action"] == "barge_in"
    assert decision["reason"] == "barge_in_on_speech"
