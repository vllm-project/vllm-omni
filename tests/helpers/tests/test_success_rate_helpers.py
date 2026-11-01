# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace

import pytest

from tests.helpers import assertions
from tests.helpers.assertions import (
    AUDIO_MISMATCH_MESSAGE,
    SuccessRateGate,
    assert_omni_response,
    collect_at_success_rate,
    is_quality_failure,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_GATE = SuccessRateGate(min_successes=7, concurrency=2)


def _fake_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(success=True, text_content=text, audio_content=None, audio_bytes=b"fake-wav")


def _omni_response_failure(monkeypatch, *, text: str, transcript: str) -> AssertionError:
    """Drive the real assert_omni_response to failure and hand back what it raised."""
    monkeypatch.setattr(assertions, "convert_audio_bytes_to_text", lambda *a, **kw: transcript)
    request_config = {"modalities": ["text", "audio"], "key_words": {"text": ["london"]}}
    with pytest.raises(AssertionError) as excinfo:
        assert_omni_response(_fake_response(text), request_config, run_level="full_model")
    return excinfo.value


def test_real_audio_mismatch_is_tolerated(monkeypatch):
    # Ties the classifier to the actual producer: if assert_omni_response's wording drifts,
    # this fails instead of the gate silently turning every quality failure into a hard one.
    assert is_quality_failure(_omni_response_failure(monkeypatch, text="London", transcript="런던"))


def test_real_keyword_failure_is_not_tolerated(monkeypatch):
    assert not is_quality_failure(_omni_response_failure(monkeypatch, text="Paris", transcript="London"))


@pytest.mark.parametrize("exc", [RuntimeError(AUDIO_MISMATCH_MESSAGE), ValueError("boom")])
def test_non_assertion_failures_are_not_tolerated(exc):
    # A transport error carrying the same text is still a serving failure, not a sample.
    assert not is_quality_failure(exc)


class _ScriptedSampler:
    """One scripted outcome per call: raised if it is an exception, else returned."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = outcomes
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self):
        with self._lock:
            outcome = self._outcomes[self.calls]
            self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _mismatch(detail: str) -> AssertionError:
    return AssertionError(f"{AUDIO_MISMATCH_MESSAGE} (transcript={detail!r})")


def test_tolerated_failures_are_counted_and_every_sample_is_sent():
    sampler = _ScriptedSampler(["ok"] * 7 + [_mismatch("a"), _mismatch("b"), _mismatch("c")])

    responses = collect_at_success_rate(sampler, _GATE, request_num=10, report=lambda _: None)

    # No early exit once the gate is already satisfied: it would bias the measured rate.
    assert sampler.calls == 10
    assert responses == ["ok"] * 7


def test_gate_failure_reports_the_count_and_every_failing_sample():
    outcomes = ["ok"] * 6 + [_mismatch("first"), _mismatch("second"), _mismatch("third"), _mismatch("fourth")]

    with pytest.raises(AssertionError) as excinfo:
        collect_at_success_rate(_ScriptedSampler(outcomes), _GATE, request_num=10, report=lambda _: None)

    message = str(excinfo.value)
    assert "6/10" in message and "7/10 gate" in message
    assert all(detail in message for detail in ("first", "second", "third", "fourth"))


def test_serving_failure_is_not_absorbed_into_the_budget():
    sampler = _ScriptedSampler(["ok", AssertionError("No audio output is generated"), *["ok"] * 8])

    with pytest.raises(AssertionError, match="No audio output"):
        collect_at_success_rate(sampler, _GATE, request_num=10, report=lambda _: None)

    # Stopped on the serving failure rather than spending the remaining samples.
    assert sampler.calls < 10


def test_unreachable_gate_is_rejected_before_sending():
    sampler = _ScriptedSampler(["ok"] * 3)

    with pytest.raises(ValueError, match="can never pass"):
        collect_at_success_rate(sampler, _GATE, request_num=3, report=lambda _: None)

    assert sampler.calls == 0
