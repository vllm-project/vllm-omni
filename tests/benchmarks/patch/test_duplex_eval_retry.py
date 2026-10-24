# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-DuplexEval retry accounting: per-attempt latency and retry visibility."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import GenerateSampleResult
from vllm_omni.benchmarks.duplex.omni_duplex_eval_serving import OmniDuplexEvalSampleRequest
from vllm_omni.benchmarks.patch import patch

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

_PREPARE_S = 3.0
_STALL_S = 40.0
_SESSION_S = 2.0


class _FakeClock:
    """perf_counter that only moves when the test advances it."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _sample_request() -> OmniDuplexEvalSampleRequest:
    return OmniDuplexEvalSampleRequest(
        prompt="",
        prompt_len=1,
        expected_output_len=1,
        multi_modal_data=None,
        sample=SimpleNamespace(id="s1", split="dev"),
        response_root=Path("response-root"),
    )


def _request() -> RequestFuncInput:
    request = RequestFuncInput(
        model="test-model",
        model_name="test-model",
        prompt="",
        api_url="ws://localhost:8000/v1/realtime",
        prompt_len=1,
        output_len=1,
        request_id="rid-1",
    )
    request.duplex_eval_sample = _sample_request()
    return request


def _session_result(metadata: dict[str, object] | None = None, error: str = "") -> GenerateSampleResult:
    return GenerateSampleResult(
        output=Path("response-root/dev/s1.json"),
        request_metrics=[],
        session_metrics={"ttft_ms": [120.0]},
        timed_sentences=[{"sentence": "hello"}],
        timed_chunks=[],
        # A non-VAD session commits its input, which is what makes a missing
        # response a stall rather than a server VAD that never opened a turn.
        metadata={"response_done": True, "input_committed": True, **(metadata or {})},
        output_tokens=1,
        audio_bytes=480,
        audio_sample_rate=24_000,
        error=error,
    )


def _install_fakes(monkeypatch, clock, generate_sample, *, prepare_s: float = _PREPARE_S) -> None:
    def prepare(sample):
        clock.advance(prepare_s)
        return SimpleNamespace(prepared=True)

    monkeypatch.setattr(patch, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    monkeypatch.setattr(patch, "prepare_duplex_eval_sample", prepare)
    monkeypatch.setattr("vllm_omni.benchmarks.duplex.omni_duplex_eval_runner.generate_sample", generate_sample)


async def test_latency_covers_only_the_attempt_that_succeeded(monkeypatch) -> None:
    """A discarded attempt must not inflate the sample's latency."""
    clock = _FakeClock()
    second = _session_result()
    attempts: list[int] = []

    async def generate_sample(*_args, **_kwargs):
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            clock.advance(_STALL_S)  # the whole per-request timeout, burned
            raise asyncio.TimeoutError
        clock.advance(_SESSION_S)
        return second

    _install_fakes(monkeypatch, clock, generate_sample)
    output = await patch._async_request_duplex_eval(_request(), pbar=None)

    assert attempts == [1, 2]
    assert output.success is True
    assert output.duplex_retries == 1
    assert second.metadata["retry_count"] == 1
    # Preparation stays out of the measurement; so does the failed attempt.
    assert output.latency == pytest.approx(_SESSION_S)
    assert output.latency < _STALL_S


async def test_transport_stall_is_retried_once_and_counted(monkeypatch) -> None:
    clock = _FakeClock()
    stalled = _session_result(
        {"drain_timeout": "transport idle for 30.0s"},
        error="transport idle for 30.0s",
    )
    recovered = _session_result()
    returned = iter([stalled, recovered])

    async def generate_sample(*_args, **_kwargs):
        clock.advance(_SESSION_S)
        return next(returned)

    _install_fakes(monkeypatch, clock, generate_sample)
    output = await patch._async_request_duplex_eval(_request(), pbar=None)

    assert output.duplex_retries == 1
    assert next(returned, None) is None  # exactly two attempts, no third
    assert recovered.metadata["retry_count"] == 1
    assert output.latency == pytest.approx(_SESSION_S)
    assert output.duplex_session_metrics == recovered.session_metrics


async def test_stall_is_not_retried_beyond_the_last_attempt(monkeypatch) -> None:
    clock = _FakeClock()
    results = [_session_result({"drain_timeout": "transport idle"}, error="transport idle") for _ in range(2)]
    returned = iter(results)

    async def generate_sample(*_args, **_kwargs):
        clock.advance(_SESSION_S)
        return next(returned)

    _install_fakes(monkeypatch, clock, generate_sample)
    output = await patch._async_request_duplex_eval(_request(), pbar=None)

    assert output.duplex_retries == 1
    assert next(returned, None) is None
    # The last attempt is the reported one, stall recorded and all.
    assert output.duplex_session_metrics is results[-1].session_metrics
    assert results[-1].metadata["retry_count"] == 1
    assert results[0].metadata.get("retry_count") is None


async def test_clean_sample_records_no_retry(monkeypatch) -> None:
    clock = _FakeClock()
    result = _session_result()

    async def generate_sample(*_args, **_kwargs):
        clock.advance(_SESSION_S)
        return result

    _install_fakes(monkeypatch, clock, generate_sample)
    output = await patch._async_request_duplex_eval(_request(), pbar=None)

    assert output.duplex_retries == 0
    assert result.metadata["retry_count"] == 0
    assert output.latency == pytest.approx(_SESSION_S)


def test_retry_tally_aggregates_per_sample_counts() -> None:
    outputs = []
    for retries in (0, 1, 0, 2):
        output = patch.MixRequestFuncOutput()
        output.duplex_retries = retries
        outputs.append(output)

    assert patch._duplex_retry_tally(outputs) == {"duplex_retried_samples": 2, "duplex_retry_count": 3}
    assert patch._duplex_retry_tally([SimpleNamespace()]) == {
        "duplex_retried_samples": 0,
        "duplex_retry_count": 0,
    }


async def test_missing_response_is_not_retried(monkeypatch) -> None:
    """A session that ends without a completed response is reported, not retried.

    In server-turn mode a VAD session can legitimately produce no turn, and in
    commit mode a missing response always comes with a ``drain_timeout`` (which
    the retry path below covers).
    """
    clock = _FakeClock()
    silent = _session_result({"response_done": False, "input_committed": False})
    returned = iter([silent])

    async def generate_sample(*_args, **_kwargs):
        clock.advance(_SESSION_S)
        return next(returned)

    _install_fakes(monkeypatch, clock, generate_sample)
    output = await patch._async_request_duplex_eval(_request(), pbar=None)

    assert output.duplex_retries == 0
    assert next(returned, None) is None
    assert output.success is False
    assert output.error == "no completed response"


@pytest.mark.parametrize(
    ("timeout_key", "timeout_text"),
    [
        ("drain_timeout", "Timed out waiting for response.done"),
        ("close_timeout", "Timed out closing the duplex session after 20s"),
    ],
)
async def test_server_error_with_a_transport_timeout_is_not_retried(monkeypatch, timeout_key, timeout_text) -> None:
    """A reported server failure stays a single attempt, whichever timeout rode with it.

    ``generate_sample`` folds both ``drain_timeout`` and ``close_timeout`` into
    ``result.error``, so a retry policy has to look at the separately recorded
    ``server_errors`` entry: a ``session.close`` that never lands looks exactly
    like a drain that never finished.
    """
    clock = _FakeClock()
    failed = _session_result(
        {
            "response_done": False,
            timeout_key: timeout_text,
            "server_errors": ["{'type': 'error', 'error': {'code': 'invalid_audio'}}"],
        },
        error=f"{{'type': 'error', 'error': {{'code': 'invalid_audio'}}}}; {timeout_text}",
    )
    returned = iter([failed])

    async def generate_sample(*_args, **_kwargs):
        clock.advance(_SESSION_S)
        return next(returned)

    _install_fakes(monkeypatch, clock, generate_sample)
    output = await patch._async_request_duplex_eval(_request(), pbar=None)

    assert output.duplex_retries == 0
    assert next(returned, None) is None
    assert output.success is False
    assert "invalid_audio" in output.error


async def test_reported_error_beside_a_transport_timeout_is_not_retried(monkeypatch) -> None:
    """Any recorded failure besides the timeout keeps the sample at one attempt.

    Not every reported failure lands in ``metadata["server_errors"]``: a
    ``response.done`` whose status is not ``completed`` is recorded in
    ``result.error`` only. It is a deterministic failure too, so the timeout it
    came with must not buy it a second attempt.
    """
    clock = _FakeClock()
    timeout_text = "Timed out waiting for response.done"
    failed = _session_result(
        {"response_done": False, "drain_timeout": timeout_text},
        error=f"Response did not complete: {{'id': 'r1', 'status': 'failed'}}; {timeout_text}",
    )
    returned = iter([failed])

    async def generate_sample(*_args, **_kwargs):
        clock.advance(_SESSION_S)
        return next(returned)

    _install_fakes(monkeypatch, clock, generate_sample)
    output = await patch._async_request_duplex_eval(_request(), pbar=None)

    assert output.duplex_retries == 0
    assert next(returned, None) is None
    assert output.success is False
    assert "did not complete" in output.error
