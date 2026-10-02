# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc, _bind_first_audio_sink

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "first_decoder,stream_decoder,stream_first_audio,expected",
    [(True, False, False, True), (False, True, False, False), (False, True, True, True), (False, False, True, False)],
)
def test_first_audio_sink_preserves_first_decoder_and_opts_in_stream_decoder(
    mocker,
    first_decoder,
    stream_decoder,
    stream_first_audio,
    expected,
):
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.driver_worker = mocker.Mock()
    runner = executor.driver_worker.worker.model_runner
    runner.model.first_frame_decoder = object() if first_decoder else None
    runner.model.stream_decoder = object() if stream_decoder else None
    runner.model.stream_first_audio = stream_first_audio
    assert _bind_first_audio_sink(executor, mocker.Mock(), mocker.Mock()) is expected
    assert runner.model_state.set_first_audio_sink.call_count == int(expected)


def test_preprocess_add_request_preserves_omni_fields():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
    )
    scheduler_request = SimpleNamespace()

    with patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        return_value=(scheduler_request, 3),
    ):
        result, current_wave = engine.preprocess_add_request(request)

    assert result is scheduler_request
    assert current_wave == 3
    assert result.external_req_id == "external"
    assert result.additional_information == {"conditioning": "payload"}


# --- event-driven idle wait (VLLM_OMNI_STAGE_IDLE_WAIT_S) -------------------------------------


import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
from vllm.v1.engine import EngineCoreRequestType  # noqa: E402


def _idle_engine(*, idle_wait_s: float, step_result=({}, False), has_requests=True):
    """A StageEngineCoreProc shell with just the loop state the idle wait touches."""
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.input_queue = queue.Queue()
    engine.output_queue = queue.Queue()
    engine.batch_queue = None
    engine._idle_wait_s = idle_wait_s
    engine._last_step_idle = False
    engine.scheduler = MagicMock()
    engine.scheduler.has_requests.return_value = has_requests
    engine.step_fn = MagicMock(return_value=step_result)
    engine.post_step = MagicMock()
    engine.is_running = MagicMock(return_value=True)
    engine.handled = []
    engine._handle_client_request = lambda request_type, request: engine.handled.append(request_type)
    return engine


def test_idle_step_marks_the_loop_parked_without_sleeping():
    engine = _idle_engine(idle_wait_s=5.0)
    with patch("vllm_omni.engine.stage_engine_core_proc.time.sleep") as sleep:
        assert engine._process_engine_step() is False
    assert engine._last_step_idle is True
    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("step_result", "has_requests"),
    [
        (({}, True), True),  # executed: not idle
        (({0: object()}, False), True),  # produced output: not idle
        (({}, False), False),  # no requests: vLLM's own blocking get applies
        (({0: SimpleNamespace(outputs=[], finished_requests={"r"})}, False), True),  # finished a request
    ],
)
def test_busy_or_empty_steps_do_not_park(step_result, has_requests):
    engine = _idle_engine(idle_wait_s=5.0, step_result=step_result, has_requests=has_requests)
    engine._process_engine_step()
    assert engine._last_step_idle is False


def test_stats_only_step_output_still_parks():
    # log_stats attaches scheduler stats to an otherwise empty output every step.
    stats_only = SimpleNamespace(outputs=[], scheduler_stats=object(), finished_requests=None, utility_output=None)
    engine = _idle_engine(idle_wait_s=5.0, step_result=({0: stats_only}, False))
    engine._process_engine_step()
    assert engine._last_step_idle is True
    assert engine.output_queue.get_nowait() == (0, stats_only)


def test_parked_loop_wakes_on_chunk_ready_before_the_fallback():
    engine = _idle_engine(idle_wait_s=5.0)
    engine._last_step_idle = True
    timer = threading.Timer(0.05, engine._post_idle_wakeup)
    timer.start()
    start = time.monotonic()
    with patch.object(EngineCoreProc, "_process_input_queue") as upstream:
        engine._process_input_queue()
    elapsed = time.monotonic() - start
    timer.join()
    assert elapsed < 2.0
    assert engine.handled == [EngineCoreRequestType.WAKEUP]
    assert engine._last_step_idle is False
    upstream.assert_called_once()


def test_parked_loop_falls_back_after_the_bound_when_no_wakeup_arrives():
    engine = _idle_engine(idle_wait_s=0.05)
    engine._last_step_idle = True
    start = time.monotonic()
    with patch.object(EngineCoreProc, "_process_input_queue"):
        engine._process_input_queue()
    elapsed = time.monotonic() - start
    assert 0.04 <= elapsed < 2.0
    assert engine.handled == []
    # The next loop pass steps once (deadlines, missed hints) instead of parking again.
    assert engine._last_step_idle is False


def test_loop_does_not_park_with_pending_input_or_batches():
    engine = _idle_engine(idle_wait_s=5.0)
    engine._last_step_idle = True
    engine.batch_queue = [object()]
    start = time.monotonic()
    with patch.object(EngineCoreProc, "_process_input_queue"):
        engine._process_input_queue()
    assert time.monotonic() - start < 1.0


def test_idle_wait_off_keeps_the_upstream_step():
    engine = _idle_engine(idle_wait_s=0.0)
    with patch.object(EngineCoreProc, "_process_engine_step", return_value=False) as upstream:
        assert engine._process_engine_step() is False
    upstream.assert_called_once()
    engine.step_fn.assert_not_called()


def test_init_idle_wait_binds_the_chunk_ready_wakeup(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_STAGE_IDLE_WAIT_S", "0.2")
    engine = _idle_engine(idle_wait_s=0.0)
    adapter = MagicMock()
    engine.scheduler._batch_window_s = 0.05
    engine.scheduler.chunk_transfer_adapter = adapter
    engine._init_idle_wait()
    assert engine._idle_wait_s == pytest.approx(0.05)  # clamped to the admission window
    adapter.set_chunk_ready_callback.assert_called_once()
    callback = adapter.set_chunk_ready_callback.call_args.args[0]
    callback()
    assert engine.input_queue.get_nowait() == (EngineCoreRequestType.WAKEUP, None)


def test_init_idle_wait_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_STAGE_IDLE_WAIT_S", raising=False)
    engine = _idle_engine(idle_wait_s=1.0)
    adapter = MagicMock()
    engine.scheduler.chunk_transfer_adapter = adapter
    engine._init_idle_wait()
    assert engine._idle_wait_s == 0.0
    adapter.set_chunk_ready_callback.assert_not_called()


def test_init_idle_wait_refuses_a_stage_without_chunk_wakeup(monkeypatch):
    # e.g. the native MRv2 data plane: every chunk would wait for the fallback bound.
    monkeypatch.setenv("VLLM_OMNI_STAGE_IDLE_WAIT_S", "0.05")
    engine = _idle_engine(idle_wait_s=0.0)
    engine.scheduler.chunk_transfer_adapter = None
    engine._init_idle_wait()
    assert engine._idle_wait_s == 0.0


# --- deadline batching: park bounded by the next release ----------------------------------------


def test_park_is_bounded_by_the_schedulers_next_release():
    engine = _idle_engine(idle_wait_s=5.0)
    engine.scheduler.next_release_in = MagicMock(return_value=0.05)
    engine._last_step_idle = True
    start = time.monotonic()
    with patch.object(EngineCoreProc, "_process_input_queue"):
        engine._process_input_queue()
    elapsed = time.monotonic() - start
    assert 0.04 <= elapsed < 2.0
    assert engine.handled == []


@pytest.mark.parametrize("release_in", [None, MagicMock(), 1, "0.01"])
def test_park_ignores_non_float_release_hints(release_in):
    from vllm_omni.engine.stage_engine_core_proc import _scheduler_next_release_in

    scheduler = MagicMock()
    scheduler.next_release_in = MagicMock(return_value=release_in)
    assert _scheduler_next_release_in(scheduler) is None
    assert _scheduler_next_release_in(object()) is None


def test_release_hint_has_a_floor():
    import vllm_omni.engine.stage_engine_core_proc as proc_module

    engine = _idle_engine(idle_wait_s=5.0)
    engine.scheduler.next_release_in = MagicMock(return_value=0.0)
    engine._last_step_idle = True
    timeouts = []

    def fake_get(timeout):
        timeouts.append(timeout)
        raise queue.Empty

    engine.input_queue.get = fake_get
    with patch.object(EngineCoreProc, "_process_input_queue"):
        engine._process_input_queue()
    assert timeouts == [proc_module._MIN_RELEASE_WAIT_S]
