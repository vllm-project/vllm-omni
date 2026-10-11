# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Talker yields after releasing a response's first codec window."""

from __future__ import annotations

import pytest

from tests.model_executor.stage_input_processors.test_minicpmo_4_5_async_chunk import (
    _delta,
    _duplex_delta,
    _manager,
    _request,
)
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.stage_input_processors import minicpmo_4_5_omni as processor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "v2,native,finished", [(True, True, False), (False, True, False), (True, False, False), (True, True, True)]
)
def test_only_mrv2_native_first_stream_window_yields(monkeypatch, mocker, v2, native, finished):
    monkeypatch.setattr(processor, "_FIRST_WINDOW_YIELD_UNTIL", {})
    monkeypatch.setattr(processor.time, "monotonic", lambda: 10.0)
    manager = _manager()
    manager.config = mocker.Mock(spec=OmniModelConfig, use_v2_model_runner=v2)
    manager.connector.config["extra"]["initial_codec_chunk_frames"] = 10

    def delta(*codes: int) -> dict:
        return _duplex_delta(*codes) if native else _delta(*codes)

    request = _request("external", "internal")
    assert processor.tts2code2wav_async_chunk(manager, delta(*range(9)), request, False) is None
    assert processor.scheduling_hold_request_ids(10.0) == set()
    assert processor.tts2code2wav_async_chunk(manager, delta(9), request, finished) is not None
    expected = {"internal"} if v2 and native and not finished else set()
    assert processor.scheduling_hold_request_ids(10.0) == expected
    # A later window must not extend the first-window deadline.
    manager.put_req_chunk["external"] += 1
    monkeypatch.setattr(processor.time, "monotonic", lambda: 11.0)
    assert processor.tts2code2wav_async_chunk(manager, delta(*range(10, 35)), request, False) is not None
    assert processor.scheduling_hold_request_ids(11.0) == set()


def test_sender_waits_for_scheduler_expiry_snapshot(monkeypatch, mocker):
    from threading import Event, Lock, Thread

    original_lock = Lock()

    class CheckedDeadlines(dict):
        def items(self):
            assert original_lock.locked(), "scheduler expiry bypassed the shared lock"
            return super().items()

    table = CheckedDeadlines({"expired": 0.0})
    monkeypatch.setattr(processor, "_FIRST_WINDOW_YIELD_UNTIL", table)
    manager = _manager()
    manager.config = mocker.Mock(spec=OmniModelConfig, use_v2_model_runner=True)
    manager.connector.config["extra"]["initial_codec_chunk_frames"] = 10
    blocked = Event()
    entered = Event()
    failures = []

    class ObservedLock:
        def __enter__(self):
            entered.set()
            if not original_lock.acquire(blocking=False):
                blocked.set()
                original_lock.acquire()
            return self

        def __exit__(self, *args):
            original_lock.release()

    monkeypatch.setattr(processor, "_FIRST_WINDOW_YIELD_LOCK", ObservedLock(), raising=False)

    def publish():
        try:
            assert (
                processor.tts2code2wav_async_chunk(
                    manager, _duplex_delta(*range(10)), _request("external", "new"), False
                )
                is not None
            )
        except BaseException as exc:
            failures.append(exc)

    thread = Thread(target=publish)
    try:
        with processor._FIRST_WINDOW_YIELD_LOCK:
            entered.clear()
            thread.start()
            assert entered.wait(5), "first-window publisher bypassed the shared lock"
            assert blocked.wait(5)
            assert table == {"expired": 0.0}
            table.pop("expired")
    finally:
        if thread.ident is not None:
            thread.join(5)
    assert not thread.is_alive()
    assert not failures, failures
    assert processor.scheduling_hold_request_ids(0.0) == {"new"}


def test_hold_ids_expire_at_their_deadline(monkeypatch):
    monkeypatch.setattr(processor, "_FIRST_WINDOW_YIELD_UNTIL", {"a": 10.0, "b": 20.0})
    assert processor.scheduling_hold_request_ids(5.0) == {"a", "b"}
    assert processor.scheduling_hold_request_ids(10.0) == {"b"}
    assert processor.scheduling_hold_request_ids(25.0) == set()
    assert processor._FIRST_WINDOW_YIELD_UNTIL == {}
