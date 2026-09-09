# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import threading
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.engine.test_native_chunk_wakeup import _engine
from vllm_omni.engine.generation_completion import GenerationCompletionObserver

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class PendingOutput:
    def __init__(self, fail=False):
        self.release = threading.Event()
        self.waiting = threading.Event()
        self.fail = fail
        self.async_output = SimpleNamespace(copy_event=self)
        self.result = Mock(return_value=object())

    def done(self):
        return False

    def query(self):
        return False

    def synchronize(self):
        self.waiting.set()
        assert self.release.wait(3)
        if self.fail:
            raise RuntimeError("device error")


@pytest.mark.parametrize("fail", [False, True])
def test_completion_is_visible_before_notification_even_on_failure(fail):
    future = PendingOutput(fail)
    visible = []
    wake = threading.Event()
    observer = GenerationCompletionObserver(lambda: (visible.append(observer.ready(future)), wake.set()))
    try:
        assert not observer.ready(future)
        assert future.waiting.wait(2)
        future.release.set()
        assert wake.wait(2)
        assert visible == [True]
        future.result.assert_not_called()
        if fail:
            with pytest.raises(RuntimeError, match="device error"):
                observer.consumed(future)
        else:
            observer.consumed(future)
    finally:
        future.release.set()
        observer.close()


def test_submission_returns_to_input_loop_until_oldest_output_completes():
    engine = _engine()
    engine.batch_queue = deque()
    engine.batch_queue_size = 2
    engine.is_ec_consumer = True
    engine.is_pooling_model = False
    engine._should_throttle_prefills = lambda: False
    engine.log_error_detail = lambda _: nullcontext()
    engine.capture_iteration_details = lambda _: nullcontext(None)
    engine._attach_iteration_details = Mock()
    engine._process_aborts_queue = Mock()
    future = PendingOutput()
    output = SimpleNamespace(total_num_scheduled_tokens=1)
    engine.scheduler.schedule = Mock(return_value=output)
    engine.scheduler.has_requests = Mock(return_value=True)
    engine.scheduler.update_from_output = Mock(return_value={})
    engine.scheduler.get_grammar_bitmask = Mock(return_value=None)
    exec_future = Future()
    exec_future.set_result(None)
    engine.model_executor = SimpleNamespace(
        execute_model=Mock(return_value=exec_future), sample_tokens=Mock(return_value=future)
    )
    wake = threading.Event()
    observer = GenerationCompletionObserver(wake.set)
    engine._omni_completion_observer = observer
    try:
        assert engine.step_with_batch_queue() == (None, True)
        future.result.assert_not_called()
        engine.model_executor.sample_tokens.assert_called_once_with(None, non_block=True)
        engine.scheduler.has_requests.return_value = False
        assert not engine.has_work()
        # New ready work can enter without consuming the pending output.
        engine.scheduler.has_requests.return_value = True
        assert engine.has_work()
        engine.scheduler.has_requests.return_value = False
        future.release.set()
        assert wake.wait(2)
        assert engine.has_work()
        assert engine.step_with_batch_queue() == ({}, False)
        future.result.assert_called_once()
        engine._process_aborts_queue.assert_called_once()
        assert not engine.batch_queue
    finally:
        future.release.set()
        observer.close()


def test_full_queue_does_not_submit_or_consume_unfinished_output():
    engine = _engine()
    future = PendingOutput()
    engine.batch_queue = deque([(future, object(), future)])
    engine.batch_queue_size = 1
    engine.scheduler.has_requests = lambda: True
    engine.scheduler.schedule = Mock()
    observer = GenerationCompletionObserver(lambda: None)
    engine._omni_completion_observer = observer
    try:
        assert not engine.has_work()
        assert engine.step_with_batch_queue() == (None, False)
        engine.scheduler.schedule.assert_not_called()
        future.result.assert_not_called()
    finally:
        future.release.set()
        observer.close()


def test_completed_or_unknown_future_preserves_owner_consumption():
    observer = GenerationCompletionObserver(lambda: None)
    try:
        future = Future()
        assert observer.ready(future)
        future.set_exception(RuntimeError("execute failed"))
        assert observer.ready(future)
        with pytest.raises(RuntimeError, match="execute failed"):
            future.result()
    finally:
        observer.close()


def test_later_completion_cannot_overtake_oldest_and_aborts_precede_update():
    engine = _engine()
    oldest = Future()
    newest = Future()
    newest.set_result("new")
    old_schedule, new_schedule = object(), object()
    engine.batch_queue = deque([(newest, new_schedule, newest), (oldest, old_schedule, oldest)])
    engine.batch_queue_size = 2
    engine._omni_completion_observer = SimpleNamespace(ready=lambda f: f.done(), consumed=lambda f: None)
    engine.scheduler.has_requests = lambda: False
    engine.capture_iteration_details = lambda _: nullcontext(None)
    engine.log_error_detail = lambda _: nullcontext()
    engine._attach_iteration_details = Mock()
    calls = []
    engine._process_aborts_queue = lambda: calls.append("abort")
    engine.scheduler.update_from_output = lambda s, o: calls.append(o)
    assert engine.step_with_batch_queue() == (None, False)
    assert not calls
    oldest.set_result("old")
    engine.step_with_batch_queue()
    engine.step_with_batch_queue()
    assert calls == ["abort", "old", "abort", "new"]
    assert not engine.batch_queue
