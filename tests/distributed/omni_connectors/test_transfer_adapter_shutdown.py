# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.distributed.omni_connectors.transfer_adapter import base as adapter_module
from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_shutdown_stops_queued_io_after_inflight_operation(direction):
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    processed = []

    def transfer(request_id):
        processed.append(request_id)
        if request_id == "0":
            entered.set()
            assert release.wait(timeout=5)
        return True

    class Adapter(OmniTransferAdapterBase):
        def _send_single_request(self, task):
            transfer(task["request"].request_id)

        def _poll_single_request(self, request):
            return transfer(request.request_id)

    adapter = Adapter(None)
    adapter.connector = SimpleNamespace(close=closed.set, reap_consumed=lambda: None)
    requests = [SimpleNamespace(request_id=str(i)) for i in range(64)]
    condition = adapter._save_cond if direction == "send" else adapter._recv_cond
    with condition:
        if direction == "send":
            adapter._pending_save_reqs.extend({"request": request} for request in requests)
        else:
            adapter._pending_load_reqs.extend(requests)
        condition.notify()
    stopper = threading.Thread(target=adapter.shutdown)
    try:
        assert entered.wait(timeout=5)
        stopper.start()
        assert adapter.stop_event.wait(timeout=5)
        assert not closed.is_set()
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(timeout=5)
        adapter.shutdown()
        adapter.recv_thread.join(timeout=5)
        adapter.save_thread.join(timeout=5)
    assert not stopper.is_alive()
    assert not adapter.recv_thread.is_alive()
    assert not adapter.save_thread.is_alive()
    assert closed.is_set()
    assert processed == ["0"]


def test_shutdown_waits_for_inflight_put_before_closing_connector():
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    events = []

    class Adapter(OmniTransferAdapterBase):
        def _send_single_request(self, task):
            self.connector.put(task)

    def put(task):
        entered.set()
        assert release.wait(timeout=5)
        events.append("put_finished")

    def close():
        events.append("closed")
        closed.set()

    adapter = Adapter(None)
    adapter.connector = SimpleNamespace(put=put, close=close, reap_consumed=lambda: None)
    with adapter._save_cond:
        adapter._pending_save_reqs.append({})
        adapter._save_cond.notify()
    stopper = threading.Thread(target=adapter.shutdown)
    try:
        assert entered.wait(timeout=5)
        stopper.start()
        assert adapter.stop_event.wait(timeout=5)
        assert not closed.wait(timeout=0.1)
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(timeout=5)
        adapter.shutdown()
        adapter.recv_thread.join(timeout=5)
        adapter.save_thread.join(timeout=5)
    assert not stopper.is_alive()
    assert not adapter.recv_thread.is_alive()
    assert not adapter.save_thread.is_alive()
    assert events.index("put_finished") < events.index("closed")


def test_shutdown_stops_idle_threads_and_closes_connector():
    closed = threading.Event()
    adapter = OmniTransferAdapterBase(None)
    adapter.connector = SimpleNamespace(close=closed.set, reap_consumed=lambda: None)
    adapter.shutdown()
    assert not adapter.recv_thread.is_alive()
    assert not adapter.save_thread.is_alive()
    assert closed.is_set()


@pytest.mark.parametrize("recv_alive, save_alive", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("elapsed, remaining", [(4.0, 1.0), (5.0, 0.0), (6.0, 0.0)])
def test_shutdown_does_not_close_connector_while_a_thread_is_stuck(
    monkeypatch, recv_alive, save_alive, elapsed, remaining
):
    # The receive join consumes the budget; the sender must not get a fresh one.
    ticks = iter([100.0, 100.0, 100.0 + elapsed])
    monkeypatch.setattr(adapter_module, "time", SimpleNamespace(monotonic=lambda: next(ticks)), raising=False)
    adapter = OmniTransferAdapterBase.__new__(OmniTransferAdapterBase)
    adapter.stop_event = threading.Event()
    adapter._recv_cond = threading.Condition()
    adapter._save_cond = threading.Condition()
    adapter.recv_thread = Mock()
    adapter.recv_thread.is_alive.return_value = recv_alive
    adapter.save_thread = Mock()
    adapter.save_thread.is_alive.return_value = save_alive
    adapter.connector = Mock()
    adapter.shutdown()
    adapter.recv_thread.join.assert_called_once_with(timeout=5)
    adapter.save_thread.join.assert_called_once_with(timeout=remaining)
    adapter.connector.close.assert_not_called()
