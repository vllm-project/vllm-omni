from __future__ import annotations

import threading

import pytest

from vllm_omni.worker_v2.delivery import (
    DeliveryCancelledError,
    DeliveryState,
    DeliveryTimeoutError,
    OmniDeliveryManager,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_delivery_timeout_quarantines_manager_and_rejects_new_tickets() -> None:
    manager = OmniDeliveryManager(delivery_timeout_s=0.02)
    ticket = manager.create_ticket(request_id="req-0", put_key="req-0_0_0")

    with pytest.raises(DeliveryTimeoutError, match="req-0_0_0"):
        ticket.wait()

    assert ticket.state is DeliveryState.FAILED
    assert manager.is_quarantined
    with pytest.raises(RuntimeError, match="quarantined"):
        manager.create_ticket(request_id="req-1", put_key="req-1_0_0")


def test_delivery_shutdown_cancels_inflight_waiter_exactly_once() -> None:
    manager = OmniDeliveryManager(delivery_timeout_s=10.0)
    ticket = manager.create_ticket(request_id="req-0", put_key="req-0_0_0")
    errors: list[BaseException] = []

    def wait_for_ticket() -> None:
        try:
            ticket.wait()
        except BaseException as error:
            errors.append(error)

    waiter = threading.Thread(target=wait_for_ticket)
    waiter.start()
    manager.shutdown(RuntimeError("stage shutdown"))
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DeliveryCancelledError)
    assert ticket.state is DeliveryState.CANCELLED
    assert not ticket.set_delivered()
