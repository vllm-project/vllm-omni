from __future__ import annotations

import threading
import time
from enum import Enum, auto


class DeliveryState(Enum):
    QUEUED = auto()
    IN_FLIGHT = auto()
    DELIVERED = auto()
    FAILED = auto()
    CANCELLED = auto()


class DeliveryTimeoutError(TimeoutError):
    pass


class DeliveryCancelledError(RuntimeError):
    pass


_TERMINAL_STATES = {
    DeliveryState.DELIVERED,
    DeliveryState.FAILED,
    DeliveryState.CANCELLED,
}


class DeliveryTicket:
    """Exactly-once completion for one MRv2 connector delivery."""

    def __init__(
        self,
        manager: OmniDeliveryManager,
        *,
        request_id: str,
        put_key: str,
        deadline: float,
    ) -> None:
        self._manager = manager
        self.request_id = request_id
        self.put_key = put_key
        self.deadline = deadline
        self._state = DeliveryState.QUEUED
        self._error: BaseException | None = None
        self._event = threading.Event()

    @property
    def state(self) -> DeliveryState:
        with self._manager._lock:
            return self._state

    def mark_in_flight(self) -> bool:
        return self._manager._mark_in_flight(self)

    def set_delivered(self) -> bool:
        return self._manager._set_delivered(self)

    def set_failed(self, error: BaseException) -> bool:
        return self._manager._set_failed(self, error)

    # Compatibility with the connector mixin's completion interface.
    def set_result(self) -> bool:
        return self.set_delivered()

    def set_error(self, error: BaseException) -> bool:
        return self.set_failed(error)

    def wait(self) -> None:
        while True:
            with self._manager._lock:
                state = self._state
                error = self._error
                remaining = self.deadline - time.monotonic()
            if state is DeliveryState.DELIVERED:
                return
            if state in (DeliveryState.FAILED, DeliveryState.CANCELLED):
                assert error is not None
                raise error
            if remaining <= 0:
                self._manager._timeout(self)
                continue
            self._event.wait(timeout=remaining)


class OmniDeliveryManager:
    """MRv2-only connector delivery state machine.

    A timeout or permanent delivery failure quarantines the whole manager.
    This is a stage-fatal condition: no new payload may be accepted after the
    transport can no longer prove ordered, exactly-once completion.
    """

    def __init__(
        self,
        *,
        delivery_timeout_s: float,
        shutdown_timeout_s: float = 5.0,
    ) -> None:
        if delivery_timeout_s <= 0:
            raise ValueError("delivery_timeout_s must be positive")
        if shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        self.delivery_timeout_s = float(delivery_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self._lock = threading.Lock()
        self._tickets: set[DeliveryTicket] = set()
        self._quarantine_error: BaseException | None = None
        self._closed = False

    @property
    def is_quarantined(self) -> bool:
        with self._lock:
            return self._quarantine_error is not None

    def create_ticket(self, *, request_id: str, put_key: str) -> DeliveryTicket:
        with self._lock:
            if self._closed:
                raise RuntimeError("MRv2 delivery manager is closed")
            if self._quarantine_error is not None:
                raise RuntimeError("MRv2 delivery manager is quarantined") from self._quarantine_error
            ticket = DeliveryTicket(
                self,
                request_id=request_id,
                put_key=put_key,
                deadline=time.monotonic() + self.delivery_timeout_s,
            )
            self._tickets.add(ticket)
            return ticket

    def quarantine(self, error: BaseException) -> None:
        with self._lock:
            self._quarantine_locked(error, primary_ticket=None)

    def shutdown(self, cause: BaseException | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._quarantine_error is None:
                self._quarantine_error = cause or RuntimeError("MRv2 stage shutdown")
            for ticket in tuple(self._tickets):
                if ticket._state in _TERMINAL_STATES:
                    continue
                message = f"connector delivery cancelled for request {ticket.request_id!r}, key {ticket.put_key!r}"
                if cause is not None:
                    message += f": {cause}"
                self._finish_locked(
                    ticket,
                    DeliveryState.CANCELLED,
                    DeliveryCancelledError(message),
                )

    def _mark_in_flight(self, ticket: DeliveryTicket) -> bool:
        with self._lock:
            if ticket._state is not DeliveryState.QUEUED:
                return False
            ticket._state = DeliveryState.IN_FLIGHT
            return True

    def _set_delivered(self, ticket: DeliveryTicket) -> bool:
        with self._lock:
            if ticket._state in _TERMINAL_STATES:
                return False
            return self._finish_locked(ticket, DeliveryState.DELIVERED, None)

    def _set_failed(self, ticket: DeliveryTicket, error: BaseException) -> bool:
        with self._lock:
            if ticket._state in _TERMINAL_STATES:
                return False
            self._quarantine_locked(error, primary_ticket=ticket)
            return True

    def _timeout(self, ticket: DeliveryTicket) -> bool:
        error = DeliveryTimeoutError(
            "connector delivery timed out after "
            f"{self.delivery_timeout_s:.3f}s for request "
            f"{ticket.request_id!r}, key {ticket.put_key!r}"
        )
        with self._lock:
            if ticket._state in _TERMINAL_STATES:
                return False
            self._quarantine_locked(error, primary_ticket=ticket)
            return True

    def _quarantine_locked(
        self,
        error: BaseException,
        *,
        primary_ticket: DeliveryTicket | None,
    ) -> None:
        if self._quarantine_error is None:
            self._quarantine_error = error
        stage_error = self._quarantine_error
        for ticket in tuple(self._tickets):
            if ticket._state in _TERMINAL_STATES:
                continue
            ticket_error = error if ticket is primary_ticket else stage_error
            self._finish_locked(ticket, DeliveryState.FAILED, ticket_error)

    def _finish_locked(
        self,
        ticket: DeliveryTicket,
        state: DeliveryState,
        error: BaseException | None,
    ) -> bool:
        if ticket._state in _TERMINAL_STATES:
            return False
        ticket._state = state
        ticket._error = error
        self._tickets.discard(ticket)
        ticket._event.set()
        return True
