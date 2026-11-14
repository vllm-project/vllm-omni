# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the startup resource ledger (RFC 9536 01-1)."""

from __future__ import annotations

import logging

import pytest
import torch

from vllm_omni.diffusion.weight_management.errors import LifecycleError
from vllm_omni.diffusion.weight_management.startup import ResourceKind, ResourceLedger

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

_CLOSE_LOG: list[str] = []


class _Recorder:
    def __init__(self, label: str) -> None:
        self.label = label

    def close(self) -> None:
        _CLOSE_LOG.append(self.label)


@pytest.fixture(autouse=True)
def _clear_close_log():
    _CLOSE_LOG.clear()
    yield
    _CLOSE_LOG.clear()


def _register(ledger: ResourceLedger, label: str, kind: ResourceKind) -> object:
    handle = object()
    ledger.register(handle, kind, _Recorder(label).close)
    return handle


def test_resource_kinds_are_closed_in_rfc_order():
    assert [int(kind) for kind in ResourceKind] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [kind.name for kind in ResourceKind] == [
        "EXECUTION",
        "INFLIGHT",
        "HOOKS",
        "DEVICE",
        "REGISTRATION",
        "STAGING",
        "PLACEHOLDER",
        "HOST_STORAGE",
    ]


def test_close_orders_by_kind_then_reverse_registration():
    ledger = ResourceLedger()
    _register(ledger, "host", ResourceKind.HOST_STORAGE)
    _register(ledger, "exec-1", ResourceKind.EXECUTION)
    _register(ledger, "device", ResourceKind.DEVICE)
    _register(ledger, "exec-2", ResourceKind.EXECUTION)
    _register(ledger, "staging", ResourceKind.STAGING)

    ledger.close()

    # Kind ascending; within one kind the most recently registered entry closes first.
    assert _CLOSE_LOG == ["exec-2", "exec-1", "device", "staging", "host"]


def test_duplicate_registration_is_rejected_by_identity():
    ledger = ResourceLedger()
    handle = object()
    ledger.register(handle, ResourceKind.DEVICE, lambda: None)
    with pytest.raises(LifecycleError, match="already registered"):
        ledger.register(handle, ResourceKind.HOOKS, lambda: None)


def test_equal_but_distinct_handles_can_be_registered():
    ledger = ResourceLedger()
    first = [1, 2]
    second = [1, 2]
    ledger.register(first, ResourceKind.DEVICE, lambda: None)
    ledger.register(second, ResourceKind.DEVICE, lambda: None)
    ledger.close()


def test_hand_over_removes_entry_without_closing_it():
    ledger = ResourceLedger()
    kept = _register(ledger, "kept", ResourceKind.DEVICE)
    handed_over = _register(ledger, "handed-over", ResourceKind.PLACEHOLDER)

    ledger.hand_over(handed_over)
    with pytest.raises(LifecycleError, match="not registered"):
        ledger.hand_over(handed_over)

    ledger.close()

    assert _CLOSE_LOG == ["kept"]
    assert handed_over is not kept


def test_transfer_moves_entries_and_leaves_source_empty():
    source = ResourceLedger()
    _register(source, "host", ResourceKind.HOST_STORAGE)
    _register(source, "device", ResourceKind.DEVICE)

    target = source.transfer()

    assert _CLOSE_LOG == []
    source.close()  # empty after transfer; no callbacks run
    assert _CLOSE_LOG == []

    target.close()
    assert _CLOSE_LOG == ["device", "host"]


def test_transfer_target_accepts_new_resources_and_closes_them_in_kind_order():
    source = ResourceLedger()
    _register(source, "host", ResourceKind.HOST_STORAGE)

    target = source.transfer()
    _register(target, "device", ResourceKind.DEVICE)

    target.close()
    # The moved entry keeps its kind, so the new device resource closes first.
    assert _CLOSE_LOG == ["device", "host"]


def test_register_and_transfer_after_transfer_are_rejected():
    source = ResourceLedger()
    source.transfer()

    with pytest.raises(LifecycleError, match="already transferred"):
        source.transfer()
    with pytest.raises(LifecycleError, match="after the ledger was transferred"):
        source.register(object(), ResourceKind.DEVICE, lambda: None)
    with pytest.raises(LifecycleError, match="after the ledger was transferred"):
        source.hand_over(object())


def test_register_after_close_is_rejected():
    ledger = ResourceLedger()
    ledger.close()
    with pytest.raises(LifecycleError, match="after the ledger is closed"):
        ledger.register(object(), ResourceKind.DEVICE, lambda: None)
    with pytest.raises(LifecycleError, match="after the ledger is closed"):
        ledger.transfer()
    with pytest.raises(LifecycleError, match="after the ledger is closed"):
        ledger.hand_over(object())


def test_close_is_idempotent():
    ledger = ResourceLedger()
    _register(ledger, "device", ResourceKind.DEVICE)
    ledger.close()
    ledger.close()
    assert _CLOSE_LOG == ["device"]


def test_close_runs_every_callback_and_reports_first_failure_as_cause(caplog):
    first_failure = RuntimeError("device close failed")
    second_failure = RuntimeError("host close failed")
    calls: list[str] = []

    def fail_with(error: Exception):
        def close() -> None:
            calls.append(str(error))
            raise error

        return close

    ledger = ResourceLedger()
    ledger.register(object(), ResourceKind.EXECUTION, fail_with(first_failure))
    ledger.register(object(), ResourceKind.HOST_STORAGE, fail_with(second_failure))

    with caplog.at_level(logging.ERROR), pytest.raises(LifecycleError) as exc_info:
        ledger.close()

    # Both callbacks ran even though the first one raised.
    assert calls == ["device close failed", "host close failed"]
    assert exc_info.value.__cause__ is first_failure
    assert "2 resource close callback(s) failed" in str(exc_info.value)
    # The failures after the first one are logged instead of being swallowed.
    assert any("host close failed" in record.getMessage() for record in caplog.records)

    # The ledger is closed even after a failure; a second close is a no-op.
    calls.clear()
    ledger.close()
    assert calls == []


def test_close_finishes_cleanup_before_propagating_a_cancellation(caplog):
    """Ctrl-C during cleanup must still release the remaining kinds, exactly once."""
    interrupt = KeyboardInterrupt()
    calls: list[str] = []

    def interrupt_close() -> None:
        calls.append("exec")
        raise interrupt

    def failing_close() -> None:
        calls.append("device")
        raise RuntimeError("device close failed")

    def ok_close() -> None:
        calls.append("host")

    ledger = ResourceLedger()
    ledger.register(object(), ResourceKind.EXECUTION, interrupt_close)
    ledger.register(object(), ResourceKind.DEVICE, failing_close)
    ledger.register(object(), ResourceKind.HOST_STORAGE, ok_close)

    with caplog.at_level(logging.ERROR), pytest.raises(KeyboardInterrupt) as exc_info:
        ledger.close()

    # The interrupt at kind 1 neither skipped kind 4 nor kind 8, and the original
    # exception object is propagated instead of a LifecycleError wrapper.
    assert calls == ["exec", "device", "host"]
    assert exc_info.value is interrupt
    # The ordinary failure that lost the precedence race is still reported.
    assert any("device close failed" in record.getMessage() for record in caplog.records)

    # Exactly once: a retried close neither re-runs a callback nor raises again.
    calls.clear()
    ledger.close()
    assert calls == []


def test_close_prefers_a_late_cancellation_over_an_earlier_ordinary_failure(caplog):
    ordinary = RuntimeError("staging close failed")
    shutdown = SystemExit(1)
    calls: list[str] = []

    def failing_close() -> None:
        calls.append("staging")
        raise ordinary

    def shutdown_close() -> None:
        calls.append("host")
        raise shutdown

    ledger = ResourceLedger()
    ledger.register(object(), ResourceKind.STAGING, failing_close)
    ledger.register(object(), ResourceKind.HOST_STORAGE, shutdown_close)

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
        ledger.close()

    assert calls == ["staging", "host"]
    assert exc_info.value is shutdown
    assert any("staging close failed" in record.getMessage() for record in caplog.records)


def test_close_reports_a_lone_cancellation_without_a_lifecycle_error():
    shutdown = SystemExit("worker shutdown")
    calls: list[str] = []

    def shutdown_close() -> None:
        calls.append("device")
        raise shutdown

    ledger = ResourceLedger()
    ledger.register(object(), ResourceKind.DEVICE, shutdown_close)
    ledger.register(object(), ResourceKind.HOST_STORAGE, lambda: calls.append("host"))

    with pytest.raises(SystemExit) as exc_info:
        ledger.close()

    assert calls == ["device", "host"]
    assert exc_info.value is shutdown
    assert not isinstance(exc_info.value, LifecycleError)


def test_ledger_never_calls_torch_distributed(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("the resource ledger must not call torch.distributed")

    for name in ("barrier", "all_reduce", "all_gather_object", "destroy_process_group"):
        monkeypatch.setattr(torch.distributed, name, _forbidden)

    ledger = ResourceLedger()
    _register(ledger, "device", ResourceKind.DEVICE)
    ledger.close()
    assert _CLOSE_LOG == ["device"]


def test_close_keeps_torch_distributed_out_of_the_cancellation_path(monkeypatch):
    """An interrupted cleanup must not try to release the failure through a collective."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("the resource ledger must not call torch.distributed")

    for name in ("barrier", "all_reduce", "all_gather_object", "destroy_process_group"):
        monkeypatch.setattr(torch.distributed, name, _forbidden)

    ledger = ResourceLedger()
    ledger.register(object(), ResourceKind.EXECUTION, lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    _register(ledger, "host", ResourceKind.HOST_STORAGE)

    with pytest.raises(KeyboardInterrupt):
        ledger.close()
    assert _CLOSE_LOG == ["host"]
