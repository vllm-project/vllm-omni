# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the startup resource ledger (RFC 9536 01-1)."""

from __future__ import annotations

import logging

import pytest
import torch

from vllm_omni.diffusion.weight_stack.errors import LifecycleError
from vllm_omni.diffusion.weight_stack.startup import ResourceKind, ResourceLedger

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


def test_ledger_never_calls_torch_distributed(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("the resource ledger must not call torch.distributed")

    for name in ("barrier", "all_reduce", "all_gather_object", "destroy_process_group"):
        monkeypatch.setattr(torch.distributed, name, _forbidden)

    ledger = ResourceLedger()
    _register(ledger, "device", ResourceKind.DEVICE)
    ledger.close()
    assert _CLOSE_LOG == ["device"]
