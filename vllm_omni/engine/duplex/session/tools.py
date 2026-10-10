# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session-owned function-call ledger.

The duplex session records which calls are open, finished, cancelled, or stale.
A model plugin only recognizes a call in model output and, later, how to feed
an accepted result back into that model. This ledger does not execute tools
and does not gate audio append: input stays available while a result is
outstanding. Cancelling with ``reason="timeout"`` is how a caller records a
timeout; there is no background timer here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DuplexToolLedgerError(ValueError):
    """A call or result the session will not carry."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ToolCallStatus(str, Enum):
    OPEN = "open"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    STALE = "stale"


@dataclass
class ToolCallRecord:
    call_id: str
    name: str
    arguments: str
    epoch: int
    status: ToolCallStatus
    cancel_reason: str | None = None


class DuplexToolLedger:
    """Outstanding and finished function calls for one duplex session."""

    def __init__(self) -> None:
        self._calls: dict[str, ToolCallRecord] = {}

    def get(self, call_id: str) -> ToolCallRecord | None:
        return self._calls.get(call_id)

    def open_call(self, *, call_id: str, name: str, arguments: str, epoch: int) -> ToolCallRecord:
        if not call_id or not name:
            raise DuplexToolLedgerError(
                "A function call requires call_id and name",
                code="invalid_function_call",
            )
        if call_id in self._calls:
            raise DuplexToolLedgerError(
                f"function call {call_id} is already registered",
                code="duplicate_function_call",
            )
        record = ToolCallRecord(
            call_id=call_id,
            name=name,
            arguments=arguments,
            epoch=epoch,
            status=ToolCallStatus.OPEN,
        )
        self._calls[call_id] = record
        return record

    def retire_before(self, epoch: int) -> None:
        """Mark open calls from an older epoch stale. Does not touch newer calls."""
        for record in self._calls.values():
            if record.status is ToolCallStatus.OPEN and record.epoch < epoch:
                record.status = ToolCallStatus.STALE

    def cancel(self, call_id: str, *, reason: str = "cancelled") -> ToolCallRecord:
        record = self._calls.get(call_id)
        if record is None or record.status is ToolCallStatus.STALE:
            raise DuplexToolLedgerError(
                f"No open function call {call_id}",
                code="unknown_function_call",
            )
        if record.status is ToolCallStatus.COMPLETED:
            raise DuplexToolLedgerError(
                f"function call {call_id} already completed",
                code="duplicate_function_call_output",
            )
        if record.status is ToolCallStatus.OPEN:
            record.status = ToolCallStatus.CANCELLED
            record.cancel_reason = reason
        return record

    def accept_result(self, call_id: str, *, epoch: int) -> ToolCallRecord:
        record = self._calls.get(call_id)
        if record is None:
            raise DuplexToolLedgerError(
                f"function_call_output requires the call_id of an open function call, got {call_id!r}",
                code="unknown_function_call",
            )
        if record.status is ToolCallStatus.COMPLETED:
            raise DuplexToolLedgerError(
                f"function_call_output already exists for call_id {call_id}",
                code="duplicate_function_call_output",
            )
        if record.status is ToolCallStatus.CANCELLED:
            raise DuplexToolLedgerError(
                f"function_call_output for {call_id} arrived after cancel ({record.cancel_reason})",
                code="late_function_call_output",
            )
        if record.status is ToolCallStatus.STALE or record.epoch != epoch:
            record.status = ToolCallStatus.STALE
            raise DuplexToolLedgerError(
                f"function_call_output for {call_id} belongs to epoch {record.epoch}, session is {epoch}",
                code="stale_function_call_output",
            )
        record.status = ToolCallStatus.COMPLETED
        return record
