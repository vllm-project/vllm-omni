# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resource accounting for the unified weight stack startup transaction.

This module starts as the ``ResourceLedger`` delivered by sub-RFC 01-1.  The
weight-stack switch (01-2), ``execution_offload_facts`` (01-3) and the startup
driver with ``FaultPlan`` (01-5) extend the same module later; the ledger is the
only part that exists today.

The ledger is *not* a global resource manager: it is created by the startup
transaction (and, after commit, by the ``OffloadRuntime``), lives only as long
as its owner, and never calls ``torch.distributed``.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from dataclasses import dataclass

from vllm_omni.diffusion.weight_management.errors import LifecycleError

logger = logging.getLogger(__name__)


class ResourceKind(enum.IntEnum):
    """Resource categories, closed in ascending order (RFC 01 section 5.2)."""

    EXECUTION = 1  # stop accepting new executions
    INFLIGHT = 2  # wait for device copies and collectives to finish
    HOOKS = 3  # remove hooks and model references
    DEVICE = 4  # device slots, events, streams, collective workspaces
    REGISTRATION = 5  # undo memory registration
    STAGING = 6  # release the staging area
    PLACEHOLDER = 7  # repoint parameters at placeholders
    HOST_STORAGE = 8  # close this process's host storage mapping


@dataclass
class _LedgerEntry:
    handle: object
    kind: ResourceKind
    close: Callable[[], None]


class ResourceLedger:
    """Resources owned by one startup transaction or one ``OffloadRuntime``.

    Every resource handle is registered exactly once (by object identity).  The
    ledger owns the handle until it is handed over to an owner outside the
    ledger (:meth:`hand_over`) or moved to another ledger (:meth:`transfer`).

    :meth:`close` releases the remaining entries in ``ResourceKind`` order and,
    within one kind, in reverse registration order.  Closing is idempotent: the
    second call does nothing, and every callback runs at most once even when one
    of them fails.  A failing close callback does not stop the other callbacks;
    after all of them have run, one :class:`LifecycleError` reports the ordinary
    failures with the first one as ``__cause__``.  A cancellation
    (``KeyboardInterrupt``, ``SystemExit`` or another ``BaseException`` that is
    not an ``Exception``) is re-raised in its own type once the remaining
    callbacks have run, and takes precedence over ordinary failures.
    """

    def __init__(self) -> None:
        self._entries: list[_LedgerEntry] = []
        self._closed = False
        self._transferred = False

    def register(self, handle: object, kind: ResourceKind, close: Callable[[], None]) -> None:
        """Register one resource handle; registering the same handle twice is an error."""
        self._check_mutable("register")
        if any(entry.handle is handle for entry in self._entries):
            raise LifecycleError(
                "resource handle is already registered in this ledger",
                where="ResourceLedger.register",
            )
        self._entries.append(_LedgerEntry(handle=handle, kind=kind, close=close))

    def hand_over(self, handle: object) -> None:
        """Remove one entry and hand the handle over to an owner outside the ledger."""
        self._check_mutable("hand_over")
        for index, entry in enumerate(self._entries):
            if entry.handle is handle:
                del self._entries[index]
                return
        raise LifecycleError(
            "resource handle is not registered in this ledger",
            where="ResourceLedger.hand_over",
        )

    def transfer(self) -> ResourceLedger:
        """Move every entry into a new ledger and leave this ledger empty."""
        if self._closed:
            raise LifecycleError(
                "cannot transfer resources after the ledger is closed",
                where="ResourceLedger.transfer",
            )
        if self._transferred:
            raise LifecycleError(
                "resource ledger was already transferred",
                where="ResourceLedger.transfer",
            )
        target = ResourceLedger()
        target._entries = self._entries
        self._entries = []
        self._transferred = True
        return target

    def close(self) -> None:
        """Close every remaining resource; safe to call more than once.

        The ledger is marked closed and emptied before the first callback runs, so
        every registered callback runs exactly once even when one of them raises or
        is interrupted, and a retried ``close()`` cannot run a callback twice.
        """
        if self._closed:
            return
        self._closed = True
        entries = self._entries
        self._entries = []

        # ResourceKind ascending, then reverse registration order inside one kind.
        ordered = sorted(enumerate(entries), key=lambda item: (int(item[1].kind), -item[0]))
        failures: list[tuple[_LedgerEntry, BaseException]] = []
        for _, entry in ordered:
            try:
                entry.close()
            except BaseException as exc:  # noqa: BLE001 - every callback must run
                failures.append((entry, exc))

        if not failures:
            return

        # A cancellation keeps its own type and wins over ordinary cleanup failures:
        # RFC 01 section 6.2 requires ``KeyboardInterrupt``/``SystemExit`` to bypass
        # confirmation and stop the worker immediately, which a ``LifecycleError``
        # wrapper would prevent.  Ordinary failures are logged instead of raised.
        cancellation = next(((entry, exc) for entry, exc in failures if not isinstance(exc, Exception)), None)
        first_entry, first_error = failures[0] if cancellation is None else cancellation
        for entry, exc in failures:
            if exc is first_error:
                continue
            logger.error(
                "resource close callback failed (kind=%s): %s",
                entry.kind.name,
                exc,
                exc_info=exc,
            )
        if cancellation is not None:
            logger.error(
                "resource close was interrupted by %s while closing %s; "
                "the remaining callbacks were attempted and the interruption is re-raised",
                type(first_error).__name__,
                first_entry.kind.name,
                exc_info=first_error,
            )
            raise first_error
        raise LifecycleError(
            f"{len(failures)} resource close callback(s) failed; "
            f"first failure was {first_entry.kind.name}: {type(first_error).__name__}: {first_error}",
            where="ResourceLedger.close",
        ) from first_error

    def _check_mutable(self, action: str) -> None:
        if self._closed:
            raise LifecycleError(
                f"cannot {action} resources after the ledger is closed",
                where=f"ResourceLedger.{action}",
            )
        if self._transferred:
            raise LifecycleError(
                f"cannot {action} resources after the ledger was transferred",
                where=f"ResourceLedger.{action}",
            )
