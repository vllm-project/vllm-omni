# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Error types for the unified weight stack.

The hierarchy follows RFC #9536, ``rfcs/01-startup-transaction.md`` section 3.
Every error carries a ``phase`` (the startup step that raised it) and, once the
startup driver has filled them in, the stage, rank and world size.  The driver
wraps exceptions that are not :class:`WeightStackError` into the type that
belongs to the failing step and keeps the original exception as ``__cause__``.

This module only depends on the standard library.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_ERROR_MESSAGE_CHARS = 4096
"""Longest error message carried in an :class:`ErrorSummary`."""

_ERROR_HEADER = "[weight-stack]"


@dataclass(frozen=True)
class ErrorSummary:
    """Picklable description of one rank's error, exchanged at confirmation points."""

    error_type: str
    phase: str
    rank: int
    message: str  # truncated to MAX_ERROR_MESSAGE_CHARS characters


class WeightStackError(RuntimeError):
    """Base class of every error raised by the unified weight stack.

    ``phase`` is the startup step that raised the error.  Subclasses that can be
    raised at more than one step (confirmation points, ``allocate`` /
    ``connect_writer``, lifecycle calls) accept a ``phase`` keyword to override
    the class default.  ``stage_id``, ``rank`` and ``world_size`` are filled in
    by the startup driver before an error crosses a confirmation point.
    """

    phase: str = "unknown"

    def __init__(
        self,
        what: str,
        *,
        where: str | None = None,
        fix: str | None = None,
        component: str | None = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(what)
        self.what = what
        self.where = where
        self.fix = fix
        self.component = component
        if phase is not None:
            self.phase = phase
        self.stage_id: int | None = None
        self.rank: int | None = None
        self.world_size: int | None = None

    def __str__(self) -> str:
        parts = [_ERROR_HEADER]
        if self.stage_id is not None:
            parts.append(f"[stage {self.stage_id}]")
        if self.rank is not None:
            if self.world_size is not None:
                parts.append(f"[rank {self.rank}/{self.world_size}]")
            else:
                parts.append(f"[rank {self.rank}]")
        parts.append(f"[{self.phase}]")
        message = f"{''.join(parts)} {self.what}"
        if self.where is not None:
            message += f"; at {self.where}"
        if self.fix is not None:
            message += f"; fix: {self.fix}"
        return message

    def summary(self) -> ErrorSummary:
        """Return the picklable summary exchanged at confirmation points."""
        message = str(self)
        if len(message) > MAX_ERROR_MESSAGE_CHARS:
            message = message[:MAX_ERROR_MESSAGE_CHARS]
        return ErrorSummary(
            error_type=type(self).__name__,
            phase=self.phase,
            rank=self.rank if self.rank is not None else -1,
            message=message,
        )


class ConfigError(WeightStackError, ValueError):
    """Argument parsing: invalid switch value, conflicting process state or unsupported combination."""

    phase = "config"


class ConstructionError(WeightStackError):
    """Construction step: read trap fired, meta tensors left over, or pipeline not migrated to structure-only build."""

    phase = "build_structure"


class PlanningError(WeightStackError):
    """Offloader planning: incomplete component, invalid selection, missing repeated blocks or capacity."""

    phase = "plan"


class PreflightError(WeightStackError):
    """Loader preflight: unreadable source, adapter key mismatch, cold-start path mismatch or host request mismatch."""

    phase = "preflight"


class GroupConfirmationError(WeightStackError):
    """Replica-wide confirmation: inconsistent collective parts, single-rank AllGather group, deployment mismatch."""

    phase = "group_confirmation"


class HostCapacityError(GroupConfirmationError):
    """Host capacity admission failed on this node (raised by ``HostWeightManager.admit``)."""

    phase = "admit"


class PrepareError(WeightStackError):
    """``prepare_materialization``: device slots, streams, events or staging allocation failed."""

    phase = "prepare"


class DeviceCapacityError(PrepareError):
    """Device capacity admission failed while preparing materialization."""


class StackUnavailableError(WeightStackError):
    """A module or factory function of the new stack does not exist."""

    phase = "construct_modules"


class AllocationError(WeightStackError):
    """Host storage allocation, or a waiter connecting to the write process, failed."""

    phase = "allocate"


class WriteError(WeightStackError):
    """Write step: a write failed, or a tensor name is missing from the host storage layout."""

    phase = "write"


class FinalizeError(WeightStackError):
    """Finalize step: write verification, sealing, handover or read-only binding failed."""

    phase = "finalize"


class CommitError(WeightStackError):
    """Commit step: final manifest validation, memory registration, hooks or handover failed."""

    phase = "commit"


class PeerFailureError(WeightStackError):
    """Another rank reported an error at this confirmation point."""

    phase = "group_confirmation"


class ConfirmationTimeoutError(WeightStackError):
    """A rank did not reach this confirmation point within the timeout."""

    phase = "group_confirmation"


class LifecycleError(WeightStackError):
    """Illegal state transition, duplicate resource registration or a call after close."""

    phase = "lifecycle"


class RuntimeFault(WeightStackError):  # noqa: N818 - the RFC fixes this public name
    """Service-time fault after ``IN_SERVICE``: the whole replica must terminate."""

    phase = "service"


STEP_ERROR_TYPES: Mapping[str, type[WeightStackError]] = {
    "construct_modules": StackUnavailableError,
    "build_structure": ConstructionError,
    "plan": PlanningError,
    "preflight": PreflightError,
    "point0_topology": GroupConfirmationError,
    "point1_fingerprints": GroupConfirmationError,
    "confirm": GroupConfirmationError,
    "point1_elections": GroupConfirmationError,
    "admit": GroupConfirmationError,
    "point2_ready": GroupConfirmationError,
    "point3_commit": GroupConfirmationError,
    "allocate": AllocationError,
    "connect_writer": AllocationError,
    "prepare": PrepareError,
    "write": WriteError,
    "finalize": FinalizeError,
    "commit": CommitError,
}
"""Startup step -> error type used to wrap a non-``WeightStackError`` exception (RFC 01 section 3)."""


def wrap_step_error(step: str, exc: BaseException) -> WeightStackError:
    """Wrap ``exc`` into the error type of ``step``, preserving ``exc`` as ``__cause__``.

    A :class:`WeightStackError` is returned unchanged.  The startup driver then
    fills ``stage_id``/``rank``/``world_size`` before raising the error.
    """
    if isinstance(exc, WeightStackError):
        return exc
    error_type = STEP_ERROR_TYPES.get(step)
    if error_type is None:
        raise ValueError(f"unknown weight-stack startup step: {step!r}") from exc
    error = error_type(f"{type(exc).__name__}: {exc}", phase=step)
    error.__cause__ = exc
    return error
