# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Storage-independent completion and history-release boundary for AR KV."""

from collections.abc import Sequence
from typing import Protocol, TypeVar

OwnerT = TypeVar("OwnerT", contravariant=True)
TaskT = TypeVar("TaskT", contravariant=True)
ViewT = TypeVar("ViewT", covariant=True)


class KVBackend(Protocol[TaskT, ViewT]):
    """Execution boundary shared by chunk-window and Cell backends.

    Tasks and views remain backend-specific: no implicit layout conversion.
    Prepare returns borrowed read/write views, valid through that forward;
    callers complete model work before publish, transfers and last-use eviction.
    Publish confirms successful writes; it does not synchronize GPU computation.
    Retention policy and finish/release owners are described by KVCacheLifecycle.
    A failed forward must be retired by backend cleanup, never published.
    """

    def prepare(self, tasks: Sequence[TaskT]) -> Sequence[ViewT]:
        """Prepare attention/read-write views corresponding one-to-one to tasks."""
        ...

    def publish(self, tasks: Sequence[TaskT]) -> None:
        """Publish successful writes for the tasks, after model computation."""
        ...


class KVCacheLifecycle(Protocol[OwnerT]):
    """Separate completion of a compute request from release of its history.

    Owners are backend-local identities, not necessarily service request IDs.
    Legacy uses a session/branch adapter; Noisy KV uses a version namespace.
    Layout, history selection and logical allocation granularity stay with the
    backend. Callers must stop model computation before either operation;
    backends additionally complete their outstanding storage communication
    before recycling pages or slots. Neither operation frees pool tensors.
    """

    def finish_request(self, owner: OwnerT) -> None:
        """Retire request-local bookkeeping while preserving selected history."""
        ...

    def release_history(self, owner: OwnerT) -> None:
        """Return the owner's retained storage after its compute request ends."""
        ...
