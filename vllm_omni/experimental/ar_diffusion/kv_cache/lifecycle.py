# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Storage-independent completion and history-release boundary for AR KV."""

from typing import Protocol, TypeVar

OwnerT = TypeVar("OwnerT", contravariant=True)


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
