# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep failed LingBot operations' device buffers alive until worker teardown.

A submitted operation may still reference its buffers when it fails. Callers
retain those owners here and propagate the original exception. Recovery requires
fresh workers; this module does not retry work, roll back caches, or supervise
processes.
"""

from collections.abc import Iterable
from threading import Lock

_QUARANTINE: list[tuple[object, ...]] = []
_LOCK = Lock()


def retain_failure_owners(owners: Iterable[object]) -> None:
    """Retain in-flight owners for the lifetime of the failed worker."""
    owner_tuple = tuple(owners)
    with _LOCK:
        _QUARANTINE.append(owner_tuple)
