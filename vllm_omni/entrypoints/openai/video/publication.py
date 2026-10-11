# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drain publication workers before their caller rolls back owned resources."""

from __future__ import annotations

import asyncio
from typing import TypeVar

_T = TypeVar("_T")


async def drain_publication(task: asyncio.Task[_T]) -> _T:
    """Wait despite repeated caller cancellation, preserving the writer's result.

    Called after the first cancellation. The caller must register any returned
    resource for rollback and re-raise its original CancelledError.
    """
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()
