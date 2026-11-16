# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded asynchronous helpers shared by duplex test layers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.events import AudioDelta


async def wait_until(predicate: Callable[[], object], *, timeout: float = 2) -> None:
    """Yield until a condition holds, without an unbounded polling loop."""

    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout)


async def next_audio(buffer: DuplexOutputBuffer, *, timeout: float = 3) -> AudioDelta:
    """Read past non-audio events, bounded across the complete read loop."""

    async def read() -> AudioDelta:
        while True:
            event = await buffer.get()
            if event is None:
                raise AssertionError("duplex output closed before an audio event")
            if isinstance(event, AudioDelta):
                return event

    return await asyncio.wait_for(read(), timeout)
