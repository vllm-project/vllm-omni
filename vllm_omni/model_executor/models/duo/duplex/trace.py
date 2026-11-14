# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional host phase trace for bounded integration runs."""

import asyncio
import json
import os
import threading
import time


def trace(phase, state, **values):
    path = os.environ.get("DUO_TRACE_PATH")
    if path:
        event = dict(
            phase=phase,
            monotonic_ns=time.perf_counter_ns(),
            session_id=state.session_id,
            fence=state.fence,
            block=values.pop("block", state.next_block),
            **values,
        )
        with open(path, "a") as stream:
            stream.write(json.dumps(event) + "\n")


async def trace_loop_delays(path: str) -> None:
    """Record host-loop stalls only when the operator enabled Realization tracing."""
    period_ns = 5_000_000
    while True:
        target = time.perf_counter_ns() + period_ns
        await asyncio.sleep(period_ns / 1e9)
        now = time.perf_counter_ns()
        if now - target > 10_000_000:
            event = dict(
                phase="event_loop_delay",
                monotonic_ns=now,
                target_ns=target,
                delay_ms=(now - target) / 1e6,
                pid=os.getpid(),
                tid=threading.get_native_id(),
            )
            with open(path, "a") as stream:
                stream.write(json.dumps(event) + "\n")
