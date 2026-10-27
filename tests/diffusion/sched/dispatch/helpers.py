# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Controller lifecycle and event-loop draining shared by scheduling tests."""

import asyncio
from contextlib import asynccontextmanager

from vllm_omni.diffusion.sched.dispatch.config import CentralSchedulingConfig
from vllm_omni.diffusion.sched.dispatch.controller import CentralScheduler, Decision


@asynccontextmanager
async def controller_context(replica_ids, **options):
    """Own a controller and the pending tasks started by one test."""
    policy = CentralScheduler(list(replica_ids), CentralSchedulingConfig(enabled=True, **options))
    pending: dict[str, asyncio.Task[Decision]] = {}
    try:
        yield policy, pending
    finally:
        policy.close()
        await asyncio.gather(*pending.values(), return_exceptions=True)


async def drain():
    # Register admissions, drain the queue, then resume admitted coroutines.
    for _ in range(3):
        await asyncio.sleep(0)
