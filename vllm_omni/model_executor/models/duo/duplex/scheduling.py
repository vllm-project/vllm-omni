# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-session Realization, Intent and codec admission on shared GPUs."""

import asyncio
import time

from .trace import trace


class DuoPhaseSchedule:
    def __init__(self, state, mode, *, defer_clean=False):
        if mode not in ("free", "alternating", "serial"):
            raise ValueError("Realization scheduling must be free, alternating or serial")
        if defer_clean and mode not in ("alternating", "serial"):
            raise ValueError("Deferred clean requires alternating or serial scheduling")
        self.state, self.mode, self.defer_clean = state, mode, defer_clean
        self.latent = self.published = -1
        self.intent_active = False
        self.intent = self.clean = self.codec = self.realization_sent = -1
        self.realization_active = False
        self.asr_active = set()
        self.closed = False
        self.changed = asyncio.Event()

    async def admit(self, stage, block):
        started = time.perf_counter_ns()
        trace("schedule_wait", self.state, block=block, stage=stage, mode=self.mode)
        while not self.closed:
            ready = {
                "asr_overlap": True,
                "asr_phase": not self.realization_active
                and (self.mode != "serial" or (not self.intent_active and self.clean >= max(block - 1, self.intent))),
                "cpu": self.realization_sent >= block - 1,
                "intent": (self.latent if self.defer_clean and self.mode != "serial" else self.clean) >= block - 1
                and (not self.defer_clean or not self.realization_active)
                and (self.mode != "serial" or self.codec >= block - 1),
                "realization": min(self.intent, self.clean, self.codec) >= block - 1
                and self.intent >= block
                and not self.asr_active,
                "codec": (self.latent if self.defer_clean else self.clean) >= block,
                "clean": self.codec >= block and not self.intent_active and not self.asr_active,
            }[stage]
            if self.mode == "free" or ready:
                if stage == "asr_phase":
                    self.asr_active.add(block)
                elif stage in ("realization", "clean"):
                    self.realization_active = True
                elif stage == "intent":
                    self.intent_active = True
                trace(
                    "schedule_admitted",
                    self.state,
                    block=block,
                    stage=stage,
                    mode=self.mode,
                    wait_ms=(time.perf_counter_ns() - started) / 1e6,
                )
                return
            changed = self.changed
            await changed.wait()
        raise asyncio.CancelledError

    def complete(self, stage, block):
        if stage == "asr":
            self.asr_active.discard(block)
        else:
            setattr(self, stage, block)
        if stage == "clean" or (self.defer_clean and stage == "latent"):
            self.realization_active = False
        if stage == "intent":
            self.intent_active = False
        trace("schedule_complete", self.state, block=block, stage=stage, mode=self.mode)
        self.changed.set()
        self.changed = asyncio.Event()

    def close(self):
        self.closed = True
        self.changed.set()
