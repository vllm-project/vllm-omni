# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test double for the scheduler-injection E2E tests.

``FlowMatchEulerDiscreteSchedulerForTest`` subclasses the vendored
``FlowMatchEulerDiscreteScheduler`` and records construction/timestepping/
stepping by appending marker lines to the file named by the
``VLLM_OMNI_CUSTOM_SCHEDULER_MARKER`` environment variable. Diffusion workers
run in subprocesses, so in-process counters are not observable from the test
process; the env var is inherited by child processes, making the marker file
a cross-process observable.
"""

from __future__ import annotations

import os

from vllm_omni.diffusion.models.schedulers import FlowMatchEulerDiscreteScheduler

MARKER_ENV_VAR = "VLLM_OMNI_CUSTOM_SCHEDULER_MARKER"


class FlowMatchEulerDiscreteSchedulerForTest(FlowMatchEulerDiscreteScheduler):
    """Stock flow-match Euler scheduler that proves it was used via a marker file."""

    @staticmethod
    def _mark(event: str) -> None:
        path = os.environ.get(MARKER_ENV_VAR)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{event}\n")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls._mark("constructed")
        return super().from_pretrained(*args, **kwargs)

    def set_timesteps(self, *args, **kwargs):
        type(self)._mark("set_timesteps")
        return super().set_timesteps(*args, **kwargs)

    def step(self, *args, **kwargs):
        type(self)._mark("stepped")
        return super().step(*args, **kwargs)
