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
from functools import wraps

from vllm_omni.diffusion.models.schedulers import FlowMatchEulerDiscreteScheduler

MARKER_ENV_VAR = "VLLM_OMNI_CUSTOM_SCHEDULER_MARKER"


def mark_scheduler_event(event: str) -> None:
    """Append ``event`` to the marker file named by ``MARKER_ENV_VAR``.

    No-op when the env var is unset, so instrumented schedulers can be reused
    by tests that do not care about the marker file.
    """
    path = os.environ.get(MARKER_ENV_VAR)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{event}\n")


class FlowMatchEulerDiscreteSchedulerForTest(FlowMatchEulerDiscreteScheduler):
    """Stock flow-match Euler scheduler that proves it was used via a marker file."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        mark_scheduler_event("constructed")
        return super().from_pretrained(*args, **kwargs)

    # Keep the parent parameter names. Qwen-Image / Flux / SD3 call
    # inspect.signature(scheduler.set_timesteps) and reject wrappers whose
    # signature is only (*args, **kwargs) — they look for a named ``sigmas``.
    @wraps(FlowMatchEulerDiscreteScheduler.set_timesteps)
    def set_timesteps(self, *args, **kwargs):
        mark_scheduler_event("set_timesteps")
        return super().set_timesteps(*args, **kwargs)

    @wraps(FlowMatchEulerDiscreteScheduler.step)
    def step(self, *args, **kwargs):
        mark_scheduler_event("stepped")
        return super().step(*args, **kwargs)
