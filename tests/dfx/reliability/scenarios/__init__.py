# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fault scenarios. Each class owns one fault's steps; model tests call ``run``."""

from tests.dfx.reliability.scenarios.oom import (
    GpuOomConcurrent,
    GpuOomErrorContract,
    GpuOomLarge,
    GpuOomRecovery,
)
from tests.dfx.reliability.scenarios.process_kill import (
    PartialStageIsolation,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerNoLoad,
    ProcessKillWorkerWithLoad,
)

__all__ = [
    "GpuOomConcurrent",
    "GpuOomErrorContract",
    "GpuOomLarge",
    "GpuOomRecovery",
    "PartialStageIsolation",
    "ProcessKillServeRootNoLoad",
    "ProcessKillServeRootWithLoad",
    "ProcessKillTreeNoLoad",
    "ProcessKillTreeWithLoad",
    "ProcessKillWorkerNoLoad",
    "ProcessKillWorkerWithLoad",
]
