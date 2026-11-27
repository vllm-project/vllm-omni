# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fault scenarios. Each class owns one fault's steps; model tests call ``run``."""

from tests.dfx.reliability.scenarios.launch import (
    FullCrashAtLaunch,
    GpuOomAtLaunch,
    PartialCrashAtLaunch,
    ServeRootCrashAtLaunch,
)
from tests.dfx.reliability.scenarios.oom import (
    GpuOomUnderLoad,
)
from tests.dfx.reliability.scenarios.process_kill import (
    GracefulShutdown,
    PartialCrashAtExit,
    PartialStageIsolation,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerNoLoad,
    ProcessKillWorkerWithLoad,
    ReplicaGracefulShutdown,
)
from tests.dfx.reliability.scenarios.replica import (
    ReplicaCrashAtExit,
    ReplicaCrashAtLaunch,
    ReplicaCrashIdle,
    ReplicaCrashWithLoad,
)

__all__ = [
    "FullCrashAtLaunch",
    "ServeRootCrashAtLaunch",
    "GracefulShutdown",
    "ReplicaGracefulShutdown",
    "GpuOomAtLaunch",
    "GpuOomUnderLoad",
    "PartialCrashAtExit",
    "PartialCrashAtLaunch",
    "PartialStageIsolation",
    "ProcessKillServeRootNoLoad",
    "ProcessKillServeRootWithLoad",
    "ProcessKillTreeNoLoad",
    "ProcessKillTreeWithLoad",
    "ProcessKillWorkerNoLoad",
    "ProcessKillWorkerWithLoad",
    "ReplicaCrashAtExit",
    "ReplicaCrashAtLaunch",
    "ReplicaCrashIdle",
    "ReplicaCrashWithLoad",
]
