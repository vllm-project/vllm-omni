# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OmniCoordinator process entry point.

Runs the OmniCoordinator in a dedicated subprocess, matching vLLM's
DPCoordinator pattern. Physical isolation prevents GIL contention with
the Orchestrator and makes coupling through direct object references
impossible — all communication goes through ZMQ.
"""

from __future__ import annotations

import signal
from typing import Any

from vllm_omni.distributed.omni_coordinator.omni_coordinator import OmniCoordinator


def run_omni_coordinator_proc(
    router_zmq_addr: str,
    pub_zmq_addr: str,
    heartbeat_timeout: float,
    ready_pipe: Any,
) -> None:
    """Main loop running inside the coordinator child process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    coordinator = OmniCoordinator(
        router_zmq_addr=router_zmq_addr,
        pub_zmq_addr=pub_zmq_addr,
        heartbeat_timeout=heartbeat_timeout,
    )

    ready_pipe.send("ready")
    ready_pipe.close()

    coordinator.wait_for_shutdown()
