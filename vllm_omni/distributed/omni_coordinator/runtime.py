# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lifecycle wrapper around :class:`OmniCoordinator`.

``OmniCoordinatorRuntime`` spawns the coordinator as an independent process
(matching vLLM's DPCoordinator pattern). Physical isolation prevents GIL
contention and makes direct-object coupling impossible.

The ROUTER address is later handed to :class:`OmniMasterServer` so it can be
published to registering replicas; the PUB address is handed to the
``MembershipController``, which constructs its :class:`OmniCoordClientForHub`
against it.
"""

from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.connection
import os
import signal
import weakref
from typing import Any

from vllm.utils.network_utils import get_open_ports_list

from vllm_omni.distributed.omni_coordinator.omni_coordinator import OmniCoordinator

logger = logging.getLogger(__name__)


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


def _get_coordinator_mp_context() -> multiprocessing.context.BaseContext:
    """Return the multiprocessing context used for OmniCoordinator.

    For the current vllm-omni startup path, ``spawn`` is too expensive: the
    child re-imports a heavy CLI / model stack before it can acknowledge the
    ready pipe, which can exceed the coordinator startup timeout. Prefer
    ``fork`` on platforms that support it.

    ``fork`` must happen before the parent initializes CUDA or owns long-lived
    ZMQ sockets; otherwise the child inherits unsafe state. If coordinator
    startup moves later, switch this to ``spawn``/``forkserver``.

    TODO: make the coordinator child entry cheap and spawn-safe, then revisit
    whether ``spawn`` is still needed here.
    """
    if os.name != "nt" and "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _shutdown_proc(proc: multiprocessing.Process) -> None:
    """Best-effort process termination for weakref finalizer."""
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)


class OmniCoordinatorRuntime:
    """Own one :class:`OmniCoordinator` running in a child process.

    Constructor spawns the process; :meth:`close` tears it down.
    The class deliberately does not expose the coordinator instance —
    callers consume it only via ZMQ through :class:`OmniCoordClientForStage`
    and :class:`OmniCoordClientForHub`.
    """

    def __init__(
        self,
        *,
        host: str,
        heartbeat_timeout: float,
    ) -> None:
        if not host:
            raise ValueError("host must be a non-empty string")
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be positive")

        router_port, pub_port = get_open_ports_list(count=2)
        self.router_address: str = f"tcp://{host}:{router_port}"
        self.pub_address: str = f"tcp://{host}:{pub_port}"

        self._closed = False

        ctx = _get_coordinator_mp_context()
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        self._proc: multiprocessing.Process = ctx.Process(
            target=run_omni_coordinator_proc,
            kwargs={
                "router_zmq_addr": self.router_address,
                "pub_zmq_addr": self.pub_address,
                "heartbeat_timeout": heartbeat_timeout,
                "ready_pipe": child_conn,
            },
            daemon=True,
            name="OmniCoordinator",
        )
        self._proc.start()
        child_conn.close()

        ready = multiprocessing.connection.wait([parent_conn, self._proc.sentinel], timeout=30)
        if not ready:
            self._proc.terminate()
            self._proc.join(timeout=5)
            raise RuntimeError("OmniCoordinator process failed to start within 30s")

        try:
            status = parent_conn.recv()
        except EOFError:
            raise RuntimeError("OmniCoordinator process died during startup") from None
        finally:
            parent_conn.close()

        if status != "ready":
            raise RuntimeError(f"OmniCoordinator unexpected status: {status}")

        self._finalizer = weakref.finalize(self, _shutdown_proc, self._proc)

        logger.info(
            "[OmniCoordinatorRuntime] Started (pid=%d router=%s pub=%s heartbeat_timeout=%.1fs)",
            self._proc.pid,
            self.router_address,
            self.pub_address,
            heartbeat_timeout,
        )

    def close(self) -> None:
        """Tear down the coordinator process. Idempotent."""
        if self._closed:
            return
        self._closed = True
        _shutdown_proc(self._proc)
        self._finalizer.detach()
