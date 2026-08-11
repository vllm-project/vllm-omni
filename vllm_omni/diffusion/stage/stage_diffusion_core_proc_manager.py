# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Process manager for the omni diffusion stage subprocess.

Owns a single :class:`StageDiffusionCoreProc` child process and exposes the small
process-lifecycle surface used by the orchestrator / stage pool (mirroring the
subset of vLLM's ``CoreEngineProcManager`` that applies) while keeping
diffusion's custom ZMQ wire protocol.

Split out of ``stage_diffusion_core_proc.py`` so the subprocess entry point and
its head-side lifecycle owner live in separate modules (mirroring the LLM split
between ``stage_llm_core_proc.py`` and ``stage_llm_core_proc_manager.py``).
"""

from __future__ import annotations

import multiprocessing.connection
from types import SimpleNamespace
from typing import TYPE_CHECKING

import zmq
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.utils import CoreEngine, EngineZmqAddresses, wait_for_engine_startup
from vllm.v1.utils import shutdown

from vllm_omni.diffusion.stage.stage_diffusion_core_proc import StageDiffusionCoreProc

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class StageDiffusionCoreProcManager:
    """Owns a :class:`StageDiffusionCoreProc` subprocess.

    Mirrors the small process-lifecycle surface used by vLLM's
    ``CoreEngineProcManager`` while keeping diffusion's custom wire protocol.
    """

    def __init__(
        self,
        *,
        model: str,
        od_config: OmniDiffusionConfig,
        stage_init_timeout: int,
        handshake_address: str | None = None,
        addresses: EngineZmqAddresses | None = None,
        omni_coord_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
    ) -> None:
        handshake_address = handshake_address or get_open_zmq_ipc_path()
        addresses = addresses or EngineZmqAddresses(
            inputs=[get_open_zmq_ipc_path()],
            outputs=[get_open_zmq_ipc_path()],
        )

        ctx = get_mp_context()
        proc = ctx.Process(
            target=StageDiffusionCoreProc.run_diffusion_proc,
            name="StageDiffusionCoreProc",
            kwargs={
                "model": model,
                "od_config": od_config,
                "handshake_address": handshake_address,
                "local_client": True,
                "headless": False,
                "omni_coord_address": omni_coord_address,
                "omni_stage_id": omni_stage_id,
                "omni_replica_id": omni_replica_id,
            },
        )
        proc.start()
        self.proc = proc
        self.addresses = addresses
        self.manager_stopped = False
        self.failed_proc_name: str | None = None

        self._wait_until_started(handshake_address, stage_init_timeout)

    @classmethod
    def launch_headless(
        cls,
        *,
        model: str,
        od_config: OmniDiffusionConfig,
        handshake_address: str,
        addresses: EngineZmqAddresses,
        omni_coord_address: str | None,
        omni_stage_id: int,
        omni_replica_id: int,
    ) -> StageDiffusionCoreProcManager:
        """Launch a headless diffusion backend that connects to head-owned sockets."""
        self = cls.__new__(cls)
        ctx = get_mp_context()
        proc = ctx.Process(
            target=StageDiffusionCoreProc.run_diffusion_proc,
            name="StageDiffusionCoreProc",
            kwargs={
                "model": model,
                "od_config": od_config,
                "handshake_address": handshake_address,
                "local_client": False,
                "headless": True,
                "omni_coord_address": omni_coord_address,
                "omni_stage_id": omni_stage_id,
                "omni_replica_id": omni_replica_id,
            },
        )
        proc.start()
        self.proc = proc
        self.addresses = addresses
        self.manager_stopped = False
        self.failed_proc_name = None
        return self

    def shutdown(self, timeout: float | None = None) -> None:
        self.manager_stopped = True
        shutdown([self.proc], timeout=timeout)

    def sentinels(self) -> list[int]:
        return [self.proc.sentinel]

    def finished_procs(self) -> dict[str, int]:
        if self.proc.exitcode is None:
            return {}
        return {self.proc.name: self.proc.exitcode}

    def monitor_engine_liveness(self) -> None:
        try:
            multiprocessing.connection.wait([self.proc.sentinel])
        except Exception:
            return
        if self.proc.exitcode not in (None, 0) and not self.manager_stopped:
            self.failed_proc_name = self.proc.name
        self.shutdown()

    def _wait_until_started(self, handshake_address: str, stage_init_timeout: int) -> None:
        try:
            with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
                wait_for_engine_startup(
                    handshake_socket,
                    self.addresses,
                    [CoreEngine(index=0, local=True)],
                    SimpleNamespace(
                        data_parallel_size_local=1,
                        data_parallel_hybrid_lb=False,
                        data_parallel_external_lb=False,
                    ),
                    False,
                    None,
                    self,
                    None,
                )
        except Exception:
            shutdown([self.proc])
            raise
