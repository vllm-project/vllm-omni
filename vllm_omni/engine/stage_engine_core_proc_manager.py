# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Process manager for omni stage engine subprocesses.

This is a drop-in replacement for vLLM's :class:`CoreEngineProcManager` that
spawns :meth:`StageEngineCoreProc.run_stage_core` instead of the upstream
``EngineCoreProc.run_engine_core``, and forwards omni-specific kwargs
(coordinator address, stage id, per-rank replica id).

Each spawned subprocess corresponds to exactly one omni *replica*: it has its
own ZMQ allocation from :class:`OmniMasterServer` and (when an
``omni_coordinator_address`` is provided) its own
:class:`OmniCoordClientForStage` reporting heartbeat / status.

Liveness monitoring and shutdown are inherited from
:class:`CoreEngineProcManager` unchanged.
"""

from __future__ import annotations

import contextlib
import threading
import weakref
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import numa_utils
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.utils import CoreEngineProcManager
from vllm.v1.executor import Executor
from vllm.v1.utils import shutdown

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

logger = init_logger(__name__)

# ``set_device_control_env_var`` was removed from upstream vllm.
# It was only required for non-CUDA DP; set to None so the existing
# guard at the call-site (vllm_config is not None and ... is not None)
# skips the call transparently.
set_device_control_env_var = None


class StageEngineCoreProcManager(CoreEngineProcManager):
    """Spawn :class:`StageEngineCoreProc` subprocesses with omni kwargs.

    The body mirrors :class:`CoreEngineProcManager.__init__` because the
    upstream class hardcodes ``target=EngineCoreProc.run_engine_core`` and
    does not expose an extensibility hook. The differences from upstream are:

    * ``target`` is :meth:`StageEngineCoreProc.run_stage_core`.
    * Per-rank ``omni_replica_id`` is computed as
      ``base_replica_id + rank_idx`` and added to each subprocess's kwargs.
    * ``omni_coordinator_address`` (if provided) and ``omni_stage_id`` are
      added to every subprocess's kwargs.
    """

    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        *,
        omni_stage_id: int,
        omni_coordinator_address: str | None = None,
        omni_replica_base_id: int = 0,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ) -> None:
        # NOTE: we intentionally do not call ``super().__init__`` — the
        # parent's body hardcodes the wrong target. We re-implement it here
        # while reusing the parent's instance methods (shutdown, monitor).
        if local_engine_count <= 0:
            raise ValueError(f"local_engine_count must be > 0, got {local_engine_count}")

        context = get_mp_context()
        common_kwargs: dict[str, object] = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "tensor_queue": tensor_queue,
            "omni_stage_id": int(omni_stage_id),
            "omni_coordinator_address": omni_coordinator_address,
        }

        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

        # Intra-replica vLLM DP mesh (i.e. ``data_parallel_size`` ranks sharing
        # one engine, one DPCoordinator, one set of weights). Distinct from
        # the omni-level notion of multiple independent replicas of a stage —
        # those each spawn their own StageEngineCoreProcManager and never join
        # a vLLM DP group across replicas.
        has_intra_replica_dp = vllm_config.parallel_config.data_parallel_size > 1

        self.processes: list[BaseProcess] = []
        local_dp_ranks: list[int] = []
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index
            # Each spawned subprocess is one omni replica. The replica id
            # is contiguous within this manager; the master server may have
            # pre-allocated a contiguous block starting at ``omni_replica_base_id``.
            omni_replica_id = omni_replica_base_id + index

            local_dp_ranks.append(local_index)
            self.processes.append(
                context.Process(
                    target=StageEngineCoreProc.run_stage_core,
                    name=(
                        f"StageEngineCoreProc_stage{omni_stage_id}"
                        f"_replica{omni_replica_id}" + (f"_DP{global_index}" if has_intra_replica_dp else "")
                    ),
                    kwargs=common_kwargs
                    | {
                        "dp_rank": global_index,
                        "local_dp_rank": local_index,
                        "omni_replica_id": omni_replica_id,
                    },
                )
            )

        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        self.manager_stopped = threading.Event()
        self.failed_proc_name: str | None = None

        try:
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                device_control_context: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
                if (
                    has_intra_replica_dp
                    and set_device_control_env_var is not None
                    and (not current_platform.is_cuda_alike() or vllm_config.parallel_config.use_ray)
                ):
                    device_control_context = set_device_control_env_var(vllm_config, local_dp_rank)

                with (
                    device_control_context,
                    numa_utils.configure_subprocess(
                        vllm_config,
                        local_rank=0,
                        dp_local_rank=local_dp_rank,
                        process_kind="EngineCore",
                    ),
                ):
                    proc.start()
        finally:
            if self.finished_procs():
                self.shutdown()
