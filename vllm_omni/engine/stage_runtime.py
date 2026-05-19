# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage Runtime implementations for single-node and distributed omni stages."""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import os
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import janus
from omegaconf import OmegaConf
from vllm.logger import init_logger
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.distributed.omni_connectors.utils.initialization import (
    resolve_omni_kv_config_for_stage,
)
from vllm_omni.distributed.omni_coordinator import (
    LeastQueueLengthBalancer,
    LoadBalancer,
    LoadBalancingPolicy,
    RandomBalancer,
    RoundRobinBalancer,
)
from vllm_omni.engine.messages import (
    EngineQueueMessage,
    RegisterRemoteReplicaMessage,
)
from vllm_omni.engine.stage_client import StageClient, StagePoolClient, StagePoolLLMClient
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase
from vllm_omni.engine.stage_engine_startup import (
    OmniMasterServer,
    connect_remote_engine_cores,
    launch_omni_core_engines,
    register_stage_with_omni_master,
)
from vllm_omni.engine.stage_init_utils import (
    LogicalStageInitPlan,
    ReplicaInitPlan,
    _inject_inferred_kv_tp_topology,
    acquire_device_locks,
    acquire_diffusion_device_locks,
    build_diffusion_config,
    build_engine_args_dict,
    build_llm_stage_output_processor,
    build_stage0_input_processor,
    build_vllm_config,
    compute_replica_layout,
    extract_stage_metadata,
    get_stage_connector_spec,
    initialize_diffusion_stage,
    inject_kv_stage_info,
    load_omni_transfer_config_for_model,
    prepare_engine_environment,
    release_device_locks,
    setup_stage_devices,
    terminate_alive_proc,
)
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.utils import inject_omni_kv_config
from vllm_omni.inputs.data import OmniSamplingParams
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
    from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Output modality type (duplicated from async_omni_engine to avoid circular import)
# ---------------------------------------------------------------------------
from vllm_omni.engine.output_modality import FinalOutputModalityType


@dataclass(frozen=True, slots=True)
class StageRuntimeInfo:
    final_output: bool
    final_output_type: FinalOutputModalityType | None
    stage_type: str


@dataclass
class StageRemoteFactoryContext:
    """Per-stage context for dynamic replica attach (distributed mode only)."""

    stage_id: int
    stage_type: str
    stage_cfg: Any
    base_metadata: Any
    vllm_config: Any | None = None
    executor_class: type | None = None
    diffusion_batch_size: int = 1


def _build_load_balancer_factory(policy: str) -> Callable[[], LoadBalancer]:
    try:
        normalized = LoadBalancingPolicy(policy)
    except ValueError as exc:
        valid = ", ".join(p.value for p in LoadBalancingPolicy)
        raise ValueError(f"unknown --omni-lb-policy {policy!r} (valid: {valid})") from exc
    if normalized is LoadBalancingPolicy.RANDOM:
        return RandomBalancer
    if normalized is LoadBalancingPolicy.ROUND_ROBIN:
        return RoundRobinBalancer
    if normalized is LoadBalancingPolicy.LEAST_QUEUE_LENGTH:
        return LeastQueueLengthBalancer
    raise ValueError(f"unhandled load balancing policy {normalized!r}")


# ===========================================================================
# StageRuntime
# ===========================================================================


class StageRuntime:
    """Stage runtime for single-node (non-distributed) mode.

    No coordinator, no master server, no hub. Launches stage processes
    directly and creates StagePool with static clients.
    """

    def __init__(
        self,
        stage_configs: list[Any],
        model: str,
        config_path: str,
        *,
        stage_init_timeout: int,
        diffusion_batch_size: int,
        async_chunk: bool,
        tokenizer: str | None = None,
    ) -> None:
        self._stage_configs = stage_configs
        self._model = model
        self._config_path = config_path
        self._stage_init_timeout = stage_init_timeout
        self._diffusion_batch_size = diffusion_batch_size
        self._async_chunk = async_chunk
        self._tokenizer = tokenizer
        self._num_stages = len(stage_configs)

        # Populated by initialize()
        self.stage_pools: list[StagePool] = []
        self.input_processor: InputProcessor | None = None
        self.stage_metadata: list[StageRuntimeInfo] = []
        self.default_sampling_params_list: list[OmniSamplingParams] = []
        self.prompt_expand_func: Any = None
        self.supported_tasks: tuple[str, ...] = ("generate",)

    @property
    def stage_clients(self) -> list[StageClient]:
        return [cast(StageClient, pool.stage_client) for pool in self.stage_pools]

    @property
    def stage_vllm_configs(self) -> list[Any]:
        return [pool.stage_vllm_config for pool in self.stage_pools]

    @property
    def output_processors(self) -> list[Any]:
        return [pool.output_processor for pool in self.stage_pools]

    @staticmethod
    def _client_addresses_from_zmq(addresses: Any) -> dict[str, str]:
        client_addresses = {
            "input_address": addresses.inputs[0],
            "output_address": addresses.outputs[0],
        }
        if addresses.frontend_stats_publish_address is not None:
            client_addresses["stats_update_address"] = addresses.frontend_stats_publish_address
        return client_addresses

    @staticmethod
    def _cleanup_launched_llm_resources(
        *,
        stage_id: int,
        proc: Any = None,
        engine_manager: Any = None,
        coordinator: Any = None,
    ) -> None:
        """Release launch-only LLM resources when client creation never completed."""
        if proc is not None:
            try:
                terminate_alive_proc(proc)
            except Exception as cleanup_error:
                logger.warning(
                    "[StageRuntime] Failed to terminate process for stage %s: %s",
                    stage_id,
                    cleanup_error,
                )

        for resource, resource_name in (
            (engine_manager, "engine manager"),
            (coordinator, "coordinator"),
        ):
            if resource is None:
                continue
            shutdown = getattr(resource, "shutdown", None)
            close = getattr(resource, "close", None)
            try:
                if callable(shutdown):
                    shutdown()
                elif callable(close):
                    close()
            except Exception as cleanup_error:
                logger.warning(
                    "[StageRuntime] Failed to cleanup launched %s for stage %s: %s",
                    resource_name,
                    stage_id,
                    cleanup_error,
                )

    @staticmethod
    def _collect_initialized_clients_for_cleanup(
        stage_pools: Sequence[StagePool],
        initialized_clients_by_stage: Mapping[int, Sequence[StagePoolClient | None]],
    ) -> list[StageClient]:
        """Collect initialized clients exactly once for failure cleanup."""
        collected: list[StageClient] = []
        seen: set[int] = set()

        def _add_client(client: StageClient | None) -> None:
            if client is None:
                return
            client_id = id(client)
            if client_id in seen:
                return
            seen.add(client_id)
            collected.append(client)

        for pool in stage_pools:
            for client in getattr(pool, "clients", ()):
                _add_client(client)

        for clients in initialized_clients_by_stage.values():
            for client in clients:
                _add_client(client)

        return collected

    @staticmethod
    def _shutdown_initialized_clients(clients: Sequence[StageClient]) -> None:
        """Best-effort shutdown for attached clients after init failure."""
        for client in reversed(list(clients)):
            if client is None:
                continue
            try:
                client.shutdown()
            except Exception as cleanup_error:
                logger.warning(
                    "[StageRuntime] Failed to shutdown initialized client after init failure: %s",
                    cleanup_error,
                )

    def initialize(self) -> None:
        """Run the full stage initialization sequence."""
        stage_plans = self._prepare_stage_plans()
        initialized_clients_by_stage: dict[int, list[StagePoolClient | None]] = {
            plan.stage_idx: [None] * len(plan.replicas) for plan in stage_plans
        }
        stage_pools: list[StagePool] = []
        try:
            initialized_clients = self._initialize_stage_replicas(stage_plans, self._stage_init_timeout)
            initialized_clients_by_stage = initialized_clients
            self._finalize_initialized_stages(stage_plans, initialized_clients)
        except Exception as exc:
            initialized_clients_by_stage = getattr(
                exc,
                "_initialized_clients_by_stage",
                initialized_clients_by_stage,
            )
            cleanup_clients = self._collect_initialized_clients_for_cleanup(
                stage_pools,
                initialized_clients_by_stage,
            )
            logger.exception(
                "[StageRuntime] Stage initialization failed; shutting down %s initialized client(s)",
                len(cleanup_clients),
            )
            self._shutdown_initialized_clients(cleanup_clients)
            raise exc

    def shutdown(self) -> None:
        for pool in self.stage_pools:
            for client in pool.clients:
                if client is not None and hasattr(client, "shutdown"):
                    try:
                        client.shutdown()
                    except Exception:
                        logger.warning("[StageRuntime] client shutdown failed", exc_info=True)

    def _prepare_stage_plans(self) -> list[LogicalStageInitPlan]:
        """Build logical stage plans and cache prompt expansion metadata."""
        replicas_per_stage, replica_devices_map = compute_replica_layout(self._stage_configs)
        prepare_engine_environment()
        omni_transfer_config = load_omni_transfer_config_for_model(self._model, self._config_path)
        stage_plans, self.prompt_expand_func = self._build_logical_stage_init_plans(
            omni_transfer_config,
            replicas_per_stage,
            replica_devices_map,
        )
        return stage_plans

    def _finalize_initialized_stages(
        self,
        stage_plans: Sequence[LogicalStageInitPlan],
        initialized_clients: Mapping[int, Sequence[StagePoolClient | None]],
    ) -> None:
        """Populate runtime fields after replica initialization succeeds."""
        if stage_plans and stage_plans[0].replicas[0].metadata.stage_type != "diffusion":
            stage0_vllm_config = stage_plans[0].replicas[0].stage_vllm_config
            assert stage0_vllm_config is not None
            self.input_processor = build_stage0_input_processor(stage0_vllm_config)
        self.stage_pools = self._assemble_stage_pools(stage_plans, initialized_clients)
        self._derive_metadata()

    @contextmanager
    def _stage_device_scope(self, stage_id: int, runtime_cfg: Any) -> Iterator[None]:
        """Temporarily apply the stage device env while launching a replica."""
        device_control_env = current_omni_platform.device_control_env_var
        previous_visible_devices = os.environ.get(device_control_env)
        try:
            setup_stage_devices(stage_id, runtime_cfg)
            yield
        finally:
            if previous_visible_devices is None:
                current_omni_platform.unset_device_control_env_var()
            else:
                current_omni_platform.set_device_control_env_var(previous_visible_devices)

    @contextmanager
    def _local_llm_launch_scope(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> Iterator[tuple[Any, type, list[int]]]:
        """Prepare config + hold device env/locks while launching a local LLM replica."""
        with self._stage_device_scope(plan.metadata.stage_id, plan.metadata.runtime_cfg):
            vllm_config = plan.stage_vllm_config
            executor_class = plan.executor_class
            assert vllm_config is not None
            assert executor_class is not None
            engine_args_dict = build_engine_args_dict(
                plan.stage_cfg,
                self._model,
                stage_connector_spec=plan.stage_connector_spec,
                cli_tokenizer=self._tokenizer,
            )
            lock_fds = acquire_device_locks(
                plan.metadata.stage_id,
                engine_args_dict,
                stage_init_timeout,
            )
            yield vllm_config, executor_class, lock_fds

    # ---- Internal methods (moved from AsyncOmniEngine) ----

    def _build_logical_stage_init_plans(
        self,
        omni_transfer_config: Any,
        replicas_per_stage: Sequence[int],
        replica_devices_map: Mapping[int, Sequence[str]],
    ) -> tuple[list[LogicalStageInitPlan], Any]:
        """Build startup plans for every logical stage and replica."""
        prompt_expand_func = None
        stage_plans: list[LogicalStageInitPlan] = []

        for stage_idx, stage_cfg in enumerate(self._stage_configs):
            base_metadata = extract_stage_metadata(stage_cfg)
            configured_stage_id = base_metadata.stage_id
            if base_metadata.prompt_expand_func is not None:
                prompt_expand_func = base_metadata.prompt_expand_func

            stage_connector_spec = get_stage_connector_spec(
                omni_transfer_config=omni_transfer_config,
                stage_id=configured_stage_id,
                async_chunk=self._async_chunk,
            )
            omni_kv_connector = resolve_omni_kv_config_for_stage(omni_transfer_config, configured_stage_id)
            num_replicas = replicas_per_stage[stage_idx]
            launch_mode = self._get_launch_mode(configured_stage_id)

            replicas: list[ReplicaInitPlan] = []
            stage_vllm_config = None
            executor_class = None
            if base_metadata.stage_type != "diffusion":
                engine_args_dict = build_engine_args_dict(
                    stage_cfg,
                    self._model,
                    stage_connector_spec=stage_connector_spec,
                    cli_tokenizer=self._tokenizer,
                )
                omni_conn_cfg, omni_from, omni_to = omni_kv_connector
                if omni_conn_cfg:
                    omni_kv = engine_args_dict.get("omni_kv_config") or {}
                    if not isinstance(omni_kv, dict):
                        omni_kv = dict(omni_kv)
                    omni_kv["connector_config"] = omni_conn_cfg
                    omni_kv["omni_from_stage"] = omni_from
                    omni_kv["omni_to_stage"] = omni_to
                    omni_kv.setdefault("stage_id", configured_stage_id)
                    engine_args_dict["omni_kv_config"] = omni_kv
                if self._stage_configs:
                    _inject_inferred_kv_tp_topology(
                        engine_args_dict.get("omni_kv_config"),
                        configured_stage_id,
                        self._stage_configs,
                    )
                stage_vllm_config, executor_class = build_vllm_config(
                    stage_cfg,
                    self._model,
                    stage_connector_spec=stage_connector_spec,
                    engine_args_dict=engine_args_dict,
                )

            for replica_id in range(num_replicas):
                replica_cfg = copy.deepcopy(stage_cfg) if replica_id > 0 else stage_cfg
                if stage_idx in replica_devices_map:
                    replica_cfg.runtime.devices = replica_devices_map[stage_idx][replica_id]

                replica_metadata = extract_stage_metadata(replica_cfg)
                replica_metadata.replica_id = replica_id
                if launch_mode == "remote" and replica_metadata.stage_type != "diffusion":
                    replica_metadata.runtime_cfg = None

                replicas.append(
                    ReplicaInitPlan(
                        replica_id=replica_id,
                        num_replicas=num_replicas,
                        launch_mode=launch_mode,
                        stage_cfg=replica_cfg,
                        metadata=replica_metadata,
                        stage_connector_spec=stage_connector_spec,
                        omni_kv_connector=omni_kv_connector,
                        stage_vllm_config=stage_vllm_config,
                        executor_class=executor_class,
                    )
                )

            stage_plans.append(
                LogicalStageInitPlan(
                    stage_idx=stage_idx,
                    configured_stage_id=configured_stage_id,
                    replicas=replicas,
                )
            )

        return stage_plans, prompt_expand_func

    def _get_launch_mode(self, configured_stage_id: int) -> str:
        """Determine launch mode for a stage. Overridden by DistStageRuntime."""
        return "local"

    def _initialize_stage_replicas(
        self,
        stage_plans: Sequence[LogicalStageInitPlan],
        stage_init_timeout: int,
    ) -> dict[int, list[StagePoolClient | None]]:
        """Initialize all stage replicas (diffusion inline, LLM parallel).

        Stages sharing the same GPU are initialized sequentially to avoid
        memory profiling interference. Stages on different GPUs are
        initialized in parallel.
        """
        stage_launch_lock = threading.Lock()
        initialized_clients_by_stage: dict[int, list[StagePoolClient | None]] = {
            plan.stage_idx: [None] * len(plan.replicas) for plan in stage_plans
        }
        primary_exc: Exception | None = None

        diffusion_replicas: list[tuple[int, ReplicaInitPlan]] = []
        llm_replicas: list[tuple[int, ReplicaInitPlan]] = []
        for plan in stage_plans:
            for replica in plan.replicas:
                if replica.metadata.stage_type == "diffusion":
                    diffusion_replicas.append((plan.stage_idx, replica))
                else:
                    llm_replicas.append((plan.stage_idx, replica))

        for stage_idx, replica in diffusion_replicas:
            try:
                initialized_clients_by_stage[stage_idx][replica.replica_id] = self._initialize_replica(
                    replica, stage_init_timeout, stage_launch_lock
                )
            except Exception as exc:
                primary_exc = exc
                break

        if primary_exc is None and llm_replicas:
            # Group replicas by device so co-located stages init sequentially.
            device_groups: dict[str, list[tuple[int, ReplicaInitPlan]]] = {}
            for stage_idx, replica in llm_replicas:
                runtime_cfg = replica.metadata.runtime_cfg or {}
                devices_key = str(
                    runtime_cfg.get("devices") if hasattr(runtime_cfg, "get")
                    else getattr(runtime_cfg, "devices", None)
                )
                device_groups.setdefault(devices_key, []).append((stage_idx, replica))

            def _init_device_group(group: list[tuple[int, ReplicaInitPlan]]) -> None:
                """Initialize replicas in a device group sequentially."""
                nonlocal primary_exc
                for stage_idx, replica in group:
                    if primary_exc is not None:
                        return
                    try:
                        client = self._initialize_replica(
                            replica, stage_init_timeout, stage_launch_lock
                        )
                        initialized_clients_by_stage[stage_idx][replica.replica_id] = client
                    except Exception as exc:
                        if primary_exc is None:
                            primary_exc = exc
                        return

            if len(device_groups) == 1:
                # All on same device — just run sequentially
                _init_device_group(list(device_groups.values())[0])
            else:
                # Different devices can init in parallel
                future_to_group: dict[concurrent.futures.Future[None], str] = {}
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(device_groups),
                    thread_name_prefix="stage-init",
                ) as init_executor:
                    for dev_key, group in device_groups.items():
                        future = init_executor.submit(_init_device_group, group)
                        future_to_group[future] = dev_key

                    for future in concurrent.futures.as_completed(future_to_group):
                        try:
                            future.result()
                        except Exception as exc:
                            if primary_exc is None:
                                primary_exc = exc

        if primary_exc is not None:
            setattr(primary_exc, "_initialized_clients_by_stage", initialized_clients_by_stage)
            raise primary_exc

        return initialized_clients_by_stage

    def _initialize_replica(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
        stage_launch_lock: threading.Lock,
    ) -> StagePoolClient:
        if plan.metadata.stage_type == "diffusion":
            return self._initialize_diffusion_replica(plan, stage_init_timeout, stage_launch_lock)
        return self._initialize_llm_replica(plan, stage_init_timeout, stage_launch_lock)

    def _initialize_llm_replica(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
        llm_stage_launch_lock: threading.Lock,
    ) -> StageEngineCoreClientBase:
        """Initialize one LLM replica using vLLM's launch_core_engines pattern."""
        proc = None
        engine_manager = None
        coordinator = None
        stage_client = None
        lock_fds: list[int] = []
        try:
            if plan.launch_mode == "remote":
                stage_client = self._initialize_llm_replica_remote(plan, stage_init_timeout)
            else:
                with llm_stage_launch_lock:
                    with self._local_llm_launch_scope(
                        plan,
                        stage_init_timeout,
                    ) as prepared:
                        vllm_config, executor_class, lock_fds = prepared
                        engine_manager, coordinator, addresses = self._launch_local_llm_replica(
                            plan,
                            vllm_config,
                            executor_class,
                        )

                logger.info("[StageRuntime] Stage %s engine startup completed", plan.metadata.stage_id)
                stage_client = StageEngineCoreClientBase.make_async_mp_client(
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    metadata=plan.metadata,
                    client_addresses=self._client_addresses_from_zmq(addresses),
                    engine_manager=engine_manager,
                    coordinator=coordinator,
                )

            logger.info("[StageRuntime] Stage %s initialized", plan.metadata.stage_id)
            return stage_client
        except Exception:
            if stage_client is not None:
                try:
                    stage_client.shutdown()
                except Exception as cleanup_error:
                    logger.warning(
                        "[StageRuntime] Failed to cleanup stage %s after attach failure: %s",
                        plan.metadata.stage_id,
                        cleanup_error,
                    )
            else:
                self._cleanup_launched_llm_resources(
                    stage_id=plan.metadata.stage_id,
                    proc=proc,
                    engine_manager=engine_manager,
                    coordinator=coordinator,
                )
            raise
        finally:
            if lock_fds:
                release_device_locks(lock_fds)

    def _launch_local_llm_replica(
        self,
        plan: ReplicaInitPlan,
        vllm_config: Any,
        executor_class: type,
    ) -> tuple[Any, Any, Any]:
        """Launch a local LLM replica and return client resources."""
        from vllm_omni.engine.omni_core_engine_proc_manager import OmniCoreEngineProcManager
        from vllm.v1.engine.utils import (
            CoreEngine,
            get_engine_zmq_addresses,
            wait_for_engine_startup,
        )
        from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx

        import zmq

        # Spawn the engine subprocess while CUDA_VISIBLE_DEVICES is still set
        # to this stage's devices. The child inherits the env at proc.start().
        addresses = get_engine_zmq_addresses(vllm_config)
        handshake_address = get_open_zmq_ipc_path()
        engines_to_handshake = [CoreEngine(index=0, local=True)]
        engine_manager = OmniCoreEngineProcManager(
            local_engine_count=1,
            start_index=0,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=True,
            handshake_address=handshake_address,
            executor_class=executor_class,
            log_stats=False,
            omni_stage_id=plan.metadata.stage_id,
            omni_coordinator_address=self._get_coordinator_address(),
            omni_replica_base_id=plan.replica_id,
        )
        with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
            wait_for_engine_startup(
                handshake_socket,
                addresses,
                engines_to_handshake,
                vllm_config.parallel_config,
                False,  # coordinated_dp
                vllm_config.cache_config,
                engine_manager,
                None,  # coordinator_proc
            )
        return engine_manager, None, addresses

    def _get_coordinator_address(self) -> str | None:
        """Return coordinator router address. Overridden by DistStageRuntime."""
        return None

    def _initialize_llm_replica_remote(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> StageEngineCoreClientBase:
        """Initialize a remote LLM replica. Only used in distributed mode."""
        raise NotImplementedError("Remote replicas require DistStageRuntime")

    def _initialize_diffusion_replica(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
        stage_launch_lock: threading.Lock,
    ) -> Any:
        """Initialize one diffusion replica end-to-end."""
        client = None
        proc = None
        lock_fds: list[int] = []
        try:
            if plan.launch_mode == "remote":
                client = self._initialize_diffusion_replica_remote(plan, stage_init_timeout)
            else:
                with stage_launch_lock:
                    with self._stage_device_scope(plan.metadata.stage_id, plan.metadata.runtime_cfg):
                        omni_conn_cfg, omni_from, omni_to = plan.omni_kv_connector
                        if omni_conn_cfg:
                            inject_omni_kv_config(plan.stage_cfg, omni_conn_cfg, omni_from, omni_to)
                        inject_kv_stage_info(
                            plan.stage_cfg, plan.metadata.stage_id, self._stage_configs
                        )
                        client, proc, lock_fds = self._launch_diffusion_local(plan, stage_init_timeout)

            logger.info(
                "[StageRuntime] Stage %s replica %s initialized (diffusion, batch_size=%d)",
                plan.metadata.stage_id, plan.replica_id, self._diffusion_batch_size,
            )
            return client
        except Exception:
            if proc is not None:
                terminate_alive_proc(proc)
            raise
        finally:
            if lock_fds:
                release_device_locks(lock_fds)

    def _launch_diffusion_local(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> tuple[Any, Any, list[int]]:
        """Launch a local diffusion replica. Returns (client, proc, lock_fds).

        Overridden by DistStageRuntime for coordinator-aware launch.
        """
        client = initialize_diffusion_stage(
            plan.metadata.stage_id,
            self._model,
            plan.stage_cfg,
            plan.metadata,
            stage_init_timeout=stage_init_timeout,
            batch_size=self._diffusion_batch_size,
            use_inline=self._num_stages == 1 and plan.num_replicas == 1,
        )
        return client, None, []

    def _initialize_diffusion_replica_remote(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> Any:
        """Initialize a remote diffusion replica. Only used in distributed mode."""
        raise NotImplementedError("Remote replicas require DistStageRuntime")

    def _assemble_stage_pools(
        self,
        stage_plans: Sequence[LogicalStageInitPlan],
        initialized_clients_by_stage: Mapping[int, Sequence[StagePoolClient | None]],
    ) -> list[StagePool]:
        """Assemble logical stage pools."""
        stage_pools: list[StagePool] = []

        for plan in stage_plans:
            replica_clients = initialized_clients_by_stage[plan.stage_idx]
            first_client = replica_clients[0] if replica_clients else None
            if first_client is None:
                raise RuntimeError(f"Stage {plan.stage_idx} initialization completed with a missing client")

            clients: list[StagePoolClient] = [client for client in replica_clients if client is not None]
            stage_vllm_config = None
            output_processor = None
            if plan.replicas[0].metadata.stage_type != "diffusion":
                stage_vllm_config = plan.replicas[0].stage_vllm_config
                assert stage_vllm_config is not None
                output_processor = build_llm_stage_output_processor(plan, stage_vllm_config)

            stage_pools.append(
                StagePool(
                    plan.stage_idx,
                    clients,
                    output_processor=output_processor,
                    stage_vllm_config=stage_vllm_config,
                )
            )

        return stage_pools

    def _derive_metadata(self) -> None:
        """Derive stage metadata and supported tasks from assembled pools."""
        metadata_list: list[StageRuntimeInfo] = []
        sampling_params_list: list[OmniSamplingParams] = []
        supported_tasks: set[str] = set()

        for pool in self.stage_pools:
            client = pool.stage_client
            if client is None:
                continue
            metadata_list.append(
                StageRuntimeInfo(
                    final_output=client.final_output,
                    final_output_type=client.final_output_type,
                    stage_type=client.stage_type,
                )
            )
            sampling_params_list.append(client.default_sampling_params)
            if getattr(client, "is_comprehension", False):
                supported_tasks.add("generate")

        for m in metadata_list:
            if m.final_output_type == "audio":
                supported_tasks.add("speech")

        self.stage_metadata = metadata_list
        self.default_sampling_params_list = sampling_params_list
        self.supported_tasks = tuple(supported_tasks) if supported_tasks else ("generate",)


# ===========================================================================
# DistStageRuntime
# ===========================================================================


class DistStageRuntime(StageRuntime):
    """Stage runtime for distributed (single_stage_mode) deployment.

    Extends StageRuntime with:
    - OmniCoordinatorRuntime (independent process)
    - OmniMasterServer for replica registration
    - Remote replica support
    - Dynamic membership via MembershipController
    """

    def __init__(
        self,
        stage_configs: list[Any],
        model: str,
        config_path: str,
        *,
        stage_init_timeout: int,
        diffusion_batch_size: int,
        async_chunk: bool,
        tokenizer: str | None = None,
        single_stage_id_filter: int | None,
        omni_master_address: str,
        omni_master_port: int,
        omni_dp_size_local: int = 1,
        omni_heartbeat_timeout: float = 30.0,
        omni_lb_policy: str = "random",
        request_queue: janus.Queue[EngineQueueMessage] | None = None,
    ) -> None:
        super().__init__(
            stage_configs=stage_configs,
            model=model,
            config_path=config_path,
            stage_init_timeout=stage_init_timeout,
            diffusion_batch_size=diffusion_batch_size,
            async_chunk=async_chunk,
            tokenizer=tokenizer,
        )
        self._single_stage_id_filter = single_stage_id_filter
        self._omni_master_address = omni_master_address
        self._omni_master_port = omni_master_port
        self._omni_heartbeat_timeout = omni_heartbeat_timeout
        self._omni_lb_policy = omni_lb_policy
        self._request_queue = request_queue

        self._omni_master_server: OmniMasterServer | None = None
        self._coordinator_runtime: Any | None = None
        self._omni_dp_size_local = omni_dp_size_local
        self._stage_remote_factory_contexts: dict[int, StageRemoteFactoryContext] = {}

    @property
    def coordinator_pub_address(self) -> str | None:
        if self._coordinator_runtime is not None:
            return self._coordinator_runtime.pub_address
        return None

    @property
    def load_balancer_factory(self) -> Callable[[], LoadBalancer]:
        return _build_load_balancer_factory(self._omni_lb_policy)

    @property
    def remote_replica_factory(self) -> Callable[[int, int], Awaitable[Any]] | None:
        if self._coordinator_runtime is not None:
            return self._build_remote_replica
        return None

    def _prepare_stage_plans(self) -> list[LogicalStageInitPlan]:
        self._validate_single_stage_mode_replica_constraints()
        return super()._prepare_stage_plans()

    def _validate_single_stage_mode_replica_constraints(self) -> None:
        """Apply --omni-dp-size-local to the local stage's runtime.num_replicas."""
        target_stage_id = self._single_stage_id_filter
        if target_stage_id is None:
            return

        for idx, stage_cfg in enumerate(self._stage_configs):
            stage_id = int(getattr(stage_cfg, "stage_id", idx))
            runtime_cfg = getattr(stage_cfg, "runtime", None)
            if runtime_cfg is None:
                continue
            if stage_id == target_stage_id:
                try:
                    runtime_cfg.num_replicas = self._omni_dp_size_local
                except Exception:
                    if hasattr(runtime_cfg, "__setitem__"):
                        runtime_cfg["num_replicas"] = self._omni_dp_size_local

    def initialize(self) -> None:
        """Run the full distributed stage initialization sequence."""
        stage_plans = self._prepare_stage_plans()
        self._stage_remote_factory_contexts = self._capture_stage_factory_contexts(stage_plans)
        self._start_omni_master_server(stage_plans)
        initialized_clients_by_stage: dict[int, list[StagePoolClient | None]] = {
            plan.stage_idx: [None] * len(plan.replicas) for plan in stage_plans
        }
        stage_pools: list[StagePool] = []
        try:
            initialized_clients = self._initialize_stage_replicas(stage_plans, self._stage_init_timeout)
            initialized_clients_by_stage = initialized_clients
            self._finalize_initialized_stages(stage_plans, initialized_clients)
        except Exception as exc:
            initialized_clients_by_stage = getattr(
                exc,
                "_initialized_clients_by_stage",
                initialized_clients_by_stage,
            )
            cleanup_clients = self._collect_initialized_clients_for_cleanup(
                stage_pools,
                initialized_clients_by_stage,
            )
            self._shutdown_initialized_clients(cleanup_clients)
            self._cleanup_distributed_infra()
            raise exc

    def shutdown(self) -> None:
        super().shutdown()
        self._cleanup_distributed_infra()

    def _cleanup_distributed_infra(self) -> None:
        if self._omni_master_server is not None:
            try:
                self._omni_master_server.stop()
            except Exception:
                logger.warning("[DistStageRuntime] master server stop failed", exc_info=True)
            self._omni_master_server = None
        if self._coordinator_runtime is not None:
            try:
                self._coordinator_runtime.close()
            except Exception:
                logger.warning("[DistStageRuntime] coordinator close failed", exc_info=True)
            self._coordinator_runtime = None

    # ---- Distributed overrides ----

    def _get_launch_mode(self, configured_stage_id: int) -> str:
        if self._single_stage_id_filter is not None and configured_stage_id != self._single_stage_id_filter:
            return "remote"
        return "local"

    def _get_coordinator_address(self) -> str | None:
        if self._coordinator_runtime is not None:
            return self._coordinator_runtime.router_address
        return None

    def _initialize_llm_replica_remote(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> StageEngineCoreClientBase:
        """Initialize a remote LLM replica via OmniMasterServer handshake."""
        assert self._omni_master_server is not None
        engine_manager = None
        coordinator = None
        try:
            raw_stage_cfg = self._omni_master_server.get_stage_config(
                plan.metadata.stage_id, timeout_s=stage_init_timeout, replica_id=plan.replica_id
            )
            if raw_stage_cfg is None:
                raise ValueError(f"Remote stage {plan.metadata.stage_id} registered without stage config")
            vllm_config = plan.stage_vllm_config
            executor_class = plan.executor_class
            assert vllm_config is not None
            assert executor_class is not None
            logger.info("[DistStageRuntime] Stage %s remote engine handshake started", plan.metadata.stage_id)
            engine_manager, coordinator, addresses = self._connect_remote_llm_replica(
                plan.metadata.stage_id,
                plan.replica_id,
                vllm_config,
            )
            logger.info("[DistStageRuntime] Stage %s remote engine startup completed", plan.metadata.stage_id)
            client_addresses = self._client_addresses_from_zmq(addresses)
            replica_host = self._omni_master_server.get_replica_host(plan.metadata.stage_id, plan.replica_id)
            if replica_host:
                client_addresses["replica_host"] = replica_host
            return StageEngineCoreClientBase.make_async_mp_client(
                vllm_config=vllm_config,
                executor_class=executor_class,
                metadata=plan.metadata,
                client_addresses=client_addresses,
                engine_manager=engine_manager,
                coordinator=coordinator,
            )
        except Exception:
            self._cleanup_launched_llm_resources(
                stage_id=plan.metadata.stage_id,
                engine_manager=engine_manager,
                coordinator=coordinator,
            )
            raise

    def _initialize_diffusion_replica_remote(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> Any:
        """Initialize a remote diffusion replica via OmniMasterServer."""
        assert self._omni_master_server is not None
        remote_stage_cfg = OmegaConf.create(
            self._omni_master_server.get_stage_config(
                plan.metadata.stage_id, timeout_s=stage_init_timeout, replica_id=plan.replica_id
            )
        )
        remote_metadata = extract_stage_metadata(remote_stage_cfg)
        addresses = self._omni_master_server.get_zmq_addresses(plan.metadata.stage_id, replica_id=plan.replica_id)
        logger.info("[DistStageRuntime] Stage %s remote diffusion startup completed", plan.metadata.stage_id)
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

        return StageDiffusionClient.from_addresses(
            remote_metadata,
            request_address=addresses.inputs[0],
            response_address=addresses.outputs[0],
            batch_size=self._diffusion_batch_size,
        )

    def _launch_diffusion_local(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> tuple[Any, Any, list[int]]:
        """Launch a local diffusion replica with coordinator awareness."""
        from vllm_omni.diffusion.stage_diffusion_proc import (
            complete_diffusion_handshake,
            spawn_diffusion_proc,
        )

        assert self._omni_master_server is not None
        od_config = build_diffusion_config(self._model, plan.stage_cfg, plan.metadata)
        lock_fds = acquire_diffusion_device_locks(plan.metadata.stage_id, od_config, stage_init_timeout)
        handshake_address, request_address, response_address = register_stage_with_omni_master(
            omni_master_address=self._omni_master_server.address,
            omni_master_port=self._omni_master_server.port,
            omni_stage_id=plan.metadata.stage_id,
            omni_stage_config=plan.stage_cfg,
            return_addresses=True,
            replica_id=plan.replica_id,
        )
        coord_router_addr: str | None = (
            self._coordinator_runtime.router_address if self._coordinator_runtime is not None else None
        )
        proc = None
        try:
            proc, _, _, _ = spawn_diffusion_proc(
                self._model,
                od_config,
                handshake_address=handshake_address,
                request_address=request_address,
                response_address=response_address,
                omni_coordinator_address=coord_router_addr,
                omni_stage_id=plan.metadata.stage_id,
                omni_replica_id=plan.replica_id,
            )
            complete_diffusion_handshake(proc, handshake_address, stage_init_timeout)
        except Exception:
            if proc is not None:
                terminate_alive_proc(proc)
            raise
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

        client = StageDiffusionClient.from_addresses(
            plan.metadata,
            request_address=request_address,
            response_address=response_address,
            proc=proc,
            batch_size=self._diffusion_batch_size,
        )
        return client, proc, lock_fds

    # ---- Distributed infrastructure ----

    def _start_omni_master_server(self, stage_plans: Sequence[LogicalStageInitPlan]) -> None:
        """Start OmniMasterServer and OmniCoordinatorRuntime."""
        if not self._omni_master_address or not self._omni_master_port:
            raise ValueError(
                "AsyncOmniEngine single_stage_mode requires both omni_master_address and "
                "omni_master_port to be set."
            )

        from vllm_omni.distributed.omni_coordinator import OmniCoordinatorRuntime

        all_stage_ids: list[int] = []
        stage_replica_counts: dict[int, int] = {}
        head_local_replicas: dict[int, list[int]] = {}
        seen_stage_ids: set[int] = set()
        for plan in stage_plans:
            stage_id = plan.configured_stage_id
            if stage_id in seen_stage_ids:
                raise ValueError(f"Duplicate stage_id {stage_id!r} detected")
            seen_stage_ids.add(stage_id)
            all_stage_ids.append(stage_id)
            stage_replica_counts[stage_id] = len(plan.replicas)
            local_rids = [rep.replica_id for rep in plan.replicas if rep.launch_mode == "local"]
            if local_rids:
                head_local_replicas[stage_id] = local_rids

        self._coordinator_runtime = OmniCoordinatorRuntime(
            host=self._omni_master_address,
            heartbeat_timeout=self._omni_heartbeat_timeout,
        )

        self._omni_master_server = OmniMasterServer(
            master_address=self._omni_master_address,
            master_port=self._omni_master_port,
            stage_ids=all_stage_ids,
            stage_replica_counts=stage_replica_counts,
            coordinator_router_address=self._coordinator_runtime.router_address,
            on_register=self._dispatch_master_register,
            head_local_replicas=head_local_replicas,
        )
        self._omni_master_server.start()
        logger.info("[DistStageRuntime] OmniMasterServer started for stages %s", all_stage_ids)

    def _capture_stage_factory_contexts(
        self, stage_plans: Sequence[LogicalStageInitPlan]
    ) -> dict[int, StageRemoteFactoryContext]:
        """Snapshot per-stage construction context for dynamic replica attach."""
        contexts: dict[int, StageRemoteFactoryContext] = {}
        for plan in stage_plans:
            if not plan.replicas:
                continue
            template = plan.replicas[0]
            stage_id = int(plan.configured_stage_id)
            stage_type = template.metadata.stage_type or "llm"
            contexts[stage_id] = StageRemoteFactoryContext(
                stage_id=stage_id,
                stage_type=stage_type,
                stage_cfg=template.stage_cfg,
                base_metadata=template.metadata,
                vllm_config=template.stage_vllm_config,
                executor_class=template.executor_class,
                diffusion_batch_size=self._diffusion_batch_size,
            )
        return contexts

    def _dispatch_master_register(self, stage_id: int, replica_id: int, alloc: Any) -> None:
        """Callback from OmniMasterServer when a headless replica registers."""
        if self._request_queue is None:
            logger.warning("[DistStageRuntime] on_register fired but no request_queue wired")
            return
        try:
            self._request_queue.sync_q.put_nowait(
                RegisterRemoteReplicaMessage(stage_id=stage_id, replica_id=replica_id)
            )
        except Exception:
            logger.exception("[DistStageRuntime] Failed to enqueue register message")

    async def _build_remote_replica(self, stage_id: int, replica_id: int) -> Any:
        """Build a head-side client for a newly registered remote replica."""
        ctx = self._stage_remote_factory_contexts.get(stage_id)
        if ctx is None:
            raise ValueError(f"No factory context for stage {stage_id}")

        if ctx.stage_type == "diffusion":
            assert self._omni_master_server is not None
            addresses = self._omni_master_server.get_zmq_addresses(stage_id, replica_id=replica_id)
            metadata = copy.deepcopy(ctx.base_metadata)
            metadata.replica_id = replica_id
            from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

            return StageDiffusionClient.from_addresses(
                metadata,
                request_address=addresses.inputs[0],
                response_address=addresses.outputs[0],
                batch_size=ctx.diffusion_batch_size,
            )

        # LLM replica
        assert ctx.vllm_config is not None
        assert ctx.executor_class is not None
        vllm_config = copy.deepcopy(ctx.vllm_config)
        engine_manager, coordinator, addresses = self._connect_remote_llm_replica(
            stage_id,
            replica_id,
            vllm_config,
        )
        metadata = copy.deepcopy(ctx.base_metadata)
        metadata.replica_id = replica_id
        client_addresses = self._client_addresses_from_zmq(addresses)
        replica_host = self._omni_master_server.get_replica_host(stage_id, replica_id)
        if replica_host:
            client_addresses["replica_host"] = replica_host
        return StageEngineCoreClientBase.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=ctx.executor_class,
            metadata=metadata,
            client_addresses=client_addresses,
            engine_manager=engine_manager,
            coordinator=coordinator,
        )

    def _connect_remote_llm_replica(
        self,
        stage_id: int,
        replica_id: int,
        vllm_config: Any,
    ) -> tuple[Any, Any, Any]:
        """Handshake with a remote LLM replica and return client resources."""
        assert self._omni_master_server is not None
        vllm_config.parallel_config.data_parallel_size_local = 0
        with connect_remote_engine_cores(
            vllm_config=vllm_config,
            omni_master_server=self._omni_master_server,
            stage_id=stage_id,
            replica_id=replica_id,
        ) as remote_resources:
            engine_manager, coordinator, addresses, _ = remote_resources
        return engine_manager, coordinator, addresses

    def _launch_local_llm_replica(
        self,
        plan: ReplicaInitPlan,
        vllm_config: Any,
        executor_class: type,
    ) -> tuple[Any, Any, Any]:
        """Launch a distributed local LLM replica via OmniMasterServer."""
        assert self._omni_master_server is not None
        with launch_omni_core_engines(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,
            omni_master_server=self._omni_master_server,
            stage_id=plan.metadata.stage_id,
            stage_config=plan.stage_cfg,
            replica_id=plan.replica_id,
            omni_coordinator_address=self._get_coordinator_address(),
        ) as resources:
            engine_manager, coordinator, addresses = resources
        return engine_manager, coordinator, addresses


# ===========================================================================
# Factory
# ===========================================================================


def create_stage_runtime(
    stage_configs: list[Any],
    model: str,
    config_path: str,
    *,
    single_stage_mode: bool,
    stage_init_timeout: int,
    diffusion_batch_size: int,
    async_chunk: bool,
    tokenizer: str | None = None,
    # Distributed-only params:
    single_stage_id_filter: int | None = None,
    omni_master_address: str | None = None,
    omni_master_port: int | None = None,
    omni_dp_size_local: int = 1,
    omni_heartbeat_timeout: float = 30.0,
    omni_lb_policy: str = "random",
    request_queue: janus.Queue[EngineQueueMessage] | None = None,
) -> StageRuntime:
    """Factory: select StageRuntime or DistStageRuntime."""
    if single_stage_mode:
        if not omni_master_address or not omni_master_port:
            raise ValueError("Distributed mode requires omni_master_address and omni_master_port")
        return DistStageRuntime(
            stage_configs=stage_configs,
            model=model,
            config_path=config_path,
            stage_init_timeout=stage_init_timeout,
            diffusion_batch_size=diffusion_batch_size,
            async_chunk=async_chunk,
            tokenizer=tokenizer,
            single_stage_id_filter=single_stage_id_filter,
            omni_master_address=omni_master_address,
            omni_master_port=omni_master_port,
            omni_dp_size_local=omni_dp_size_local,
            omni_heartbeat_timeout=omni_heartbeat_timeout,
            omni_lb_policy=omni_lb_policy,
            request_queue=request_queue,
        )
    return StageRuntime(
        stage_configs=stage_configs,
        model=model,
        config_path=config_path,
        stage_init_timeout=stage_init_timeout,
        diffusion_batch_size=diffusion_batch_size,
        async_chunk=async_chunk,
        tokenizer=tokenizer,
    )
