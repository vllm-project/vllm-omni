# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage Runtime implementations for single-node and distributed omni stages."""

from __future__ import annotations

import concurrent.futures
import copy
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import janus
from omegaconf import OmegaConf
from vllm.logger import init_logger

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
from vllm_omni.engine.output_modality import FinalOutputModalityType
from vllm_omni.engine.stage_client import StageClient, StagePoolClient
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase
from vllm_omni.engine.stage_engine_startup import (
    OmniMasterServer,
    StageReplicaResources,
    connect_remote_diffusion_proc,
    connect_remote_engine_cores,
    launch_diffusion_stage_replica,
    launch_stage_replica,
)
from vllm_omni.engine.stage_init_utils import (
    LogicalStageInitPlan,
    ReplicaInitPlan,
    _inject_inferred_kv_tp_topology,
    acquire_device_locks,
    build_engine_args_dict,
    build_llm_stage_output_processor,
    build_vllm_config,
    compute_replica_layout,
    extract_stage_metadata,
    get_stage_connector_spec,
    inject_kv_stage_info,
    inject_omni_kv_connector_config,
    load_omni_transfer_config_for_model,
    prepare_engine_environment,
    release_device_locks,
)
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.stage_utils import resolve_stage_physical_devices
from vllm_omni.entrypoints.utils import inject_omni_kv_config
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class StageRuntimeInfo:
    final_output: bool
    final_output_type: FinalOutputModalityType | None
    stage_type: str
    model_stage: str | None = None


@dataclass
class StageRemoteFactoryContext:
    """Per-stage context for creating head-side clients for remote replicas."""

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
        log_stats: bool = False,
        tracing_enabled: bool = False,
    ) -> None:
        self._stage_configs = stage_configs
        self._model = model
        self._config_path = config_path
        self._stage_init_timeout = stage_init_timeout
        self._diffusion_batch_size = diffusion_batch_size
        self._async_chunk = async_chunk
        self._tokenizer = tokenizer
        self._num_stages = len(stage_configs)
        self._log_stats = log_stats
        self._tracing_enabled = tracing_enabled

        # Populated by initialize()
        self.stage_pools: list[StagePool] = []
        self._stage_init_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._spawn_device_lock = threading.Lock()
        self._init_visible_devices_baseline: str | None = None

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
    def _cleanup_launched_resources(
        *,
        stage_id: int,
        resources: StageReplicaResources | None = None,
    ) -> None:
        """Release launch-only resources when client creation never completed."""
        if resources is None:
            return

        for resource, resource_name in (
            (resources.manager, "manager"),
            (resources.coordinator, "coordinator"),
        ):
            if resource is None:
                continue
            try:
                resource.shutdown()
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
        try:
            self._before_initialize_stage_replicas(stage_plans)
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
                self.stage_pools,
                initialized_clients_by_stage,
            )
            logger.exception(
                "[StageRuntime] Stage initialization failed; shutting down %s initialized client(s)",
                len(cleanup_clients),
            )
            self._shutdown_initialized_clients(cleanup_clients)
            self._cleanup_after_initialize_failure()
            raise exc

    def shutdown(self) -> None:
        for pool in self.stage_pools:
            for client in pool.clients:
                if client is not None and hasattr(client, "shutdown"):
                    try:
                        client.shutdown()
                    except Exception:
                        logger.warning("[StageRuntime] client shutdown failed", exc_info=True)
        if self._stage_init_executor is not None:
            self._stage_init_executor.shutdown(wait=True, cancel_futures=True)
            self._stage_init_executor = None

    def create_membership_controller(self) -> Any | None:
        """Return a distributed membership controller, if this runtime needs one."""
        return None

    def _prepare_stage_plans(self) -> list[LogicalStageInitPlan]:
        """Build logical stage plans and cache prompt expansion metadata."""
        replicas_per_stage, replica_devices_map = compute_replica_layout(self._stage_configs)
        prepare_engine_environment()
        omni_transfer_config = load_omni_transfer_config_for_model(self._model, self._config_path)
        stage_plans = self._build_logical_stage_init_plans(
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
        self.stage_pools = self._assemble_stage_pools(stage_plans, initialized_clients)

    def _before_initialize_stage_replicas(self, stage_plans: Sequence[LogicalStageInitPlan]) -> None:
        """Hook for runtimes that need infrastructure before replica init."""
        return None

    def _cleanup_after_initialize_failure(self) -> None:
        """Hook for runtimes that own extra infrastructure during init."""
        return None

    @contextmanager
    def _scoped_spawn_device_env(self, physical_devices: str | None) -> Iterator[None]:
        """Briefly scope device visibility for spawn-sensitive setup steps."""
        from vllm_omni.engine.stage_engine_startup import scoped_spawn_device_env

        with scoped_spawn_device_env(
            physical_devices,
            self._spawn_device_lock,
        ):
            yield

    def _resolve_replica_physical_devices(self, stage_id: int, runtime_cfg: Any) -> str | None:
        if runtime_cfg is None:
            runtime_cfg = {}
        devices = runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "devices", None)
        return resolve_stage_physical_devices(
            stage_id,
            devices,
            visible_baseline=self._init_visible_devices_baseline,
        )

    @contextmanager
    def _stage_device_scope(self, stage_id: int, runtime_cfg: Any) -> Iterator[None]:
        """Temporarily apply the stage device env while launching a replica."""
        physical_devices = self._resolve_replica_physical_devices(stage_id, runtime_cfg)
        with self._scoped_spawn_device_env(physical_devices):
            yield

    # ---- Internal methods ----

    def _build_logical_stage_init_plans(
        self,
        omni_transfer_config: Any,
        replicas_per_stage: Sequence[int],
        replica_devices_map: Mapping[int, Sequence[str]],
    ) -> list[LogicalStageInitPlan]:
        """Build startup plans for every logical stage and replica."""
        stage_plans: list[LogicalStageInitPlan] = []

        for stage_idx, stage_cfg in enumerate(self._stage_configs):
            base_metadata = extract_stage_metadata(stage_cfg)
            stage_id = int(base_metadata.stage_id)
            if stage_id != stage_idx:
                raise ValueError(
                    "stage_id must match its position in stage_configs: "
                    f"stage_configs[{stage_idx}] has stage_id={stage_id}. "
                    "Stage ids must be contiguous and zero-based because the "
                    "orchestrator indexes stage pools by stage_id."
                )

            stage_connector_spec = get_stage_connector_spec(
                omni_transfer_config=omni_transfer_config,
                stage_id=stage_id,
                async_chunk=self._async_chunk,
            )
            omni_kv_connector = resolve_omni_kv_config_for_stage(omni_transfer_config, stage_id)
            num_replicas = replicas_per_stage[stage_idx]
            launch_mode = self._get_launch_mode(stage_id)

            replicas: list[ReplicaInitPlan] = []
            stage_vllm_config = None
            executor_class = None
            engine_args_dict = None
            if base_metadata.stage_type != "diffusion":
                engine_args_dict = build_engine_args_dict(
                    stage_cfg,
                    self._model,
                    stage_connector_spec=stage_connector_spec,
                    cli_tokenizer=self._tokenizer,
                )
                inject_omni_kv_connector_config(
                    engine_args_dict,
                    omni_kv_connector,
                    stage_id,
                )
                _inject_inferred_kv_tp_topology(
                    engine_args_dict.get("omni_kv_config"),
                    stage_id,
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
                        engine_args_dict=copy.deepcopy(engine_args_dict) if engine_args_dict is not None else None,
                    )
                )

            stage_plans.append(
                LogicalStageInitPlan(
                    stage_idx=stage_idx,
                    stage_id=stage_id,
                    replicas=replicas,
                )
            )

        return stage_plans

    def _get_launch_mode(self, stage_id: int) -> str:
        """Determine launch mode for a stage. Overridden by DistStageRuntime."""
        return "local"

    def _initialize_stage_replicas(
        self,
        stage_plans: Sequence[LogicalStageInitPlan],
        stage_init_timeout: int,
    ) -> dict[int, list[StagePoolClient | None]]:
        """Initialize all stage replicas.

        Stages sharing the same GPU are initialized sequentially to avoid
        memory profiling interference. Stages on different GPUs are
        initialized in parallel.
        """
        initialized_clients_by_stage: dict[int, list[StagePoolClient | None]] = {
            plan.stage_idx: [None] * len(plan.replicas) for plan in stage_plans
        }
        primary_exc: Exception | None = None
        init_state_lock = threading.Lock()
        self._init_visible_devices_baseline = os.environ.get(current_omni_platform.device_control_env_var)

        init_groups: dict[str, list[tuple[int, ReplicaInitPlan]]] = {}
        for plan in stage_plans:
            for replica in plan.replicas:
                init_groups.setdefault(self._replica_init_group_key(replica), []).append((plan.stage_idx, replica))

        def _init_group(group: list[tuple[int, ReplicaInitPlan]]) -> None:
            """Initialize replicas in one scheduling group sequentially."""
            nonlocal primary_exc
            for stage_idx, replica in group:
                with init_state_lock:
                    if primary_exc is not None:
                        return
                try:
                    client = self._initialize_replica(
                        replica,
                        stage_init_timeout,
                    )
                except Exception as exc:
                    with init_state_lock:
                        if primary_exc is None:
                            primary_exc = exc
                    return
                with init_state_lock:
                    initialized_clients_by_stage[stage_idx][replica.replica_id] = client

        inline_keys = [key for key in init_groups if key.startswith("inline:")]
        for key in inline_keys:
            _init_group(init_groups.pop(key))

        if primary_exc is None and init_groups:
            if len(init_groups) == 1:
                _init_group(next(iter(init_groups.values())))
            else:
                future_to_group: dict[concurrent.futures.Future[None], str] = {}
                if self._stage_init_executor is None:
                    self._stage_init_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=len(init_groups),
                        thread_name_prefix="stage-init",
                    )
                for group_key, group in init_groups.items():
                    future_to_group[self._stage_init_executor.submit(_init_group, group)] = group_key

                for future in concurrent.futures.as_completed(future_to_group):
                    try:
                        future.result()
                    except Exception as exc:
                        with init_state_lock:
                            if primary_exc is None:
                                primary_exc = exc

        if primary_exc is not None:
            setattr(primary_exc, "_initialized_clients_by_stage", initialized_clients_by_stage)
            raise primary_exc

        return initialized_clients_by_stage

    def _replica_init_group_key(self, replica: ReplicaInitPlan) -> str:
        """Return the scheduling group used during replica initialization."""
        if replica.launch_mode == "local" and replica.metadata.stage_type == "diffusion":
            # Local diffusion process spawning must stay on the orchestrator
            # thread. Keep all local diffusion replicas in one sequential group.
            return "inline:diffusion"
        if replica.launch_mode == "remote":
            return f"remote:{replica.metadata.stage_id}:{replica.replica_id}"

        runtime_cfg = replica.metadata.runtime_cfg or {}
        devices = runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "devices", None)
        return f"device:{devices}"

    def _initialize_replica(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> StagePoolClient:
        if plan.launch_mode == "remote":
            return self._initialize_remote_replica(plan, stage_init_timeout)
        if plan.metadata.stage_type == "diffusion":
            return self._initialize_local_diffusion_replica(plan, stage_init_timeout)
        return self._initialize_local_llm_replica(plan, stage_init_timeout)

    def _initialize_remote_replica(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> StagePoolClient:
        """Initialize a remote replica. Only distributed runtime implements this."""
        raise NotImplementedError("Remote replicas require DistStageRuntime")

    def _initialize_local_llm_replica(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> StageEngineCoreClientBase:
        """Initialize one local LLM replica using vLLM's launch/attach pattern."""
        resources: StageReplicaResources | None = None
        stage_client = None
        lock_fds: list[int] = []
        try:
            physical_devices = self._resolve_replica_physical_devices(
                plan.metadata.stage_id,
                plan.metadata.runtime_cfg,
            )
            if physical_devices:
                logger.info(
                    "[stage_init] Stage-%s set runtime devices: %s",
                    plan.metadata.stage_id,
                    physical_devices,
                )
            vllm_config = plan.stage_vllm_config
            executor_class = plan.executor_class
            if vllm_config is None:
                raise RuntimeError(f"LLM stage {plan.metadata.stage_id} is missing vllm_config")
            if executor_class is None:
                raise RuntimeError(f"LLM stage {plan.metadata.stage_id} is missing executor_class")
            if plan.engine_args_dict is None:
                raise RuntimeError(f"LLM stage {plan.metadata.stage_id} is missing engine args")
            with self._scoped_spawn_device_env(physical_devices):
                lock_fds = acquire_device_locks(
                    plan.metadata.stage_id,
                    plan.engine_args_dict,
                    stage_init_timeout,
                )
            with launch_stage_replica(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
                stage_id=plan.metadata.stage_id,
                replica_id=plan.replica_id,
                stage_config=plan.stage_cfg,
                omni_master_server=self._get_omni_master_server(),
                omni_coordinator_address=self._get_coordinator_address(),
                stage_visible_devices=physical_devices,
                spawn_device_lock=self._spawn_device_lock,
            ) as resources:
                pass

            logger.info("[StageRuntime] Stage %s engine startup completed", plan.metadata.stage_id)
            if resources is None:
                raise RuntimeError(f"LLM stage {plan.metadata.stage_id} launcher returned no resources")
            if resources.addresses is None:
                raise RuntimeError(f"LLM stage {plan.metadata.stage_id} launcher returned no addresses")
            stage_client = StageEngineCoreClientBase.make_async_mp_client(
                vllm_config=vllm_config,
                executor_class=executor_class,
                metadata=plan.metadata,
                client_addresses=self._client_addresses_from_zmq(resources.addresses),
                engine_manager=resources.manager,
                coordinator=resources.coordinator,
            )

            logger.info("[StageRuntime] Stage %s initialized", plan.metadata.stage_id)
            return stage_client
        except Exception:
            if stage_client is not None:
                try:
                    stage_client.shutdown()
                except Exception as cleanup_error:
                    logger.warning(
                        "[StageRuntime] Failed to cleanup stage %s after init failure: %s",
                        plan.metadata.stage_id,
                        cleanup_error,
                    )
            else:
                self._cleanup_launched_resources(
                    stage_id=plan.metadata.stage_id,
                    resources=resources,
                )
            raise
        finally:
            if lock_fds:
                release_device_locks(lock_fds)

    def _get_coordinator_address(self) -> str | None:
        """Return coordinator router address. Overridden by DistStageRuntime."""
        return None

    def _get_omni_master_server(self) -> OmniMasterServer | None:
        """Return the master server for local distributed launches, if any."""
        return None

    def _initialize_local_diffusion_replica(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> Any:
        """Initialize one local diffusion replica end-to-end."""
        client = None
        resources = None
        try:
            with self._stage_device_scope(plan.metadata.stage_id, plan.metadata.runtime_cfg):
                omni_conn_cfg, omni_from, omni_to = plan.omni_kv_connector
                if omni_conn_cfg:
                    inject_omni_kv_config(plan.stage_cfg, omni_conn_cfg, omni_from, omni_to)
                inject_kv_stage_info(plan.stage_cfg, plan.metadata.stage_id, self._stage_configs)
                client, resources = launch_diffusion_stage_replica(
                    model=self._model,
                    stage_config=plan.stage_cfg,
                    metadata=plan.metadata,
                    stage_init_timeout=stage_init_timeout,
                    batch_size=self._diffusion_batch_size,
                    use_inline=self._num_stages == 1 and plan.num_replicas == 1,
                    replica_id=plan.replica_id,
                    omni_master_server=self._get_omni_master_server(),
                    omni_coordinator_address=self._get_coordinator_address(),
                )

            logger.info(
                "[StageRuntime] Stage %s replica %s initialized (diffusion, batch_size=%d)",
                plan.metadata.stage_id,
                plan.replica_id,
                self._diffusion_batch_size,
            )
            return client
        except Exception:
            if client is not None:
                try:
                    client.shutdown()
                except Exception as cleanup_error:
                    logger.warning(
                        "[StageRuntime] Failed to cleanup stage %s after init failure: %s",
                        plan.metadata.stage_id,
                        cleanup_error,
                    )
            else:
                self._cleanup_launched_resources(
                    stage_id=plan.metadata.stage_id,
                    resources=resources,
                )
            raise
        finally:
            if resources is not None and resources.lock_fds:
                release_device_locks(resources.lock_fds)

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
                if stage_vllm_config is None:
                    raise RuntimeError(f"Stage {plan.stage_id} is missing vllm_config")
                output_processor = build_llm_stage_output_processor(
                    plan,
                    stage_vllm_config,
                    log_stats=self._log_stats,
                    tracing_enabled=self._tracing_enabled,
                )

            stage_pools.append(
                StagePool(
                    plan.stage_idx,
                    clients,
                    output_processor=output_processor,
                    stage_vllm_config=stage_vllm_config,
                )
            )

        return stage_pools


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
        log_stats: bool = False,
        tracing_enabled: bool = False,
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
            log_stats=log_stats,
            tracing_enabled=tracing_enabled,
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

    def create_membership_controller(self) -> Any | None:
        if self._coordinator_runtime is None:
            return None

        from vllm_omni.engine.membership_controller import MembershipController

        return MembershipController(
            stage_pools=self.stage_pools,
            coordinator_pub_address=self._coordinator_runtime.pub_address,
            load_balancer_factory=_build_load_balancer_factory(self._omni_lb_policy),
            remote_replica_factory=self._build_remote_replica,
        )

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
                except (AttributeError, TypeError):
                    if hasattr(runtime_cfg, "__setitem__"):
                        runtime_cfg["num_replicas"] = self._omni_dp_size_local
                        continue
                    logger.warning(
                        "[DistStageRuntime] Failed to apply omni_dp_size_local=%s to stage %s runtime config",
                        self._omni_dp_size_local,
                        stage_id,
                    )

    def _before_initialize_stage_replicas(self, stage_plans: Sequence[LogicalStageInitPlan]) -> None:
        self._start_omni_master_server(stage_plans)
        self._stage_remote_factory_contexts = self._capture_stage_factory_contexts(stage_plans)

    def _cleanup_after_initialize_failure(self) -> None:
        self._cleanup_distributed_infra()

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

    def _get_launch_mode(self, stage_id: int) -> str:
        if self._single_stage_id_filter is not None and stage_id != self._single_stage_id_filter:
            return "remote"
        return "local"

    def _get_coordinator_address(self) -> str | None:
        if self._coordinator_runtime is not None:
            return self._coordinator_runtime.router_address
        return None

    def _get_omni_master_server(self) -> OmniMasterServer | None:
        return self._omni_master_server

    def _initialize_remote_replica(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
    ) -> StagePoolClient:
        """Wait for a configured remote replica and create its head-side client."""
        if self._omni_master_server is None:
            raise RuntimeError("OmniMasterServer is not running; cannot initialize remote replica")
        registered_stage_cfg = self._omni_master_server.get_stage_config(
            plan.metadata.stage_id,
            timeout_s=stage_init_timeout,
            replica_id=plan.replica_id,
        )
        if registered_stage_cfg is None:
            raise ValueError(f"Remote stage {plan.metadata.stage_id} registered without stage config")

        metadata = (
            extract_stage_metadata(OmegaConf.create(registered_stage_cfg))
            if plan.metadata.stage_type == "diffusion"
            else copy.deepcopy(plan.metadata)
        )
        metadata.replica_id = plan.replica_id
        ctx = StageRemoteFactoryContext(
            stage_id=plan.metadata.stage_id,
            stage_type=plan.metadata.stage_type,
            stage_cfg=plan.stage_cfg,
            base_metadata=metadata,
            vllm_config=plan.stage_vllm_config,
            executor_class=plan.executor_class,
            diffusion_batch_size=self._diffusion_batch_size,
        )
        return self._create_remote_replica_client(ctx, plan.replica_id)

    # ---- Distributed infrastructure ----

    def _start_omni_master_server(self, stage_plans: Sequence[LogicalStageInitPlan]) -> None:
        """Start OmniMasterServer and OmniCoordinatorRuntime."""
        if not self._omni_master_address or not self._omni_master_port:
            raise ValueError(
                "AsyncOmniEngine single_stage_mode requires both omni_master_address and omni_master_port to be set."
            )

        from vllm_omni.distributed.omni_coordinator import OmniCoordinatorRuntime

        all_stage_ids: list[int] = []
        stage_replica_counts: dict[int, int] = {}
        head_local_replicas: dict[int, list[int]] = {}
        seen_stage_ids: set[int] = set()
        for plan in stage_plans:
            stage_id = plan.stage_id
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
        contexts: dict[int, StageRemoteFactoryContext] = {}
        for plan in stage_plans:
            if not plan.replicas:
                continue
            template = plan.replicas[0]
            stage_id = int(plan.stage_id)
            contexts[stage_id] = StageRemoteFactoryContext(
                stage_id=stage_id,
                stage_type=template.metadata.stage_type or "llm",
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

    def _build_remote_replica(self, stage_id: int, replica_id: int) -> StagePoolClient:
        ctx = self._stage_remote_factory_contexts.get(stage_id)
        if ctx is None:
            raise ValueError(f"No factory context for stage {stage_id}")
        return self._create_remote_replica_client(ctx, replica_id)

    def _create_remote_replica_client(
        self,
        ctx: StageRemoteFactoryContext,
        replica_id: int,
    ) -> StagePoolClient:
        """Create the head-side client for a remote replica.

        Used by both initial remote slots and dynamic headless registrations.
        """
        if self._omni_master_server is None:
            raise RuntimeError("OmniMasterServer is not running; cannot create remote replica")
        stage_id = ctx.stage_id
        metadata = copy.deepcopy(ctx.base_metadata)
        metadata.replica_id = replica_id

        if ctx.stage_type == "diffusion":
            from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

            resources = None
            try:
                logger.info(
                    "[DistStageRuntime] Remote diffusion handshake started stage=%d replica=%d",
                    stage_id,
                    replica_id,
                )
                with connect_remote_diffusion_proc(
                    omni_master_server=self._omni_master_server,
                    stage_id=stage_id,
                    replica_id=replica_id,
                ) as remote_resources:
                    resources = remote_resources
                if resources is None:
                    raise RuntimeError(f"Remote diffusion stage {stage_id} returned no resources")
                if resources.addresses is None:
                    raise RuntimeError(f"Remote diffusion stage {stage_id} returned no addresses")
            except Exception:
                self._cleanup_launched_resources(
                    stage_id=stage_id,
                    resources=resources,
                )
                raise

            client = StageDiffusionClient.from_addresses(
                metadata,
                request_address=resources.addresses.inputs[0],
                response_address=resources.addresses.outputs[0],
                batch_size=ctx.diffusion_batch_size,
            )
            logger.info(
                "[DistStageRuntime] Remote diffusion replica attached stage=%d replica=%d",
                stage_id,
                replica_id,
            )
            return client

        if ctx.vllm_config is None:
            raise RuntimeError(f"Remote LLM stage {stage_id} is missing vllm_config")
        if ctx.executor_class is None:
            raise RuntimeError(f"Remote LLM stage {stage_id} is missing executor_class")
        vllm_config = copy.deepcopy(ctx.vllm_config)
        vllm_config.parallel_config.data_parallel_size_local = 0
        resources = None
        try:
            logger.info("[DistStageRuntime] Remote LLM handshake started stage=%d replica=%d", stage_id, replica_id)
            with connect_remote_engine_cores(
                vllm_config=vllm_config,
                omni_master_server=self._omni_master_server,
                stage_id=stage_id,
                replica_id=replica_id,
            ) as remote_resources:
                resources = remote_resources
            if resources is None:
                raise RuntimeError(f"Remote LLM stage {stage_id} returned no resources")
            if resources.addresses is None:
                raise RuntimeError(f"Remote LLM stage {stage_id} returned no addresses")
            client_addresses = self._client_addresses_from_zmq(resources.addresses)
            replica_host = self._omni_master_server.get_replica_host(stage_id, replica_id)
            if replica_host:
                client_addresses["replica_host"] = replica_host
            client = StageEngineCoreClientBase.make_async_mp_client(
                vllm_config=vllm_config,
                executor_class=ctx.executor_class,
                metadata=metadata,
                client_addresses=client_addresses,
                engine_manager=resources.manager,
                coordinator=resources.coordinator,
            )
            logger.info("[DistStageRuntime] Remote LLM replica attached stage=%d replica=%d", stage_id, replica_id)
            return client
        except Exception:
            self._cleanup_launched_resources(
                stage_id=stage_id,
                resources=resources,
            )
            raise


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
    log_stats: bool = False,
    tracing_enabled: bool = False,
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
            log_stats=log_stats,
            tracing_enabled=tracing_enabled,
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
        log_stats=log_stats,
        tracing_enabled=tracing_enabled,
    )
