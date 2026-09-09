# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage Runtime implementations for single-node and distributed omni stages."""

from __future__ import annotations

import concurrent.futures
import copy
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

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
    extract_legacy_stage_metadata,
    get_stage_connector_spec,
    inject_kv_stage_info,
    inject_omni_kv_connector_config,
    load_omni_transfer_config_for_model,
    prepare_engine_environment,
    release_device_locks,
    stage_runtime_env,
)
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.stage_utils import resolve_stage_physical_devices
from vllm_omni.entrypoints.utils import inject_omni_kv_config
from vllm_omni.outputs.output_metadata import FinalOutputModalityType
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
        async_chunk: bool,
        tokenizer: str | None = None,
        parallel_stage_init: bool = False,
        log_stats: bool = False,
    ) -> None:
        self._stage_configs = stage_configs
        self._model = model
        self._config_path = config_path
        self._stage_init_timeout = stage_init_timeout
        self._async_chunk = async_chunk
        self._tokenizer = tokenizer
        # When True, same-device stages initialize concurrently, coordinated by
        # pre-launch admission control + the engine-core-held SH/EX device locks
        # (see VllmOmniOrchestratorConfig.parallel_stage_init). Default False
        # keeps the legacy per-device LOCK_EX serialization.
        self._parallel_stage_init = parallel_stage_init
        self._log_stats = log_stats
        self._num_stages = len(stage_configs)

        # Populated by initialize()
        self.stage_pools: list[StagePool] = []
        self._stage_init_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._spawn_device_lock = threading.Lock()
        # Serialize all LLM replica spawning + handshake across device groups
        # to prevent ZMQ port-allocation races (get_engine_zmq_addresses) and
        # CUDA-context conflicts when multiple engine core subprocesses
        # initialize simultaneously on different GPUs.  Matches the old
        # AsyncOmniEngine._initialize_llm_replica pattern which used a single
        # ``llm_stage_launch_lock`` for all replicas.
        self._replica_launch_lock = threading.Lock()
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
        replica_devices_map: Mapping[int, Sequence[str | None]],
    ) -> list[LogicalStageInitPlan]:
        """Build startup plans for every logical stage and replica."""
        stage_plans: list[LogicalStageInitPlan] = []

        # RFC #4021 transition boundary: stage planning still relies on legacy
        # StageConfig.runtime and StageConfig.engine_args for replica and engine
        # setup. Keep metadata extraction on the legacy path until the
        # coordinated stage-init cutover.
        for stage_idx, stage_cfg in enumerate(self._stage_configs):
            base_metadata = extract_legacy_stage_metadata(stage_cfg)
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
                # The stable adapter entry point still receives the same
                # legacy stage object as replica planning. Its implementation
                # switches only at the coordinated RFC #4021 cutover.
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

                replica_metadata = extract_legacy_stage_metadata(replica_cfg)
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

        if self._parallel_stage_init:
            # Refuse executor backends the phase locks cannot cover, then prove
            # every physical device's summed budget fits BEFORE any stage spawns.
            # Both fail fast here rather than OOM at allocation time.
            self._reject_unguardable_executors(stage_plans)
            self._run_stage_admission(stage_plans)

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

    def _reject_unguardable_executors(self, stage_plans: Sequence[LogicalStageInitPlan]) -> None:
        """Fail closed for executors the SH/EX phase locks cannot actually guard.

        ``DevicePhaseLock`` is a node-local ``flock`` over files on the engine-core
        driver's host, and the admission ledger is built only from this host's
        devices. A Ray executor may place the workers that load weights, profile,
        allocate KV cache and capture graphs on **other** nodes, which participate
        in neither mechanism. Running there would take the parallel path — the
        parent skips its own device lock when ``parallel_stage_init`` is on —
        while only advertising the protection, so refuse before anything spawns.

        Ray detection uses both the executor's ``uses_ray`` class attribute and
        ``ParallelConfig.use_ray``. Multi-node multiprocessing/external-launcher
        configurations are detected through ``ParallelConfig.nnodes``. Both are
        refused because some workers can live outside the driver's host.

        Remote replicas are skipped: they are launched by the runtime owning their
        node and are already ``ADMISSION_EXEMPT`` in this host's ledger.
        """
        offenders: list[str] = []
        for plan in stage_plans:
            for replica in plan.replicas:
                if replica.launch_mode == "remote":
                    continue
                executor_class = replica.executor_class
                parallel_config = getattr(replica.stage_vllm_config, "parallel_config", None)
                uses_ray = bool(getattr(executor_class, "uses_ray", False)) or bool(
                    getattr(parallel_config, "use_ray", False)
                )
                nnodes = int(getattr(parallel_config, "nnodes", 1) or 1)
                reasons: list[str] = []
                if uses_ray:
                    reasons.append("Ray-backed")
                if nnodes > 1:
                    reasons.append(f"multi-node (nnodes={nnodes})")
                if reasons:
                    executor_name = getattr(executor_class, "__name__", "unknown executor")
                    offenders.append(
                        f"stage{replica.metadata.stage_id}/replica{replica.replica_id} "
                        f"({executor_name}; {', '.join(reasons)})"
                    )
        if offenders:
            raise RuntimeError(
                "parallel_stage_init cannot guard Ray-backed or multi-node executors: the SH/EX "
                "phase locks are node-local file locks and the admission ledger only accounts "
                "for this host's devices, so workers placed on other nodes would initialize unprotected. "
                f"Offending replicas: {', '.join(offenders)}. Disable parallel_stage_init for "
                "this deployment, or use a single-node non-Ray executor backend."
            )

    def _run_stage_admission(self, stage_plans: Sequence[LogicalStageInitPlan]) -> None:
        """Pre-launch per-device admission for parallel stage init (fail-fast)."""
        from vllm_omni.engine.stage_admission import (
            ADMISSION_EXEMPT,
            AdmissionExempt,
            StageAdmissionError,
            check_admission,
        )

        def _resolve(replica: ReplicaInitPlan) -> list[int] | AdmissionExempt | None:
            if replica.launch_mode == "remote":
                # Remote replicas consume a remote node's memory and the ledger
                # is per-LOCAL-device. Remote LLM plans already carry
                # runtime_cfg=None, but remote diffusion replicas keep their
                # cfg — mark both exempt explicitly. Only this exemption may
                # skip the ledger: a LOCAL replica that resolves to None below
                # fails admission (fail-closed in check_admission).
                return ADMISSION_EXEMPT
            resolved = self._resolve_replica_physical_devices(replica.metadata.stage_id, replica.metadata.runtime_cfg)
            if not resolved:
                return None
            ids: list[int] = []
            for tok in str(resolved).split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    ids.append(int(tok))
                except ValueError:
                    # Non-integer visibility (UUID / MIG) can't be reasoned about.
                    return None
            return ids or None

        def _visible_ordinal(physical_id: int) -> int:
            """Translate a physical device id to this process's visible ordinal.

            The ledger is keyed by *physical* id (resolved against the visibility
            baseline captured before any stage narrowed the env), but
            ``get_device_total_memory`` and ``torch.cuda.get_device_properties``
            interpret their argument as an ordinal in the **current** process's
            visible-device namespace. Passing the physical id straight through
            raises an invalid-ordinal error under a restricted
            ``CUDA_VISIBLE_DEVICES`` and, when visibility is reordered, silently
            reads a different GPU — which yields a wrong capacity, and therefore
            an unsafe admission result, on heterogeneous devices.
            """
            baseline = self._init_visible_devices_baseline
            if not baseline:
                # No device-control env captured: ordinals are physical ids.
                return physical_id
            for ordinal, tok in enumerate(baseline.split(",")):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    candidate = int(tok)
                except ValueError:
                    # UUID/MIG entries cannot be converted to physical integer
                    # ids, but they still occupy an ordinal in the visible list.
                    continue
                if candidate == physical_id:
                    return ordinal
            raise StageAdmissionError(
                f"Physical device {physical_id} is not in the visibility baseline "
                f"{baseline!r} captured at init, so its capacity cannot be measured "
                "from this process. Refusing to admit against an unknown device."
            )

        def _total_memory(device_id: int) -> int:
            ordinal = _visible_ordinal(device_id)
            try:
                mem = current_omni_platform.get_device_total_memory(ordinal)
                if mem:
                    return int(mem)
            except (NotImplementedError, AttributeError, TypeError):
                pass
            import torch

            return int(torch.cuda.get_device_properties(ordinal).total_memory)

        check_admission(
            stage_plans,
            resolve_physical_devices=_resolve,
            device_total_memory=_total_memory,
        )

    def _replica_init_group_key(self, replica: ReplicaInitPlan) -> str:
        """Return the scheduling group used during replica initialization.

        Replicas sharing a group initialize sequentially; different groups
        initialize in parallel threads. Local LLM replicas are keyed by their
        **resolved canonical physical device set** so stages on different
        physical GPUs land in different groups (parallel) while stages sharing a
        device stay in one group (serialized, then further guarded by the
        per-device ``LOCK_EX`` file lock). Keying on the raw ``runtime.devices``
        config value instead would fail to parallelize logically-distinct stages
        that map to different physical GPUs.
        """
        if replica.launch_mode == "local" and replica.metadata.stage_type == "diffusion":
            # Local diffusion process spawning must stay on the orchestrator
            # thread. Keep all local diffusion replicas in one sequential group.
            return "inline:diffusion"
        if replica.launch_mode == "remote":
            return f"remote:{replica.metadata.stage_id}:{replica.replica_id}"

        physical_devices = self._resolve_replica_physical_devices(
            replica.metadata.stage_id,
            replica.metadata.runtime_cfg,
        )
        if self._parallel_stage_init:
            # Same-device concurrency is coordinated by the engine-core SH/EX
            # device locks + pre-launch admission, so give every replica its own
            # group to let them all initialize in parallel.
            return f"parallel:{replica.metadata.stage_id}:{replica.replica_id}"

        runtime_cfg = replica.metadata.runtime_cfg or {}
        raw_devices = (
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "devices", None)
        )
        if str(raw_devices) != str(physical_devices):
            logger.debug(
                "[stage_init] Stage-%s init-group key: raw devices=%s -> resolved physical=%s",
                replica.metadata.stage_id,
                raw_devices,
                physical_devices,
            )
        return f"device:{physical_devices}"

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
    ) -> StagePoolClient:
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
            # G3 device locks: in the default (serial) path the parent holds a
            # per-device LOCK_EX across the whole child init. When parallel stage
            # init is enabled the engine-core child takes phased SH/EX locks
            # itself (see stage_phase_lock), so the parent must NOT also hold the
            # lock here — it would self-deadlock while waiting for the child's
            # READY handshake.
            if not self._parallel_stage_init:
                g3_start = time.perf_counter()
                with self._scoped_spawn_device_env(physical_devices):
                    lock_fds = acquire_device_locks(
                        plan.metadata.stage_id,
                        plan.engine_args_dict,
                        stage_init_timeout,
                    )
                logger.debug(
                    "[stage_init] Stage-%s G3 device-lock acquire took %.3fs",
                    plan.metadata.stage_id,
                    time.perf_counter() - g3_start,
                )

            launch_cm = launch_stage_replica(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=self._log_stats,
                stage_id=plan.metadata.stage_id,
                replica_id=plan.replica_id,
                stage_config=plan.stage_cfg,
                omni_master_server=self._get_omni_master_server(),
                omni_coordinator_address=self._get_coordinator_address(),
                stage_visible_devices=physical_devices,
                spawn_device_lock=self._spawn_device_lock,
                omni_parallel_stage_init=self._parallel_stage_init,
            )
            # G2 launch lock serializes engine-core *spawning* across replicas
            # (ZMQ port-allocation races + simultaneous CUDA context init).
            #   * Default path: hold it across the whole launch context manager,
            #     whose __exit__ waits for READY — this serializes the full child
            #     init (spawn + load + profile + KV + capture).
            #   * Parallel path: hold it only around the spawn (__enter__); run
            #     the READY-wait (__exit__) outside the lock so replicas init
            #     concurrently, coordinated by the child SH/EX device locks.
            if self._parallel_stage_init:
                # The per-stage runtime.env overlay mutates os.environ
                # (process-global), so it must be applied under the launch lock
                # and only needs to cover the spawn (__enter__) — children
                # inherit the env at spawn time; the READY-wait needs no env.
                g2_start = time.perf_counter()
                with self._replica_launch_lock:
                    with stage_runtime_env(plan.metadata.stage_id, plan.metadata.runtime_cfg):
                        resources = launch_cm.__enter__()
                g2_spawned = time.perf_counter()
                launch_cm.__exit__(None, None, None)
                logger.debug(
                    "[stage_init] Stage-%s G2 spawn(locked)=%.3fs, READY(unlocked)=%.3fs",
                    plan.metadata.stage_id,
                    g2_spawned - g2_start,
                    time.perf_counter() - g2_spawned,
                )
            else:
                g2_start = time.perf_counter()
                with self._replica_launch_lock, stage_runtime_env(plan.metadata.stage_id, plan.metadata.runtime_cfg):
                    g2_locked = time.perf_counter()
                    with launch_cm as resources:
                        pass
                logger.debug(
                    "[stage_init] Stage-%s G2 launch-lock wait=%.3fs, spawn+READY=%.3fs",
                    plan.metadata.stage_id,
                    g2_locked - g2_start,
                    time.perf_counter() - g2_locked,
                )

            logger.info("[StageRuntime] Stage %s engine startup completed", plan.metadata.stage_id)
            if resources is None:
                raise RuntimeError(f"LLM stage {plan.metadata.stage_id} launcher returned no resources")
            if resources.addresses is None:
                raise RuntimeError(f"LLM stage {plan.metadata.stage_id} launcher returned no addresses")
            stage_client = StageEngineCoreClientBase.make_async_mp_client(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=self._log_stats,
                metadata=plan.metadata,
                client_addresses=self._client_addresses_from_zmq(resources.addresses),
                engine_manager=resources.manager,
                coordinator=resources.coordinator,
            )

            logger.info("[StageRuntime] Stage %s initialized", plan.metadata.stage_id)
            return cast(StagePoolClient, stage_client)
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
    ) -> StagePoolClient:
        """Initialize one local diffusion replica end-to-end."""
        client = None
        resources = None
        try:
            with (
                stage_runtime_env(plan.metadata.stage_id, plan.metadata.runtime_cfg),
                self._stage_device_scope(plan.metadata.stage_id, plan.metadata.runtime_cfg),
            ):
                omni_conn_cfg, omni_from, omni_to = plan.omni_kv_connector
                if omni_conn_cfg:
                    if omni_from is None or omni_to is None:
                        raise RuntimeError("Omni KV connector requires source and destination stages")
                    inject_omni_kv_config(plan.stage_cfg, omni_conn_cfg, omni_from, omni_to)
                inject_kv_stage_info(plan.stage_cfg, plan.metadata.stage_id, self._stage_configs)
                engine_args = getattr(plan.stage_cfg, "engine_args", {})
                inline_diffusion = (
                    engine_args.get("inline_diffusion", False)
                    if hasattr(engine_args, "get")
                    else getattr(engine_args, "inline_diffusion", False)
                )
                custom_pipeline_args = (
                    engine_args.get("custom_pipeline_args")
                    if hasattr(engine_args, "get")
                    else getattr(engine_args, "custom_pipeline_args", None)
                )
                client, resources = launch_diffusion_stage_replica(
                    model=self._model,
                    stage_config=plan.stage_cfg,
                    metadata=plan.metadata,
                    stage_init_timeout=stage_init_timeout,
                    use_inline=plan.num_replicas == 1
                    and bool(self._num_stages == 1 or inline_diffusion or custom_pipeline_args),
                    replica_id=plan.replica_id,
                    omni_master_server=self._get_omni_master_server(),
                    omni_coordinator_address=self._get_coordinator_address(),
                )

            logger.info(
                "[StageRuntime] Stage %s replica %s initialized (diffusion)",
                plan.metadata.stage_id,
                plan.replica_id,
            )
            return cast(StagePoolClient, client)
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
        async_chunk: bool,
        single_stage_id_filter: int | None,
        omni_master_address: str,
        omni_master_port: int,
        tokenizer: str | None = None,
        log_stats: bool = False,
        omni_dp_size_local: int = 1,
        omni_heartbeat_timeout: float = 30.0,
        omni_lb_policy: str = "random",
        request_queue: janus.Queue[EngineQueueMessage] | None = None,
        parallel_stage_init: bool = False,
    ) -> None:
        super().__init__(
            stage_configs=stage_configs,
            model=model,
            config_path=config_path,
            stage_init_timeout=stage_init_timeout,
            async_chunk=async_chunk,
            tokenizer=tokenizer,
            parallel_stage_init=parallel_stage_init,
            log_stats=log_stats,
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

        # Remote diffusion registration still transports the legacy mapping
        # shape. Reconstruct and project that shape until its RFC #4021 cutover.
        metadata = (
            extract_legacy_stage_metadata(OmegaConf.create(registered_stage_cfg))
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
            )
            logger.info(
                "[DistStageRuntime] Remote diffusion replica attached stage=%d replica=%d",
                stage_id,
                replica_id,
            )
            return cast(StagePoolClient, client)

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
                log_stats=self._log_stats,
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
    async_chunk: bool,
    tokenizer: str | None = None,
    parallel_stage_init: bool = False,
    # Distributed-only params:
    single_stage_id_filter: int | None = None,
    omni_master_address: str | None = None,
    omni_master_port: int | None = None,
    omni_dp_size_local: int = 1,
    omni_heartbeat_timeout: float = 30.0,
    omni_lb_policy: str = "random",
    request_queue: janus.Queue[EngineQueueMessage] | None = None,
    log_stats: bool = False,
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
            async_chunk=async_chunk,
            tokenizer=tokenizer,
            parallel_stage_init=parallel_stage_init,
            log_stats=log_stats,
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
        async_chunk=async_chunk,
        tokenizer=tokenizer,
        parallel_stage_init=parallel_stage_init,
        log_stats=log_stats,
    )
