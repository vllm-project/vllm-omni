"""
Async Omni Engine for vLLM-Omni multi-stage runtime.

AsyncOmniEngine in the caller's thread is a thin proxy that communicates
with the Orchestrator (running in a background thread) via janus queues.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import dataclasses
import json
import os
import queue
import threading
import time
import uuid
import weakref
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import janus
import torch
from omegaconf import OmegaConf
from vllm import envs as vllm_envs
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.config.stage_config import strip_parent_engine_args
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.diffusion_engine import supports_audio_output
from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.diffusion.stage_diffusion_proc import (
    complete_diffusion_handshake,
    spawn_diffusion_proc,
)
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    resolve_omni_kv_config_for_stage,
)
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase
from vllm_omni.engine.stage_engine_core_proc import (
    complete_stage_handshake,
    spawn_stage_core,
)
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
from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin
from vllm_omni.entrypoints.utils import (
    inject_omni_kv_config,
    load_and_resolve_stage_configs,
)
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.engine.arg_utils import OmniEngineArgs

logger = init_logger(__name__)

_STARTUP_POLL_INTERVAL_S = 1.0


# ============================================================================
# Parent-EngineArgs field-routing contracts (consumed by
# AsyncOmniEngine._strip_parent_engine_args when ``stage_configs_path`` is set).
# ============================================================================

# Fields that must survive the "equal to default → strip" filter because
# diffusion stages need them even when equal to vllm's default value
# (e.g. colocate worker setup relies on worker_extension_cls being forwarded).
_PARENT_ARGS_KEEP: frozenset[str] = frozenset(
    {
        "worker_extension_cls",
        "allowed_local_media_path",
        "allowed_media_domains",
        # Legacy stage-config YAMLs may intentionally leave parallel or
        # distributed knobs unspecified at the stage level and rely on
        # top-level CLI values to fill them in during the per-stage merge.
        # Keep these fields so stages that omit them can inherit CLI values,
        # while stages with explicit YAML values still win because the legacy
        # stage-config loader prefers stage-local engine args.
        "tensor_parallel_size",
    }
)

# Omni orchestrator-level fields consumed by ``_resolve_stage_configs`` that
# must never leak into per-stage EngineArgs (``stage_configs_path`` would
# trigger the ``create_model_config`` guard).
_PARENT_ARGS_STRIP: frozenset[str] = frozenset({"stage_configs_path"})

# Fields always populated by callers (via ``from_cli_args`` / ``asdict``) so
# their presence as an override is never a surprise — suppress the
# "override ignored" warning for these.
_PARENT_ARGS_NO_WARN: frozenset[str] = frozenset({"model"})


def _inject_global_id(target: Any, request_id: str) -> None:
    """Inject global_request_id into a prompt dict's additional_information."""
    if isinstance(target, dict):
        if "additional_information" not in target:
            target["additional_information"] = {}
        if target["additional_information"] is None:
            target["additional_information"] = {}
        if isinstance(target["additional_information"], dict):
            target["additional_information"]["global_request_id"] = [str(request_id)]


def _upgrade_to_omni_request(
    request: EngineCoreRequest,
    raw_prompt: Any,
) -> EngineCoreRequest:
    """Restore omni-only fields omitted by upstream InputProcessor."""
    prompt_embeds = request.prompt_embeds
    additional_information = None

    if isinstance(raw_prompt, dict):
        if prompt_embeds is None:
            raw_prompt_embeds = raw_prompt.get("prompt_embeds")
            if isinstance(raw_prompt_embeds, torch.Tensor):
                prompt_embeds = raw_prompt_embeds
        additional_information = serialize_additional_information(
            raw_prompt.get("additional_information"),
            log_prefix="AsyncOmniEngine",
        )

    if prompt_embeds is None and additional_information is None:
        return request

    return OmniEngineCoreRequest.from_request(
        request,
        prompt_embeds=prompt_embeds,
        additional_information=additional_information,
    )


def _apply_omni_final_stage_metadata(
    request: EngineCoreRequest,
    final_stage_id: int,
) -> EngineCoreRequest:
    """Tag EngineCoreRequest so OmniARScheduler can skip DiT KV when final_stage_id is 0."""
    merged: dict[str, Any] = {}
    if isinstance(request, OmniEngineCoreRequest) and request.additional_information is not None:
        merged = deserialize_additional_information(request.additional_information)
    merged["omni_final_stage_id"] = final_stage_id
    payload = serialize_additional_information(merged)
    return OmniEngineCoreRequest.from_request(
        request,
        additional_information=payload,
    )


def _weak_shutdown_async_omni_engine(
    orchestrator_thread: threading.Thread | None,
    request_queue: janus.Queue[dict[str, Any]] | None,
    output_queue: janus.Queue[dict[str, Any]] | None,
    rpc_output_queue: janus.Queue[dict[str, Any]] | None,
) -> None:
    """Best-effort orchestrator cleanup for GC finalization."""
    try:
        if request_queue is not None:
            request_queue.sync_q.put_nowait({"type": "shutdown"})
    except Exception:
        pass

    try:
        if orchestrator_thread is not None and orchestrator_thread.is_alive():
            orchestrator_thread.join(timeout=10)
    except Exception:
        pass

    for q in (request_queue, output_queue, rpc_output_queue):
        if q is None:
            continue
        try:
            q.close()
        except Exception:
            pass


class AsyncOmniEngine:
    """Thin proxy that launches an Orchestrator in a background thread.

    All stage clients, input/output processors, and stage-to-stage transfer
    logic live inside the Orchestrator coroutine (running in its own thread
    with a dedicated asyncio event loop). This class communicates with it
    via janus queues (sync side for callers, async side for orchestrator).

    Args:
        model: Model name or path
        init_timeout: Total timeout waiting for orchestrator startup (seconds).
        stage_init_timeout: Timeout for stage initialization (seconds)
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        model: str,
        engine_args: OmniEngineArgs | None = None,
        stage_init_timeout: int = 300,
        init_timeout: int = 600,
        diffusion_batch_size: int = 1,
        single_stage_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.diffusion_batch_size = diffusion_batch_size
        startup_timeout = int(init_timeout)

        logger.info(f"[AsyncOmniEngine] Initializing with model {model}")

        # Merge tracked engine_args fields into kwargs; explicit kwargs take priority.
        if engine_args is not None:
            if not hasattr(engine_args, "_explicit_fields"):
                raise TypeError(
                    "engine_args=OmniEngineArgs(...) is ambiguous under "
                    "sentinel-default precedence. Use "
                    "OmniEngineArgs.create(**explicit) or pass explicit kwargs "
                    "directly."
                )
            ea_dict = engine_args.explicit_kwargs()
            # Remove model since it is passed as a positional arg already.
            ea_dict.pop("model", None)
            kwargs = {**ea_dict, **kwargs}

        self.tokenizer: str | None = kwargs.get("tokenizer")

        # ------------------------------------------------------------------ #
        # Single-stage mode detection                                        #
        # ------------------------------------------------------------------ #
        # Single-stage mode is enabled when the caller explicitly passes      #
        # single_stage_mode=True, or when a stage_id is provided in the args. #
        _stage_id_kwarg = kwargs.get("stage_id")
        if isinstance(_stage_id_kwarg, int) and not single_stage_mode:
            single_stage_mode = True

        self.single_stage_mode: bool = single_stage_mode
        self._single_stage_id_filter: int | None = (
            int(_stage_id_kwarg) if single_stage_mode and isinstance(_stage_id_kwarg, int) else None
        )
        self._omni_master_address: str | None = kwargs.get("omni_master_address")
        self._omni_master_port: int | None = kwargs.get("omni_master_port")
        self._omni_master_server: OmniMasterServer | None = None

        if single_stage_mode:
            logger.info(
                "[AsyncOmniEngine] Single-stage mode enabled (stage_id_filter=%s, master=%s:%s)",
                self._single_stage_id_filter,
                self._omni_master_address,
                self._omni_master_port,
            )

        self.config_path, self.stage_configs = self._resolve_stage_configs(model, kwargs)
        self._validate_single_stage_mode_replica_constraints()

        self.num_stages = len(self.stage_configs)
        stage0_args = getattr(self.stage_configs[0], "engine_args", None) if self.num_stages > 0 else None
        self.async_chunk = bool(getattr(stage0_args, "async_chunk", False))
        self.stage_pools: list[StagePool] = []
        self.stage_clients: list[Any] = []  # logical-stage view for external readers
        self.input_processor: InputProcessor | None = None
        self.supported_tasks: tuple[str, ...] = ("generate",)
        self.default_sampling_params_list: list[Any] = []
        self.stage_metadata: list[dict[str, Any]] = []
        self.request_queue: janus.Queue[dict[str, Any]] | None = None
        self.output_queue: janus.Queue[dict[str, Any]] | None = None
        self.rpc_output_queue: janus.Queue[dict[str, Any]] | None = None
        self._shutdown_called = False
        self._weak_finalizer: weakref.finalize | None = None
        self._rpc_lock = threading.Lock()

        logger.info(f"[AsyncOmniEngine] Launching Orchestrator thread with {self.num_stages} stages")

        # Launch orchestrator background thread
        startup_future: concurrent.futures.Future = concurrent.futures.Future()

        self.orchestrator_thread = threading.Thread(
            target=self._bootstrap_orchestrator,
            args=(
                stage_init_timeout,
                startup_future,
            ),
            daemon=True,
            name="orchestrator",
        )
        self.orchestrator_thread.start()
        self._wait_for_orchestrator_init(startup_future, startup_timeout)

        # Stage runtime fields are assigned directly on self by the bootstrap thread.
        self._weak_finalizer = weakref.finalize(
            self,
            _weak_shutdown_async_omni_engine,
            self.orchestrator_thread,
            self.request_queue,
            self.output_queue,
            self.rpc_output_queue,
        )

        logger.info(f"[AsyncOmniEngine] Orchestrator ready with {self.num_stages} stages")

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
                    "[AsyncOmniEngine] Failed to terminate process for stage %s: %s",
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
                    "[AsyncOmniEngine] Failed to cleanup launched %s for stage %s: %s",
                    resource_name,
                    stage_id,
                    cleanup_error,
                )

    @staticmethod
    def _collect_initialized_clients_for_cleanup(
        stage_pools: Sequence[Any],
        initialized_clients_by_stage: Mapping[int, Sequence[Any | None]],
    ) -> list[Any]:
        """Collect initialized clients exactly once for failure cleanup."""

        collected: list[Any] = []
        seen: set[int] = set()

        def _add_client(client: Any) -> None:
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
    def _shutdown_initialized_clients(clients: Sequence[Any]) -> None:
        """Best-effort shutdown for attached clients after init failure."""

        for client in reversed(list(clients)):
            if client is None:
                continue
            try:
                client.shutdown()
            except Exception as cleanup_error:
                logger.warning(
                    "[AsyncOmniEngine] Failed to shutdown initialized client after init failure: %s",
                    cleanup_error,
                )

    def _validate_single_stage_mode_replica_constraints(self) -> None:
        """Reject unsupported replica fan-out in single-stage mode."""
        if not self.single_stage_mode:
            return

        unsupported: list[tuple[int, int]] = []
        for idx, stage_cfg in enumerate(self.stage_configs):
            runtime_cfg = getattr(stage_cfg, "runtime", {})
            num_replicas = int(
                runtime_cfg.get("num_replicas", 1)
                if hasattr(runtime_cfg, "get")
                else getattr(runtime_cfg, "num_replicas", 1)
            )
            if num_replicas <= 1:
                continue
            if getattr(stage_cfg, "stage_type", "llm") == "diffusion":
                continue
            stage_id = int(getattr(stage_cfg, "stage_id", idx))
            unsupported.append((stage_id, num_replicas))

        if unsupported:
            raise ValueError(
                "single_stage_mode only supports num_replicas > 1 for diffusion stages; "
                f"found non-diffusion stages {unsupported}"
            )

    def _build_logical_stage_init_plans(
        self,
        omni_transfer_config: Any,
        replicas_per_stage: Sequence[int],
        replica_devices_map: Mapping[int, Sequence[str]],
    ) -> tuple[list[LogicalStageInitPlan], Any]:
        """Build startup plans for every logical stage and replica."""

        prompt_expand_func = None
        stage_plans: list[LogicalStageInitPlan] = []

        for stage_idx, stage_cfg in enumerate(self.stage_configs):
            base_metadata = extract_stage_metadata(stage_cfg)
            configured_stage_id = base_metadata.stage_id
            if base_metadata.prompt_expand_func is not None:
                prompt_expand_func = base_metadata.prompt_expand_func

            stage_connector_spec = get_stage_connector_spec(
                omni_transfer_config=omni_transfer_config,
                stage_id=configured_stage_id,
                async_chunk=self.async_chunk,
            )
            omni_kv_connector = resolve_omni_kv_config_for_stage(omni_transfer_config, configured_stage_id)
            num_replicas = replicas_per_stage[stage_idx]
            launch_mode = "local"
            if (
                self.single_stage_mode
                and self._single_stage_id_filter is not None
                and configured_stage_id != self._single_stage_id_filter
            ):
                launch_mode = "remote"

            replicas: list[ReplicaInitPlan] = []
            stage_vllm_config = None
            executor_class = None
            if base_metadata.stage_type != "diffusion":
                engine_args_dict = build_engine_args_dict(
                    stage_cfg,
                    self.model,
                    stage_connector_spec=stage_connector_spec,
                    cli_tokenizer=getattr(self, "tokenizer", None),
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
                if self.stage_configs:
                    _inject_inferred_kv_tp_topology(
                        engine_args_dict.get("omni_kv_config"),
                        configured_stage_id,
                        self.stage_configs,
                    )
                stage_vllm_config, executor_class = build_vllm_config(
                    stage_cfg,
                    self.model,
                    stage_connector_spec=stage_connector_spec,
                    engine_args_dict=engine_args_dict,
                )

            for replica_id in range(num_replicas):
                replica_cfg = copy.deepcopy(stage_cfg) if replica_id > 0 else stage_cfg
                if stage_idx in replica_devices_map:
                    replica_cfg.runtime.devices = replica_devices_map[stage_idx][replica_id]

                replica_metadata = extract_stage_metadata(replica_cfg)
                replica_metadata.replica_id = replica_id
                if self.single_stage_mode:
                    if replica_metadata.stage_type != "diffusion":
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

    def _start_omni_master_server(self, stage_plans: Sequence[LogicalStageInitPlan]) -> None:
        """Start OmniMasterServer for single-stage mode."""

        if not self._omni_master_address or not self._omni_master_port:
            raise ValueError(
                "AsyncOmniEngine single_stage_mode requires both omni_master_address and omni_master_port to be set."
            )

        all_stage_ids: list[int] = []
        stage_replica_counts: dict[int, int] = {}
        seen_stage_ids: set[int] = set()
        for plan in stage_plans:
            stage_id = plan.configured_stage_id
            if stage_id in seen_stage_ids:
                raise ValueError(
                    f"Duplicate stage_id {stage_id!r} detected among configured stages; stage_ids must be unique."
                )
            seen_stage_ids.add(stage_id)
            all_stage_ids.append(stage_id)
            stage_replica_counts[stage_id] = len(plan.replicas)

        self._omni_master_server = OmniMasterServer(
            master_address=self._omni_master_address,
            master_port=self._omni_master_port,
            stage_ids=all_stage_ids,
            stage_replica_counts=stage_replica_counts,
        )
        self._omni_master_server.start()
        logger.info(
            "[AsyncOmniEngine] OmniMasterServer started for stages %s",
            all_stage_ids,
        )

    def _initialize_llm_replica(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
        llm_stage_launch_lock: threading.Lock,
    ) -> Any:
        """Initialize one LLM replica end-to-end."""

        proc = None
        engine_manager = None
        coordinator = None
        stage_client = None
        lock_fds: list[int] = []
        device_control_env = current_omni_platform.device_control_env_var
        stage_cfg = plan.stage_cfg

        try:
            if plan.launch_mode == "remote":
                assert self._omni_master_server is not None
                raw_stage_cfg = self._omni_master_server.get_stage_config(
                    plan.metadata.stage_id,
                    timeout_s=stage_init_timeout,
                    replica_id=plan.replica_id,
                )
                if raw_stage_cfg is None:
                    raise ValueError(f"Remote stage {plan.metadata.stage_id} registered without stage config")
                vllm_config = plan.stage_vllm_config
                executor_class = plan.executor_class
                assert vllm_config is not None
                assert executor_class is not None
                vllm_config.parallel_config.data_parallel_size_local = 0
                launch_cm = connect_remote_engine_cores(
                    vllm_config=vllm_config,
                    omni_master_server=self._omni_master_server,
                    stage_id=plan.metadata.stage_id,
                    replica_id=plan.replica_id,
                )
                logger.info(
                    "[AsyncOmniEngine] Stage %s remote engine handshake started",
                    plan.metadata.stage_id,
                )
                with launch_cm as remote_resources:
                    engine_manager, coordinator, addresses, _tensor_queue = remote_resources

                logger.info(
                    "[AsyncOmniEngine] Stage %s remote engine startup completed",
                    plan.metadata.stage_id,
                )
                client_addresses: dict[str, str] = {
                    "input_address": addresses.inputs[0],
                    "output_address": addresses.outputs[0],
                }
                if addresses.frontend_stats_publish_address is not None:
                    client_addresses["stats_update_address"] = addresses.frontend_stats_publish_address
                stage_client = StageEngineCoreClientBase.make_async_mp_client(
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    metadata=plan.metadata,
                    client_addresses=client_addresses,
                    engine_manager=engine_manager,
                    coordinator=coordinator,
                )
            else:
                handshake_address = None
                with ExitStack() as launch_stack:
                    with llm_stage_launch_lock:
                        previous_visible_devices = os.environ.get(device_control_env)
                        try:
                            setup_stage_devices(plan.metadata.stage_id, plan.metadata.runtime_cfg)
                            vllm_config = plan.stage_vllm_config
                            executor_class = plan.executor_class
                            assert vllm_config is not None
                            assert executor_class is not None
                            engine_args_dict = build_engine_args_dict(
                                stage_cfg,
                                self.model,
                                stage_connector_spec=plan.stage_connector_spec,
                                cli_tokenizer=getattr(self, "tokenizer", None),
                            )
                            lock_fds = acquire_device_locks(
                                plan.metadata.stage_id,
                                engine_args_dict,
                                stage_init_timeout,
                            )
                            if self.single_stage_mode and self._omni_master_server is not None:
                                engine_manager, coordinator, addresses = launch_stack.enter_context(
                                    launch_omni_core_engines(
                                        vllm_config=vllm_config,
                                        executor_class=executor_class,
                                        log_stats=False,
                                        omni_master_server=self._omni_master_server,
                                        stage_id=plan.metadata.stage_id,
                                        stage_config=stage_cfg,
                                        replica_id=plan.replica_id,
                                    )
                                )
                            else:
                                addresses, proc, handshake_address = spawn_stage_core(
                                    vllm_config=vllm_config,
                                    executor_class=executor_class,
                                    log_stats=False,
                                )
                            logger.info(
                                "[AsyncOmniEngine] Stage %s engine launch started",
                                plan.metadata.stage_id,
                            )
                        finally:
                            if previous_visible_devices is None:
                                current_omni_platform.unset_device_control_env_var()
                            else:
                                current_omni_platform.set_device_control_env_var(previous_visible_devices)

                    if self.single_stage_mode and self._omni_master_server is not None:
                        launch_stack.close()
                    else:
                        assert proc is not None
                        assert handshake_address is not None
                        complete_stage_handshake(proc, handshake_address, addresses, vllm_config, stage_init_timeout)
                    logger.info(
                        "[AsyncOmniEngine] Stage %s engine startup completed",
                        plan.metadata.stage_id,
                    )

                    client_addresses: dict[str, str] = {
                        "input_address": addresses.inputs[0],
                        "output_address": addresses.outputs[0],
                    }
                    if addresses.frontend_stats_publish_address is not None:
                        client_addresses["stats_update_address"] = addresses.frontend_stats_publish_address
                    stage_client = StageEngineCoreClientBase.make_async_mp_client(
                        vllm_config=vllm_config,
                        executor_class=executor_class,
                        metadata=plan.metadata,
                        client_addresses=client_addresses,
                        proc=proc,
                        engine_manager=engine_manager,
                        coordinator=coordinator,
                    )

            logger.info("[AsyncOmniEngine] Stage %s initialized", plan.metadata.stage_id)
            return stage_client
        except Exception:
            if stage_client is not None:
                try:
                    stage_client.shutdown()
                except Exception as cleanup_error:
                    logger.warning(
                        "[AsyncOmniEngine] Failed to cleanup stage %s after attach failure: %s",
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
                assert self._omni_master_server is not None
                remote_stage_cfg = OmegaConf.create(
                    self._omni_master_server.get_stage_config(
                        plan.metadata.stage_id,
                        timeout_s=stage_init_timeout,
                        replica_id=plan.replica_id,
                    )
                )
                remote_metadata = extract_stage_metadata(remote_stage_cfg)
                addresses = self._omni_master_server.get_zmq_addresses(
                    plan.metadata.stage_id,
                    replica_id=plan.replica_id,
                )
                logger.info(
                    "[AsyncOmniEngine] Stage %s remote diffusion startup completed",
                    plan.metadata.stage_id,
                )
                client = StageDiffusionClient.from_addresses(
                    remote_metadata,
                    request_address=addresses.inputs[0],
                    response_address=addresses.outputs[0],
                    batch_size=self.diffusion_batch_size,
                )
            else:
                device_control_env = current_omni_platform.device_control_env_var
                with stage_launch_lock:
                    previous_visible_devices = os.environ.get(device_control_env)
                    try:
                        setup_stage_devices(plan.metadata.stage_id, plan.metadata.runtime_cfg)
                        omni_conn_cfg, omni_from, omni_to = plan.omni_kv_connector
                        if omni_conn_cfg:
                            inject_omni_kv_config(plan.stage_cfg, omni_conn_cfg, omni_from, omni_to)
                        inject_kv_stage_info(plan.stage_cfg, plan.metadata.stage_id, self.stage_configs)
                        if self.single_stage_mode:
                            assert self._omni_master_server is not None
                            od_config = build_diffusion_config(self.model, plan.stage_cfg, plan.metadata)
                            lock_fds = acquire_diffusion_device_locks(
                                plan.metadata.stage_id,
                                od_config,
                                stage_init_timeout,
                            )
                            handshake_address, request_address, response_address = register_stage_with_omni_master(
                                omni_master_address=self._omni_master_server.address,
                                omni_master_port=self._omni_master_server.port,
                                omni_stage_id=plan.metadata.stage_id,
                                omni_stage_config=plan.stage_cfg,
                                return_addresses=True,
                                replica_id=plan.replica_id,
                            )
                            logger.info(
                                "[AsyncOmniEngine] Stage %s diffusion registration completed",
                                plan.metadata.stage_id,
                            )
                            proc, _, _, _ = spawn_diffusion_proc(
                                self.model,
                                od_config,
                                handshake_address=handshake_address,
                                request_address=request_address,
                                response_address=response_address,
                            )
                            complete_diffusion_handshake(proc, handshake_address, stage_init_timeout)
                            logger.info(
                                "[AsyncOmniEngine] Stage %s diffusion startup completed",
                                plan.metadata.stage_id,
                            )
                            client = StageDiffusionClient.from_addresses(
                                plan.metadata,
                                request_address=request_address,
                                response_address=response_address,
                                proc=proc,
                                batch_size=self.diffusion_batch_size,
                            )
                        else:
                            client = initialize_diffusion_stage(
                                plan.metadata.stage_id,
                                self.model,
                                plan.stage_cfg,
                                plan.metadata,
                                stage_init_timeout=stage_init_timeout,
                                batch_size=self.diffusion_batch_size,
                                use_inline=self.num_stages == 1 and plan.num_replicas == 1,
                            )
                    finally:
                        if previous_visible_devices is None:
                            current_omni_platform.unset_device_control_env_var()
                        else:
                            current_omni_platform.set_device_control_env_var(previous_visible_devices)

            logger.info(
                "[AsyncOmniEngine] Stage %s replica %s initialized (diffusion, batch_size=%d, devices=%s)",
                plan.metadata.stage_id,
                plan.replica_id,
                self.diffusion_batch_size,
                getattr(getattr(plan.stage_cfg, "runtime", None), "devices", "default"),
            )
            return client
        except Exception:
            if proc is not None:
                terminate_alive_proc(proc)
            raise
        finally:
            if lock_fds:
                release_device_locks(lock_fds)

    def _initialize_replica(
        self,
        plan: ReplicaInitPlan,
        stage_init_timeout: int,
        stage_launch_lock: threading.Lock,
    ) -> Any:
        """Initialize one replica, regardless of backend type."""

        if plan.metadata.stage_type == "diffusion":
            return self._initialize_diffusion_replica(plan, stage_init_timeout, stage_launch_lock)
        return self._initialize_llm_replica(plan, stage_init_timeout, stage_launch_lock)

    def _initialize_stage_replicas(
        self,
        stage_plans: Sequence[LogicalStageInitPlan],
        stage_init_timeout: int,
    ) -> dict[int, list[Any | None]]:
        """Initialize all stage replicas in parallel."""

        stage_launch_lock = threading.Lock()
        initialized_clients_by_stage: dict[int, list[Any | None]] = {
            plan.stage_idx: [None] * len(plan.replicas) for plan in stage_plans
        }
        total_replicas = sum(len(plan.replicas) for plan in stage_plans)
        future_to_replica: dict[concurrent.futures.Future[Any], tuple[int, int]] = {}
        primary_exc: Exception | None = None

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, total_replicas),
            thread_name_prefix="stage-init",
        ) as init_executor:
            for plan in stage_plans:
                for replica in plan.replicas:
                    future = init_executor.submit(
                        self._initialize_replica,
                        replica,
                        stage_init_timeout,
                        stage_launch_lock,
                    )
                    future_to_replica[future] = (plan.stage_idx, replica.replica_id)

            for future in concurrent.futures.as_completed(future_to_replica):
                stage_idx, replica_id = future_to_replica[future]
                try:
                    initialized_clients_by_stage[stage_idx][replica_id] = future.result()
                except concurrent.futures.CancelledError:
                    continue
                except Exception as exc:
                    if primary_exc is None:
                        primary_exc = exc
                        for other_future in future_to_replica:
                            if other_future is future:
                                continue
                            other_future.cancel()

        if primary_exc is not None:
            setattr(primary_exc, "_initialized_clients_by_stage", initialized_clients_by_stage)
            raise primary_exc

        return initialized_clients_by_stage

    def _assemble_stage_pools(
        self,
        stage_plans: Sequence[LogicalStageInitPlan],
        initialized_clients_by_stage: Mapping[int, Sequence[Any | None]],
    ) -> list[StagePool]:
        """Assemble logical stage pools and update top-level stage metadata."""

        stage_pools: list[StagePool] = []
        default_sampling_params_list: list[Any] = []
        stage_metadata_list: list[dict[str, Any]] = []

        for plan in stage_plans:
            replica_clients = initialized_clients_by_stage[plan.stage_idx]
            first_client = replica_clients[0] if replica_clients else None
            if first_client is None:
                raise RuntimeError(f"Stage {plan.stage_idx} initialization completed with a missing client")

            clients = [client for client in replica_clients if client is not None]
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
            default_sampling_params_list.append(first_client.default_sampling_params)
            stage_metadata_list.append(
                {
                    "final_output": first_client.final_output,
                    "final_output_type": first_client.final_output_type,
                    "stage_type": first_client.stage_type,
                }
            )

        self.default_sampling_params_list = list(default_sampling_params_list)
        self.stage_metadata = list(stage_metadata_list)
        return stage_pools

    def _initialize_stages(self, stage_init_timeout: int) -> None:
        """Initialize stage clients/processors in orchestrator thread and assign to self.

        Phases:
          1. Compute replica layout (counts + device splits).
          2. Build per-stage/per-replica startup plans.
          3. Initialize all replicas in parallel via backend-specific launchers.
          4. Build logical StagePools and finalize runtime metadata.

        TODO(stage-pool): move per-stage launch + attach logic into a
        StagePool.build_from_config() classmethod so this method only
        iterates stage_configs, collects pools, and finalizes metadata.
        """
        num_stages = len(self.stage_configs)
        self.num_stages = num_stages
        self._validate_single_stage_mode_replica_constraints()

        replicas_per_stage, replica_devices_map = compute_replica_layout(self.stage_configs)

        prepare_engine_environment()
        omni_transfer_config = load_omni_transfer_config_for_model(self.model, self.config_path)
        stage_plans, prompt_expand_func = self._build_logical_stage_init_plans(
            omni_transfer_config,
            replicas_per_stage,
            replica_devices_map,
        )
        if self.single_stage_mode:
            self._start_omni_master_server(stage_plans)

        stage_pools: list[StagePool] = []
        input_processor: InputProcessor | None = None
        initialized_clients_by_stage: dict[int, list[Any | None]] = {
            plan.stage_idx: [None] * len(plan.replicas) for plan in stage_plans
        }

        try:
            initialized_clients_by_stage = self._initialize_stage_replicas(stage_plans, stage_init_timeout)
            if stage_plans and stage_plans[0].replicas[0].metadata.stage_type != "diffusion":
                stage0_vllm_config = stage_plans[0].replicas[0].stage_vllm_config
                assert stage0_vllm_config is not None
                input_processor = build_stage0_input_processor(stage0_vllm_config)
            stage_pools = self._assemble_stage_pools(stage_plans, initialized_clients_by_stage)
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
                "[AsyncOmniEngine] Stage initialization failed; shutting down %s initialized client(s)",
                len(cleanup_clients),
            )
            self._shutdown_initialized_clients(cleanup_clients)
            if self._omni_master_server is not None:
                try:
                    self._omni_master_server.stop()
                except Exception:
                    logger.exception("[AsyncOmniEngine] Failed to stop OmniMasterServer during stage-init cleanup")
            raise

        self.stage_pools = stage_pools
        self.input_processor = input_processor
        self.prompt_expand_func = prompt_expand_func

        # Derive logical-stage views for external readers (entrypoints/async_omni.py).
        self.stage_clients = [pool.stage_client for pool in self.stage_pools]
        self.stage_vllm_configs = [pool.stage_vllm_config for pool in self.stage_pools]
        self.output_processors = [pool.output_processor for pool in self.stage_pools]

        # TODO(Peiqi): Hack here
        supported_tasks: set[str] = set()
        if any(getattr(pool.stage_client, "is_comprehension", False) for pool in self.stage_pools):
            supported_tasks.add("generate")
        if any(m.get("final_output_type") == "audio" for m in self.stage_metadata):
            supported_tasks.add("speech")
        self.supported_tasks = tuple(supported_tasks) if supported_tasks else ("generate",)

    def _initialize_janus_queues(self) -> None:
        """Initialize janus queues inside orchestrator thread loop context."""
        self.request_queue = janus.Queue()
        self.output_queue = janus.Queue()
        self.rpc_output_queue = janus.Queue()
        logger.debug("[AsyncOmniEngine] janus queues initialized in orchestrator thread loop")

    def _bootstrap_orchestrator(
        self,
        stage_init_timeout: int,
        startup_future: concurrent.futures.Future,
    ) -> None:
        """Create loop, initialize stages, then run Orchestrator."""

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run_orchestrator() -> None:
            self._initialize_janus_queues()

            self._initialize_stages(stage_init_timeout)
            pd_config = self._detect_pd_config()
            orchestrator = Orchestrator(
                request_async_queue=self.request_queue.async_q,
                output_async_queue=self.output_queue.async_q,
                rpc_async_queue=self.rpc_output_queue.async_q,
                stage_pools=self.stage_pools,
                async_chunk=self.async_chunk,
                pd_config=pd_config,
            )
            if not startup_future.done():
                startup_future.set_result(asyncio.get_running_loop())
            await orchestrator.run()

        try:
            loop.run_until_complete(_run_orchestrator())
        except Exception as e:
            if not startup_future.done():
                wrapped = RuntimeError(f"Orchestrator initialization failed: {e}")
                wrapped.__cause__ = e
                startup_future.set_exception(wrapped)
            logger.exception("[AsyncOmniEngine] Orchestrator thread crashed")
            error_text = str(e) or "Orchestrator thread crashed"
            try:
                error_msg = {"type": "error", "error": error_text, "fatal": True}
                if self.output_queue is not None:
                    self.output_queue.sync_q.put_nowait(error_msg)
                if self.rpc_output_queue is not None:
                    self.rpc_output_queue.sync_q.put_nowait(error_msg)
            except Exception:
                pass
            raise
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                if hasattr(loop, "shutdown_default_executor"):
                    loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                logger.exception("[AsyncOmniEngine] Failed during orchestrator loop cleanup")
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def _wait_for_orchestrator_init(self, startup_future: concurrent.futures.Future, startup_timeout: int) -> None:
        """
        Wait for orchestrator startup future to return ready. Raises exception on any failures to the init process.
        """
        deadline = time.monotonic() + startup_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._try_shutdown("[AsyncOmniEngine] Failed to cleanup after orchestrator startup timeout")
                raise TimeoutError(f"Orchestrator did not become ready within {startup_timeout}s")
            try:
                startup_future.result(
                    timeout=min(remaining, _STARTUP_POLL_INTERVAL_S),
                )
                break
            except concurrent.futures.TimeoutError:
                if not self.orchestrator_thread.is_alive():
                    self._try_shutdown("[AsyncOmniEngine] Failed to cleanup after orchestrator startup failure")
                    if startup_future.done():
                        startup_future.result()  # re-raises the real exception
                    raise RuntimeError("Orchestrator thread died during startup")
            except Exception:
                self._try_shutdown("[AsyncOmniEngine] Failed to cleanup after orchestrator startup failure")
                raise

    # ---- request helpers ----

    def _build_add_request_message(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
        lora_request: Any = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        *,
        resumable: bool = False,
        message_type: str = "add_request",
    ) -> dict[str, Any]:
        """Build an add_request message after stage-0 preprocessing."""
        effective_sampling_params_list = (
            list(sampling_params_list) if sampling_params_list is not None else list(self.default_sampling_params_list)
        )
        if not effective_sampling_params_list:
            raise ValueError(
                f"Missing sampling params for stage 0. Got {len(effective_sampling_params_list)} stage params."
            )
        params = effective_sampling_params_list[0]

        # Keep the original prompt for downstream stages (they need the raw
        # dict, e.g. for multi_modal_data).
        original_prompt = prompt

        stage_type = self.stage_metadata[0].get("stage_type")
        output_prompt_text: Any = None
        _preprocess_ms = 0.0
        if stage_type != "diffusion" and not isinstance(prompt, EngineCoreRequest):
            # Inject global_request_id into the raw prompt.
            if isinstance(prompt, dict):
                _inject_global_id(prompt, request_id)
            elif isinstance(prompt, list):
                for item in prompt:
                    _inject_global_id(item, request_id)

            # Full input processing (tokenization, multimodal, etc.)
            _t_preprocess = time.perf_counter()
            request = self.input_processor.process_inputs(
                request_id=request_id,
                prompt=prompt,
                params=params,
                supported_tasks=self.supported_tasks,
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                resumable=resumable,
            )
            _preprocess_ms = (time.perf_counter() - _t_preprocess) * 1000.0
            # TODO (Peiqi): add this for Qwen3-TTS only. Other models don't have
            # additional_information field in the prompt.
            request = _upgrade_to_omni_request(request, prompt)

            if reasoning_ended is not None:
                request.reasoning_ended = reasoning_ended

            # Restore external_req_id to the original user-facing request_id.
            # InputProcessor.process_inputs() renames request_id to an internal
            # UUID (saving the original in external_req_id), but then overwrites
            # external_req_id with the new internal ID. We need external_req_id
            # to match the key used in Orchestrator.request_states so that
            # output routing (output.request_id lookup) can find the req_state.
            request.external_req_id = request_id
            request = _apply_omni_final_stage_metadata(request, final_stage_id)

            # Registration with stage 0's output processor is deferred to the
            # orchestrator thread (see Orchestrator._handle_add_request), which
            # now routes admission through StagePool.submit_initial().
            output_prompt_text = prompt_text
            if output_prompt_text is None and isinstance(original_prompt, dict):
                output_prompt_text = original_prompt.get("prompt")
            prompt = request

        return {
            "type": message_type,
            "request_id": request_id,
            "prompt": prompt,
            "original_prompt": original_prompt,
            "output_prompt_text": output_prompt_text,
            "sampling_params_list": effective_sampling_params_list,
            "final_stage_id": final_stage_id,
            "preprocess_ms": _preprocess_ms,
            "enqueue_ts": time.perf_counter(),
        }

    def _enqueue_cfg_companions(
        self,
        parent_id: str,
        original_prompt: Any,
        stage0_params: Any,
        sampling_params_list: list[Any],
    ) -> None:
        """Expand prompt into CFG companions, process through InputProcessor, and enqueue."""
        try:
            expanded = self.prompt_expand_func(original_prompt, stage0_params)
        except Exception:
            logger.exception("[AsyncOmniEngine] prompt_expand_func failed for req %s", parent_id)
            return

        if not expanded:
            return

        for ep in expanded:
            cid = f"{parent_id}{ep.request_id_suffix}"
            companion_prompt = ep.prompt

            companion_params, companion_spl = ep.apply_overrides(stage0_params, sampling_params_list)

            if isinstance(companion_prompt, dict):
                _inject_global_id(companion_prompt, cid)

            request = self.input_processor.process_inputs(
                request_id=cid,
                prompt=companion_prompt,
                params=companion_params,
                supported_tasks=self.supported_tasks,
            )
            request.external_req_id = cid

            # Registration of this companion on stage-0's output processor is
            # deferred to Orchestrator._handle_add_companion, which routes
            # admission through StagePool.submit_initial(..., affinity_request_id=...).
            self.request_queue.sync_q.put_nowait(
                {
                    "type": "add_companion_request",
                    "companion_id": cid,
                    "parent_id": parent_id,
                    "role": ep.role,
                    "prompt": request,
                    "companion_prompt_text": companion_prompt,
                    "sampling_params_list": companion_spl,
                }
            )

        logger.info(
            "[AsyncOmniEngine] CFG expansion for req %s: %d companions",
            parent_id,
            len(expanded),
        )

    @staticmethod
    def _get_default_cache_config(cache_backend: str | None) -> dict[str, Any] | None:
        if cache_backend == "cache_dit":
            return {
                "Fn_compute_blocks": 1,
                "Bn_compute_blocks": 0,
                "max_warmup_steps": 4,
                "residual_diff_threshold": 0.24,
                "max_continuous_cached_steps": 3,
                "enable_taylorseer": False,
                "taylorseer_order": 1,
                "scm_steps_mask_policy": None,
                "scm_steps_policy": "dynamic",
            }
        if cache_backend == "tea_cache":
            return {
                "rel_l1_thresh": 0.2,
            }
        return None

    @staticmethod
    def _normalize_cache_config(cache_backend: str | None, cache_config: Any | None) -> Any | None:
        if isinstance(cache_config, str):
            try:
                cache_config = json.loads(cache_config)
            except json.JSONDecodeError:
                logger.warning("Invalid cache_config JSON, using defaults.")
                cache_config = None
        if cache_config is None and cache_backend not in (None, "", "none"):
            cache_config = AsyncOmniEngine._get_default_cache_config(cache_backend)
        return cache_config

    def _detect_pd_config(self) -> dict[str, Any] | None:
        """Detect PD (Prefill-Decode) disaggregation config from stage_configs.
        Returns a dict with 'pd_pair' and 'bootstrap_addr', or None.
        """
        pd_pair = PDDisaggregationMixin.detect_pd_separation_from_stage_configs(self.stage_configs)
        if pd_pair is None:
            return None
        prefill_idx, decode_idx = pd_pair

        # Extract bootstrap address from prefill stage engine_args
        bootstrap_addr: str | None = None
        try:
            prefill_cfg = self.stage_configs[prefill_idx]
            ea = getattr(prefill_cfg, "engine_args", None)
            kv_cfg = getattr(ea, "kv_transfer_config", None) if ea is not None else None
            if kv_cfg is not None:
                port = vllm_envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
                kv_ip = getattr(kv_cfg, "kv_ip", None) or "127.0.0.1"
                bootstrap_addr = f"http://{kv_ip}:{port}"
        except Exception as exc:
            logger.warning("[AsyncOmniEngine] Could not extract PD bootstrap address: %s", exc)

        logger.info(
            "[AsyncOmniEngine] PD disaggregation detected: prefill=stage-%d, decode=stage-%d, bootstrap=%s",
            prefill_idx,
            decode_idx,
            bootstrap_addr,
        )
        prefill_engine_id: str | None = None
        try:
            prefill_client = self.stage_clients[prefill_idx]
            kv_cfg = getattr(getattr(prefill_client, "vllm_config", None), "kv_transfer_config", None)
            prefill_engine_id = getattr(kv_cfg, "engine_id", None)
        except Exception as exc:
            logger.warning("[AsyncOmniEngine] Could not extract prefill engine_id: %s", exc)

        return {
            "pd_pair": (prefill_idx, decode_idx),
            "bootstrap_addr": bootstrap_addr,
            "prefill_engine_id": prefill_engine_id,
        }

    @staticmethod
    def _create_default_diffusion_stage_cfg(kwargs: dict[str, Any]) -> list:
        """Create a default single-stage diffusion config from kwargs."""
        # We temporally create a default config for diffusion stage.
        # In the future, we should merge the default config with the user-provided config.
        normalized_kwargs = dict(kwargs)
        default_sampling_params = normalized_kwargs.get("default_sampling_params")
        if isinstance(default_sampling_params, str):
            try:
                default_sampling_params = json.loads(default_sampling_params)
            except json.JSONDecodeError:
                logger.warning("Invalid default_sampling_params JSON, ignoring stage defaults.")
                default_sampling_params = None
        if not isinstance(default_sampling_params, dict):
            default_sampling_params = None
        stage_default_sampling_params = default_sampling_params.get("0", {}) if default_sampling_params else {}
        if normalized_kwargs.get("dtype") is None:
            normalized_kwargs["dtype"] = "auto"

        # TODO: hack, convert dtype to string to avoid non-premitive omegaconf create error.
        if "dtype" in normalized_kwargs and not isinstance(normalized_kwargs["dtype"], str):
            if not isinstance(normalized_kwargs["dtype"], torch.dtype):
                raise TypeError(
                    f"Provided dtype must be a string or torch.dtype, got {type(normalized_kwargs['dtype']).__name__}"
                )
            normalized_kwargs["dtype"] = str(normalized_kwargs["dtype"]).removeprefix("torch.")

        cache_backend = normalized_kwargs.get("cache_backend", "none")
        cache_config = AsyncOmniEngine._normalize_cache_config(
            cache_backend,
            normalized_kwargs.get("cache_config", None),
        )

        parallel_config = normalized_kwargs.get("parallel_config")
        if isinstance(parallel_config, dict):
            parallel_config = DiffusionParallelConfig.from_dict(parallel_config)
        if parallel_config is None:
            ulysses_degree = normalized_kwargs.get("ulysses_degree") or 1
            ring_degree = normalized_kwargs.get("ring_degree") or 1
            ulysses_mode = normalized_kwargs.get("ulysses_mode") or "strict"
            sequence_parallel_size = normalized_kwargs.get("sequence_parallel_size")
            pipeline_parallel_size = normalized_kwargs.get("pipeline_parallel_size") or 1
            data_parallel_size = normalized_kwargs.get("data_parallel_size") or 1
            tensor_parallel_size = normalized_kwargs.get("tensor_parallel_size") or 1
            cfg_parallel_size = normalized_kwargs.get("cfg_parallel_size") or 1
            vae_patch_parallel_size = normalized_kwargs.get("vae_patch_parallel_size") or 1
            enable_expert_parallel = normalized_kwargs.get("enable_expert_parallel") or False
            use_hsdp = normalized_kwargs.get("use_hsdp", False)
            hsdp_shard_size = normalized_kwargs.get("hsdp_shard_size", -1)
            hsdp_replicate_size = normalized_kwargs.get("hsdp_replicate_size", 1)
            if sequence_parallel_size is None:
                sequence_parallel_size = ulysses_degree * ring_degree

            parallel_config = DiffusionParallelConfig(
                pipeline_parallel_size=pipeline_parallel_size,
                data_parallel_size=data_parallel_size,
                tensor_parallel_size=tensor_parallel_size,
                enable_expert_parallel=enable_expert_parallel,
                sequence_parallel_size=sequence_parallel_size,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                ulysses_mode=ulysses_mode,
                cfg_parallel_size=cfg_parallel_size,
                vae_patch_parallel_size=vae_patch_parallel_size,
                use_hsdp=use_hsdp,
                hsdp_shard_size=hsdp_shard_size,
                hsdp_replicate_size=hsdp_replicate_size,
            )

        num_devices = max(1, int(parallel_config.world_size))
        devices = ",".join(str(i) for i in range(num_devices))
        model_class_name = kwargs.get("model_class_name", None)
        final_output_type = "audio" if model_class_name and supports_audio_output(model_class_name) else "image"

        stage_engine_args = {
            "max_num_seqs": kwargs.get("max_num_seqs") or 1,
            "parallel_config": parallel_config,
            "model_class_name": kwargs.get("model_class_name", None),
            "step_execution": kwargs.get("step_execution", False),
            "vae_use_slicing": kwargs.get("vae_use_slicing", False),
            "vae_use_tiling": kwargs.get("vae_use_tiling", False),
            "cache_backend": cache_backend,
            "cache_config": cache_config,
            "enable_cache_dit_summary": kwargs.get("enable_cache_dit_summary", False),
            "enable_cpu_offload": kwargs.get("enable_cpu_offload", False),
            "enable_layerwise_offload": kwargs.get("enable_layerwise_offload", False),
            "enforce_eager": False if kwargs.get("enforce_eager") is None else kwargs.get("enforce_eager"),
            "boundary_ratio": kwargs.get("boundary_ratio", None),
            "flow_shift": kwargs.get("flow_shift", None),
            "diffusion_load_format": kwargs.get("diffusion_load_format", "default"),
            "custom_pipeline_args": kwargs.get("custom_pipeline_args", None),
            "worker_extension_cls": kwargs.get("worker_extension_cls", None),
            "trust_remote_code": (False if kwargs.get("trust_remote_code") is None else kwargs["trust_remote_code"]),
            "distributed_executor_backend": (
                "mp" if kwargs.get("distributed_executor_backend") is None else kwargs["distributed_executor_backend"]
            ),
            "enable_sleep_mode": kwargs.get("enable_sleep_mode", False),
            "enable_multithread_weight_load": kwargs.get("enable_multithread_weight_load", True),
            "num_weight_load_threads": kwargs.get("num_weight_load_threads", 4),
            "quantization": kwargs.get("quantization", None),
            "enable_diffusion_pipeline_profiler": kwargs.get("enable_diffusion_pipeline_profiler", False),
            "enable_ar_profiler": kwargs.get("enable_ar_profiler", False),
            **(
                {
                    "profiler_config": asdict(kwargs["profiler_config"])
                    if hasattr(kwargs["profiler_config"], "__dataclass_fields__")
                    else kwargs["profiler_config"]
                }
                if kwargs.get("profiler_config") is not None
                else {}
            ),
        }
        # Only set dtype if it was already explicitly passed and normalized
        if "dtype" in normalized_kwargs:
            stage_engine_args["dtype"] = normalized_kwargs["dtype"]

        # New split fields for diffusers adapter kwargs.
        if kwargs.get("diffusers_load_kwargs") is not None:
            stage_engine_args["diffusers_load_kwargs"] = kwargs["diffusers_load_kwargs"]
        if kwargs.get("diffusers_call_kwargs") is not None:
            stage_engine_args["diffusers_call_kwargs"] = kwargs["diffusers_call_kwargs"]

        default_stage_cfg = [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {
                    "process": True,
                    "devices": devices,
                },
                "engine_args": stage_engine_args,
                "default_sampling_params": stage_default_sampling_params,
                "final_output": True,
                "final_output_type": final_output_type,
            }
        ]
        default_stage_cfg[0]["engine_args"]["model_stage"] = "diffusion"
        return default_stage_cfg

    @staticmethod
    def _strip_single_engine_args(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Remove parent ``EngineArgs`` fields from *kwargs*.

        When ``stage_configs_path`` is set, per-stage engine args are defined
        in the YAML.  Top-level single-engine fields (``compilation_config``,
        ``tensor_parallel_size``, …) must not leak into per-stage configs via
        the ``base_engine_args`` merge in ``load_stage_configs_from_yaml`` —
        they can cause type errors (e.g. ``compilation_config`` as a JSON
        string rejected by ``VllmConfig``) or silently override YAML values.

        Logs a warning for any parent field whose value differs from the
        dataclass default, so users know their explicit overrides are ignored.
        See the module-level ``_PARENT_ARGS_*`` constants for the routing
        contracts this method enforces.
        """
        parent_fields: dict[str, dataclasses.Field] = {f.name: f for f in dataclasses.fields(EngineArgs)}
        result, overridden = strip_parent_engine_args(
            kwargs,
            parent_fields=parent_fields,
            keep_keys=_PARENT_ARGS_KEEP,
            strip_keys=_PARENT_ARGS_STRIP,
            no_warn_keys=_PARENT_ARGS_NO_WARN,
        )

        if overridden:
            logger.warning(
                "stage_configs_path is set — the following top-level engine "
                "args are ignored (per-stage YAML takes precedence): %s",
                ", ".join(sorted(overridden)),
            )

        return result

    def _resolve_stage_configs(self, model: str, kwargs: dict[str, Any]) -> tuple[str, list[Any]]:
        """Resolve stage configs and inject defaults shared by orchestrator/headless."""

        stage_configs_path = kwargs.get("stage_configs_path", None)
        deploy_config_path = kwargs.pop("deploy_config", None)
        stage_overrides_json = kwargs.pop("stage_overrides", None)
        kwargs.pop("_cli_explicit_keys", None)
        explicit_stage_configs = kwargs.pop("stage_configs", None)
        if explicit_stage_configs is not None:
            logger.warning(
                "`stage_configs` is not part of the public API. "
                "Ignoring it and resolving stages from stage_configs_path/model factory."
            )

        if stage_configs_path is not None:
            base_kwargs = self._strip_single_engine_args(kwargs)
        else:
            base_kwargs = kwargs

        # Parse --stage-overrides JSON string if provided
        stage_overrides = None
        if stage_overrides_json:
            if isinstance(stage_overrides_json, str):
                try:
                    stage_overrides = json.loads(stage_overrides_json)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"--stage-overrides is not valid JSON: {exc}. Got: {stage_overrides_json!r}"
                    ) from exc
            else:
                stage_overrides = stage_overrides_json

        config_path, stage_configs = load_and_resolve_stage_configs(
            model,
            stage_configs_path,
            base_kwargs,
            default_stage_cfg_factory=lambda: self._create_default_diffusion_stage_cfg(kwargs),
            deploy_config_path=deploy_config_path,
            stage_overrides=stage_overrides,
        )

        # Inject diffusion LoRA-related knobs from kwargs if not present in the stage config.
        for cfg in stage_configs:
            try:
                if not hasattr(cfg, "engine_args") or cfg.engine_args is None:
                    cfg.engine_args = OmegaConf.create({})
                global_sleep_mode = kwargs.get("enable_sleep_mode")
                if global_sleep_mode is not None:
                    if not hasattr(cfg.engine_args, "enable_sleep_mode") or cfg.engine_args.enable_sleep_mode is None:
                        cfg.engine_args.enable_sleep_mode = global_sleep_mode
                if getattr(cfg, "stage_type", None) != "diffusion":
                    continue
                if not hasattr(cfg, "engine_args") or cfg.engine_args is None:
                    cfg.engine_args = OmegaConf.create({})
                if kwargs.get("lora_path") is not None:
                    if not hasattr(cfg.engine_args, "lora_path") or cfg.engine_args.lora_path is None:
                        cfg.engine_args.lora_path = kwargs["lora_path"]
                lora_scale = kwargs.get("lora_scale")
                if lora_scale is None:
                    # Backwards compatibility for older callers.
                    lora_scale = kwargs.get("static_lora_scale")
                if lora_scale is not None:
                    if not hasattr(cfg.engine_args, "lora_scale") or cfg.engine_args.lora_scale is None:
                        cfg.engine_args.lora_scale = lora_scale
                quantization_config = kwargs.get("quantization_config")
                if quantization_config is not None:
                    if (
                        not hasattr(cfg.engine_args, "quantization_config")
                        or cfg.engine_args.quantization_config is None
                    ):
                        cfg.engine_args.quantization_config = quantization_config
                # Inject profiler flags for diffusion stages
                for profiler_key in (
                    "enable_diffusion_pipeline_profiler",
                    "enable_ar_profiler",
                ):
                    val = kwargs.get(profiler_key)
                    if val:
                        if not hasattr(cfg.engine_args, profiler_key) or not getattr(
                            cfg.engine_args, profiler_key, False
                        ):
                            setattr(cfg.engine_args, profiler_key, val)
                quantization = kwargs.get("quantization")
                if quantization is not None:
                    if not hasattr(cfg.engine_args, "quantization") or cfg.engine_args.quantization is None:
                        cfg.engine_args.quantization = quantization
            except Exception as e:
                logger.warning("Failed to inject LoRA config for stage: %s", e)

        return config_path, stage_configs

    # ==================== Public API ====================

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
        lora_request: Any = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        *,
        resumable: bool = False,
    ) -> None:
        """Process stage-0 input locally, then send to the Orchestrator.

        Input processing and output
        processor registration happen here in the caller's thread, avoiding
        a queue + coroutine-switch round-trip.  The Orchestrator receives a
        ready-to-submit OmniEngineCoreRequest.
        """
        msg = self._build_add_request_message(
            request_id=request_id,
            prompt=prompt,
            prompt_text=prompt_text,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            reasoning_ended=reasoning_ended,
            resumable=resumable,
        )
        if self.request_queue is None:
            raise RuntimeError("request_queue is not initialized")
        self.request_queue.sync_q.put_nowait(msg)

        # CFG companion expansion: create and enqueue companion requests
        # so the AR stage also generates their KV caches.
        if self.prompt_expand_func is not None and final_stage_id > 0:
            original_prompt = msg.get("original_prompt", prompt)
            effective_spl = msg.get("sampling_params_list", [])
            stage0_params = effective_spl[0] if effective_spl else None
            if stage0_params is not None:
                self._enqueue_cfg_companions(request_id, original_prompt, stage0_params, effective_spl)

    async def add_request_async(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
        lora_request: Any = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        *,
        resumable: bool = False,
    ) -> None:
        """Async add_request API."""
        self.add_request(
            request_id=request_id,
            prompt=prompt,
            prompt_text=prompt_text,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            reasoning_ended=reasoning_ended,
            resumable=resumable,
        )

    def add_streaming_update(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
        *,
        resumable: bool = True,
    ) -> None:
        """Send an incremental streaming update for an existing request."""
        msg = self._build_add_request_message(
            request_id=request_id,
            prompt=prompt,
            prompt_text=prompt_text,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            arrival_time=arrival_time,
            resumable=resumable,
            message_type="streaming_update",
        )
        if self.request_queue is None:
            raise RuntimeError("request_queue is not initialized")
        self.request_queue.sync_q.put_nowait(msg)

    async def add_streaming_update_async(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
        *,
        resumable: bool = True,
    ) -> None:
        """Async wrapper for add_streaming_update()."""
        self.add_streaming_update(
            request_id=request_id,
            prompt=prompt,
            prompt_text=prompt_text,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            arrival_time=arrival_time,
            resumable=resumable,
        )

    def try_get_output(self, timeout: float = 0.001) -> dict[str, Any] | None:
        """Read one output message from the Orchestrator output queue."""
        if self.output_queue is None:
            return None
        try:
            return self.output_queue.sync_q.get(timeout=timeout)
        except queue.Empty:
            if not self.is_alive():
                raise RuntimeError("Orchestrator died unexpectedly. See logs above.")
            return None

    async def try_get_output_async(self) -> dict[str, Any] | None:
        """Async read from the Orchestrator output queue."""
        if self.output_queue is None:
            return None
        try:
            return self.output_queue.sync_q.get_nowait()
        except queue.Empty:
            if not self.is_alive():
                raise RuntimeError("Orchestrator died unexpectedly. See logs above.")
            return None

    def get_stage_metadata(self, stage_id: int) -> dict[str, Any]:
        """Get cached metadata for a stage."""
        return self.stage_metadata[stage_id]

    def abort(self, request_ids: list[str]) -> None:
        """Send abort message to the Orchestrator."""
        if self.request_queue is None:
            raise RuntimeError("request_queue is not initialized")
        self.request_queue.sync_q.put_nowait(
            {
                "type": "abort",
                "request_ids": request_ids,
            }
        )

    async def abort_async(self, request_ids: list[str]) -> None:
        """Async abort API."""
        self.abort(request_ids)

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        stage_ids: list[int] | None = None,
    ) -> list[Any]:
        """Send a control RPC to the Orchestrator and wait for aggregated results.

        This uses a dedicated RPC output queue so control-plane messages do not
        race with the normal request output polling loop.
        """
        if self.request_queue is None:
            raise RuntimeError("request_queue is not initialized")
        if self.rpc_output_queue is None:
            raise RuntimeError("rpc_output_queue is not initialized")

        rpc_id = uuid.uuid4().hex
        msg = {
            "type": "collective_rpc",
            "rpc_id": rpc_id,
            "method": method,
            "args": tuple(args),
            "kwargs": kwargs or {},
            "stage_ids": stage_ids,
        }

        with self._rpc_lock:
            self.request_queue.sync_q.put_nowait(msg)
            deadline = None if timeout is None else time.monotonic() + timeout

            while True:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    result_msg = self.rpc_output_queue.sync_q.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError(f"collective_rpc timed out after {timeout} seconds") from exc

                if result_msg.get("type") == "error":
                    raise RuntimeError(result_msg.get("error", "Orchestrator returned an error message"))

                if result_msg.get("type") != "collective_rpc_result":
                    logger.warning(
                        "[AsyncOmniEngine] Dropping unexpected rpc queue message type=%s",
                        result_msg.get("type"),
                    )
                    continue

                if result_msg.get("rpc_id") != rpc_id:
                    logger.warning(
                        "[AsyncOmniEngine] Dropping mismatched rpc result rpc_id=%s expected=%s",
                        result_msg.get("rpc_id"),
                        rpc_id,
                    )
                    continue

                return list(result_msg.get("results", []))

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        stage_ids: list[int] | None = None,
    ) -> list[Any]:
        """Async wrapper around collective_rpc()."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.collective_rpc(
                method=method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
                stage_ids=stage_ids,
            ),
        )

    def is_alive(self) -> bool:
        """Whether the orchestrator thread is alive."""
        return bool(self.orchestrator_thread.is_alive())

    def shutdown(self) -> None:
        """Send shutdown message and wait for the Orchestrator thread to exit."""
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True
        finalizer = getattr(self, "_weak_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer.detach()

        logger.info("[AsyncOmniEngine] Shutting down Orchestrator")
        try:
            if self.request_queue is not None:
                self.request_queue.sync_q.put_nowait({"type": "shutdown"})
        except Exception:
            pass
        if self.is_alive():
            self.orchestrator_thread.join(timeout=10)
            if self.orchestrator_thread.is_alive():
                logger.warning("[AsyncOmniEngine] Orchestrator thread did not exit in time")

        for q in (self.request_queue, self.output_queue, self.rpc_output_queue):
            if q is None:
                continue
            try:
                q.close()
            except Exception:
                pass

        if self._omni_master_server is not None:
            try:
                self._omni_master_server.stop()
            except Exception:
                logger.exception("[AsyncOmniEngine] Failed to stop OmniMasterServer during shutdown")
            self._omni_master_server = None

    def _try_shutdown(self, *args, **kwargs) -> None:
        try:
            self.shutdown()
        except Exception:
            logger.exception(*args, **kwargs)
