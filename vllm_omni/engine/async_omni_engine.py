"""
Async Omni Engine for vLLM-Omni multi-stage runtime.

AsyncOmniEngine in the caller's thread is a thin proxy that communicates
with the Orchestrator (running in a background thread) via janus queues.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.diffusion.data import DiffusionParallelConfig
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
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
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
    StartedLlmStage,
    acquire_device_locks,
    build_diffusion_config,
    build_engine_args_dict,
    build_vllm_config,
    cleanup_failed_stage_initialization,
    close_started_llm_stage,
    extract_stage_metadata,
    finalize_initialized_stages,
    get_stage_connector_spec,
    initialize_diffusion_stage,
    inject_kv_stage_info,
    load_omni_transfer_config_for_model,
    prepare_engine_environment,
    release_device_locks,
    setup_stage_devices,
    terminate_alive_proc,
)
from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs
from vllm_omni.inputs.preprocess import OmniInputPreprocessor
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.engine.arg_utils import OmniEngineArgs

logger = init_logger(__name__)


def _patch_generation_config_if_needed(model_config: Any) -> None:
    """Ensure try_get_generation_config won't crash for models whose HF
    config.json lacks model_type (e.g. CosyVoice3). We probe it once;
    if it raises, we monkey-patch the method to return None."""
    try:
        model_config.try_get_generation_config()
    except Exception:
        model_config.try_get_generation_config = lambda: {}


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

    return OmniEngineCoreRequest(
        request_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        data_parallel_rank=request.data_parallel_rank,
        prompt_embeds=prompt_embeds,
        client_index=request.client_index,
        current_wave=request.current_wave,
        priority=request.priority,
        trace_headers=request.trace_headers,
        resumable=request.resumable,
        external_req_id=request.external_req_id,
        reasoning_ended=request.reasoning_ended,
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
    return OmniEngineCoreRequest(
        request_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        data_parallel_rank=request.data_parallel_rank,
        prompt_embeds=request.prompt_embeds,
        client_index=request.client_index,
        current_wave=request.current_wave,
        priority=request.priority,
        trace_headers=request.trace_headers,
        resumable=request.resumable,
        external_req_id=request.external_req_id,
        reasoning_ended=request.reasoning_ended,
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

        # Merge typed engine_args fields into kwargs; explicit kwargs take priority.
        if engine_args is not None:
            ea_dict = {
                f.name: getattr(engine_args, f.name)
                for f in dataclasses.fields(engine_args)
                if not f.name.startswith("_")
            }
            # Remove model since it is passed as a positional arg already.
            ea_dict.pop("model", None)
            kwargs = {**ea_dict, **kwargs}

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

        self.num_stages = len(self.stage_configs)
        stage0_args = getattr(self.stage_configs[0], "engine_args", None) if self.num_stages > 0 else None
        self.async_chunk = bool(getattr(stage0_args, "async_chunk", False))
        self.stage_clients: list[Any] = []
        self.stage_vllm_configs: list[Any] = []
        self.output_processors: list[MultimodalOutputProcessor | None] = []
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

        # Wait for stage/runtime initialization result from orchestrator thread.
        try:
            startup_future.result(timeout=startup_timeout)
        except concurrent.futures.TimeoutError as e:
            try:
                self.shutdown()
            except Exception:
                logger.exception("[AsyncOmniEngine] Failed to cleanup after orchestrator startup timeout")
            raise TimeoutError(f"Orchestrator did not become ready within {startup_timeout}s") from e
        except Exception:
            try:
                self.shutdown()
            except Exception:
                logger.exception("[AsyncOmniEngine] Failed to cleanup after orchestrator startup failure")
            raise

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

    def _launch_llm_stage(
        self,
        stage_cfg: Any,
        metadata: Any,
        stage_connector_spec: dict[str, Any],
        stage_init_timeout: int,
        llm_stage_launch_lock: threading.Lock,
        omni_kv_connector: tuple[dict[str, Any] | None, str | None, str | None] = (None, None, None),
    ) -> StartedLlmStage:
        """Launch one LLM stage to READY state in a helper thread."""
        started_stage: StartedLlmStage | None = None
        lock_fds: list[int] = []
        device_control_env = current_omni_platform.device_control_env_var
        try:
            proc = None
            handshake_address = None
            with ExitStack() as launch_stack:
                with llm_stage_launch_lock:
                    previous_visible_devices = os.environ.get(device_control_env)
                    try:
                        setup_stage_devices(metadata.stage_id, metadata.runtime_cfg)
                        engine_args_dict = build_engine_args_dict(
                            stage_cfg,
                            self.model,
                            stage_connector_spec=stage_connector_spec,
                        )
                        omni_conn_cfg, omni_from, omni_to = omni_kv_connector
                        if omni_conn_cfg:
                            omni_kv = engine_args_dict.get("omni_kv_config") or {}
                            if not isinstance(omni_kv, dict):
                                omni_kv = dict(omni_kv)
                            omni_kv["connector_config"] = omni_conn_cfg
                            omni_kv["omni_from_stage"] = omni_from
                            omni_kv["omni_to_stage"] = omni_to
                            omni_kv.setdefault("stage_id", metadata.stage_id)
                            engine_args_dict["omni_kv_config"] = omni_kv
                        vllm_config, executor_class = build_vllm_config(
                            stage_cfg,
                            self.model,
                            stage_connector_spec=stage_connector_spec,
                            engine_args_dict=engine_args_dict,
                        )
                        lock_fds = acquire_device_locks(
                            metadata.stage_id,
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
                                    stage_id=metadata.stage_id,
                                    stage_config=stage_cfg,
                                )
                            )
                            started_stage = StartedLlmStage(
                                stage_id=metadata.stage_id,
                                metadata=metadata,
                                vllm_config=vllm_config,
                                executor_class=executor_class,
                                addresses=addresses,
                                engine_manager=engine_manager,
                                coordinator=coordinator,
                            )
                        else:
                            addresses, proc, handshake_address = spawn_stage_core(
                                vllm_config=vllm_config,
                                executor_class=executor_class,
                                log_stats=False,
                            )
                            started_stage = StartedLlmStage(
                                stage_id=metadata.stage_id,
                                metadata=metadata,
                                vllm_config=vllm_config,
                                executor_class=executor_class,
                                addresses=addresses,
                                proc=proc,
                            )
                        logger.info("[AsyncOmniEngine] Stage %s engine launch started", metadata.stage_id)
                        # Keep the stage-specific device visibility until vLLM
                        # finishes starting all child processes.
                        if self.single_stage_mode and self._omni_master_server is not None:
                            launch_stack.close()
                        else:
                            assert proc is not None
                            assert handshake_address is not None
                            complete_stage_handshake(
                                proc, handshake_address, addresses, vllm_config, stage_init_timeout
                            )
                        logger.info("[AsyncOmniEngine] Stage %s engine startup completed", metadata.stage_id)
                    finally:
                        if previous_visible_devices is None:
                            current_omni_platform.unset_device_control_env_var()
                        else:
                            current_omni_platform.set_device_control_env_var(previous_visible_devices)

            assert started_stage is not None
            return started_stage
        except Exception:
            if started_stage is not None:
                close_started_llm_stage(started_stage)
            raise
        finally:
            if lock_fds:
                release_device_locks(lock_fds)

    def _create_remote_llm_stage(
        self,
        stage_cfg: Any,
        metadata: Any,
        stage_connector_spec: dict[str, Any],
        stage_init_timeout: int,
        omni_master_server: OmniMasterServer,
    ) -> StartedLlmStage:
        """Attach to a remote engine core and wait for its startup handshake."""
        started_stage: StartedLlmStage | None = None
        try:
            raw_stage_cfg = omni_master_server.get_stage_config(
                metadata.stage_id,
                timeout_s=stage_init_timeout,
            )
            if raw_stage_cfg is None:
                raise ValueError(f"Remote stage {metadata.stage_id} registered without stage config")
            stage_cfg = OmegaConf.create(raw_stage_cfg)
            engine_args_dict = build_engine_args_dict(
                stage_cfg,
                self.model,
                stage_connector_spec=stage_connector_spec,
            )
            vllm_config, executor_class = build_vllm_config(
                stage_cfg,
                self.model,
                stage_connector_spec=stage_connector_spec,
                engine_args_dict=engine_args_dict,
            )
            vllm_config.parallel_config.data_parallel_size_local = 0
            launch_cm = connect_remote_engine_cores(
                vllm_config=vllm_config,
                omni_master_server=omni_master_server,
                stage_id=metadata.stage_id,
            )
            logger.info("[AsyncOmniEngine] Stage %s remote engine handshake started", metadata.stage_id)
            with launch_cm as (engine_manager, coordinator, addresses):
                started_stage = StartedLlmStage(
                    stage_id=metadata.stage_id,
                    metadata=metadata,
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    engine_manager=engine_manager,
                    coordinator=coordinator,
                    addresses=addresses,
                )
            logger.info("[AsyncOmniEngine] Stage %s remote engine startup completed", metadata.stage_id)
            assert started_stage is not None
            return started_stage
        except Exception:
            if started_stage is not None:
                close_started_llm_stage(started_stage)
            raise

    def _launch_diffusion_stage(
        self,
        stage_cfg: Any,
        metadata: Any,
        omni_master_server: OmniMasterServer,
    ) -> StageDiffusionClient:
        """Launch a local diffusion stage on OmniMasterServer-allocated sockets."""
        proc = None
        try:
            od_config = build_diffusion_config(self.model, stage_cfg, metadata)
            handshake_address, request_address, response_address = register_stage_with_omni_master(
                omni_master_address=omni_master_server.address,
                omni_master_port=omni_master_server.port,
                omni_stage_id=metadata.stage_id,
                omni_stage_config=stage_cfg,
                return_addresses=True,
            )
            logger.info(
                "[AsyncOmniEngine] Stage %s diffusion registration completed",
                metadata.stage_id,
            )
            proc, _, _, _ = spawn_diffusion_proc(
                self.model,
                od_config,
                handshake_address=handshake_address,
                request_address=request_address,
                response_address=response_address,
            )
            complete_diffusion_handshake(proc, handshake_address)
            logger.info(
                "[AsyncOmniEngine] Stage %s diffusion startup completed",
                metadata.stage_id,
            )
            return StageDiffusionClient.from_addresses(
                metadata,
                request_address=request_address,
                response_address=response_address,
                proc=proc,
                batch_size=self.diffusion_batch_size,
            )
        except Exception:
            if proc is not None:
                terminate_alive_proc(proc)
            raise

    def _create_remote_diffusion_stage(
        self,
        metadata: Any,
        stage_init_timeout: int,
        omni_master_server: OmniMasterServer,
    ) -> StageDiffusionClient:
        """Attach to a remote diffusion stage registered with OmniMasterServer."""
        remote_stage_cfg = OmegaConf.create(
            omni_master_server.get_stage_config(
                metadata.stage_id,
                timeout_s=stage_init_timeout,
            )
        )
        remote_metadata = extract_stage_metadata(remote_stage_cfg)
        addresses = omni_master_server.get_zmq_addresses(metadata.stage_id)
        logger.info(
            "[AsyncOmniEngine] Stage %s remote diffusion startup completed",
            metadata.stage_id,
        )
        return StageDiffusionClient.from_addresses(
            remote_metadata,
            request_address=addresses.inputs[0],
            response_address=addresses.outputs[0],
            batch_size=self.diffusion_batch_size,
        )

    def _attach_llm_stage(
        self,
        started: StartedLlmStage,
    ) -> tuple[Any, Any, Any, InputProcessor | None]:
        """Attach a READY LLM stage to the orchestrator event loop."""

        client_addresses: dict[str, str] = {
            "input_address": started.addresses.inputs[0],
            "output_address": started.addresses.outputs[0],
        }
        if started.addresses.frontend_stats_publish_address is not None:
            client_addresses["stats_update_address"] = started.addresses.frontend_stats_publish_address

        try:
            stage_client = StageEngineCoreClientBase.make_async_mp_client(
                vllm_config=started.vllm_config,
                executor_class=started.executor_class,
                metadata=started.metadata,
                client_addresses=client_addresses,
                proc=started.proc,
                engine_manager=started.engine_manager,
                coordinator=started.coordinator,
            )
            started.proc = None
            started.engine_manager = None
            started.coordinator = None
        except Exception:
            close_started_llm_stage(started)
            raise

        try:
            if started.vllm_config.model_config.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = cached_tokenizer_from_config(
                    model_config=started.vllm_config.model_config,
                )
            output_processor = MultimodalOutputProcessor(
                tokenizer=tokenizer,
                log_stats=False,
                engine_core_output_type=started.metadata.engine_output_type,
            )
            input_processor = None
            if started.stage_id == 0:
                # Some omni models (e.g. CosyVoice3) have an empty HF
                # config.json without model_type, which causes
                # try_get_generation_config -> AutoConfig.from_pretrained
                # to raise ValueError. Patch it to return None so
                # InputProcessor doesn't crash.
                _patch_generation_config_if_needed(started.vllm_config.model_config)
                input_processor = InputProcessor(vllm_config=started.vllm_config)
                # Use omni preprocessor so text-only prompts with
                # mm_processor_kwargs (e.g. GLM-Image t2i target_h/target_w)
                # still go through multimodal processor path.
                input_processor.input_preprocessor = OmniInputPreprocessor(
                    vllm_config=started.vllm_config,
                    renderer=input_processor.renderer,
                )
        except Exception:
            try:
                stage_client.shutdown()
            except Exception as cleanup_error:
                logger.warning(
                    "[AsyncOmniEngine] Failed to cleanup stage %s after attach failure: %s",
                    started.stage_id,
                    cleanup_error,
                )
            raise

        logger.info("[AsyncOmniEngine] Stage %s initialized", started.stage_id)
        return stage_client, output_processor, started.vllm_config, input_processor

    def _initialize_stages(self, stage_init_timeout: int) -> None:
        """Initialize stage clients/processors in orchestrator thread and assign to self."""
        device_control_env = current_omni_platform.device_control_env_var

        num_stages = self.num_stages
        stage_clients: list[Any | None] = [None] * num_stages
        output_processors: list[Any | None] = [None] * num_stages
        stage_vllm_configs: list[Any | None] = [None] * num_stages
        input_processor: InputProcessor | None = None
        llm_stage_positions: list[int] = []
        llm_launch_futures: dict[int, concurrent.futures.Future[StartedLlmStage]] = {}
        started_llm_stages: dict[int, StartedLlmStage] = {}
        llm_stage_launch_lock = threading.Lock()

        async_chunk = self.async_chunk
        prompt_expand_func = None
        llm_stage_count = sum(
            1 for stage_cfg in self.stage_configs if getattr(stage_cfg, "stage_type", "llm") != "diffusion"
        )

        prepare_engine_environment()
        omni_transfer_config = load_omni_transfer_config_for_model(self.model, self.config_path)

        # ------------------------------------------------------------------ #
        # Single-stage mode: start OmniMasterServer before launching stages.  #
        # ------------------------------------------------------------------ #
        if self.single_stage_mode:
            if not self._omni_master_address or not self._omni_master_port:
                raise ValueError(
                    "AsyncOmniEngine single_stage_mode requires both "
                    "omni_master_address and omni_master_port to be set."
                )
            # Collect all configured stage IDs for pre-allocation.
            all_stage_ids: list[int] = []
            seen_stage_ids: set[int] = set()
            for i, sc in enumerate(self.stage_configs):
                stage_id = int(getattr(sc, "stage_id", i))
                if stage_id in seen_stage_ids:
                    raise ValueError(
                        f"Duplicate stage_id {stage_id!r} detected among configured stages; stage_ids must be unique."
                    )
                seen_stage_ids.add(stage_id)
                all_stage_ids.append(stage_id)
            self._omni_master_server = OmniMasterServer(
                master_address=self._omni_master_address,
                master_port=self._omni_master_port,
                stage_ids=all_stage_ids,
            )
            self._omni_master_server.start()
            logger.info(
                "[AsyncOmniEngine] OmniMasterServer started for stages %s",
                all_stage_ids,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, llm_stage_count),
                thread_name_prefix="llm-stage-launch",
            ) as launch_executor:
                for stage_idx, stage_cfg in enumerate(self.stage_configs):
                    metadata = extract_stage_metadata(stage_cfg)
                    configured_stage_id = metadata.stage_id
                    logger.info("[AsyncOmniEngine] Initializing stage %s", configured_stage_id)
                    if metadata.prompt_expand_func is not None:
                        prompt_expand_func = metadata.prompt_expand_func

                    if self.single_stage_mode:
                        metadata.runtime_cfg = None

                    stage_connector_spec = get_stage_connector_spec(
                        omni_transfer_config=omni_transfer_config,
                        stage_id=configured_stage_id,
                        async_chunk=async_chunk,
                    )

                    omni_kv_connector = resolve_omni_kv_config_for_stage(omni_transfer_config, configured_stage_id)

                    if metadata.stage_type == "diffusion":
                        is_remote_diffusion_stage = (
                            self.single_stage_mode
                            and self._single_stage_id_filter is not None
                            and configured_stage_id != self._single_stage_id_filter
                        )
                        if is_remote_diffusion_stage:
                            assert self._omni_master_server is not None
                            stage_clients[stage_idx] = self._create_remote_diffusion_stage(
                                metadata,
                                stage_init_timeout,
                                self._omni_master_server,
                            )
                            continue

                        with llm_stage_launch_lock:
                            previous_visible_devices = os.environ.get(device_control_env)
                            try:
                                setup_stage_devices(configured_stage_id, metadata.runtime_cfg)
                                omni_conn_cfg, omni_from, omni_to = omni_kv_connector
                                if omni_conn_cfg:
                                    from vllm_omni.entrypoints.utils import inject_omni_kv_config

                                    inject_omni_kv_config(stage_cfg, omni_conn_cfg, omni_from, omni_to)
                                inject_kv_stage_info(stage_cfg, configured_stage_id)
                                if self.single_stage_mode:
                                    assert self._omni_master_server is not None
                                    stage_clients[stage_idx] = self._launch_diffusion_stage(
                                        stage_cfg,
                                        metadata,
                                        self._omni_master_server,
                                    )
                                else:
                                    stage_clients[stage_idx] = initialize_diffusion_stage(
                                        self.model,
                                        stage_cfg,
                                        metadata,
                                        stage_init_timeout=stage_init_timeout,
                                        batch_size=self.diffusion_batch_size,
                                    )
                                logger.info(
                                    "[AsyncOmniEngine] Stage %s initialized (diffusion, batch_size=%d)",
                                    configured_stage_id,
                                    self.diffusion_batch_size,
                                )
                            finally:
                                if previous_visible_devices is None:
                                    current_omni_platform.unset_device_control_env_var()
                                else:
                                    current_omni_platform.set_device_control_env_var(previous_visible_devices)
                        continue

                    llm_stage_positions.append(stage_idx)

                    # In single-stage mode, stages that don't match the local
                    # stage_id filter are skipped.
                    if (
                        self.single_stage_mode
                        and self._single_stage_id_filter is not None
                        and configured_stage_id != self._single_stage_id_filter
                    ):
                        assert self._omni_master_server is not None
                        llm_launch_futures[stage_idx] = launch_executor.submit(
                            self._create_remote_llm_stage,
                            stage_cfg,
                            metadata,
                            stage_connector_spec,
                            stage_init_timeout,
                            self._omni_master_server,
                        )
                    else:
                        llm_launch_futures[stage_idx] = launch_executor.submit(
                            self._launch_llm_stage,
                            stage_cfg,
                            metadata,
                            stage_connector_spec,
                            stage_init_timeout,
                            llm_stage_launch_lock,
                            omni_kv_connector,
                        )

                concurrent.futures.wait(list(llm_launch_futures.values()))

                for stage_idx in llm_stage_positions:
                    started_llm_stages[stage_idx] = llm_launch_futures[stage_idx].result()

            attach_futures: dict[concurrent.futures.Future[tuple[Any, Any, Any, InputProcessor | None]], int] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(llm_stage_positions)),
                thread_name_prefix="llm-stage-attach",
            ) as attach_executor:
                for stage_idx in llm_stage_positions:
                    attach_futures[attach_executor.submit(self._attach_llm_stage, started_llm_stages[stage_idx])] = (
                        stage_idx
                    )

                for future in concurrent.futures.as_completed(attach_futures):
                    stage_idx = attach_futures[future]
                    stage_client, output_processor, vllm_config, stage0_input_processor = future.result()
                    stage_clients[stage_idx] = stage_client
                    output_processors[stage_idx] = output_processor
                    stage_vllm_configs[stage_idx] = vllm_config
                    if stage0_input_processor is not None:
                        input_processor = stage0_input_processor

            initialized_stage_clients, default_sampling_params_list, stage_metadata = finalize_initialized_stages(
                stage_clients,
                input_processor,
            )
        except Exception:
            for stage_id, future in llm_launch_futures.items():
                if not future.done() or future.cancelled() or future.exception() is not None:
                    continue
                started_llm_stages.setdefault(stage_id, future.result())
            logger.exception(
                "[AsyncOmniEngine] Stage initialization failed; shutting down %s initialized stage(s)",
                len([stage_client for stage_client in stage_clients if stage_client is not None]),
            )
            cleanup_failed_stage_initialization(
                stage_clients,
                [started_llm_stages[stage_idx] for stage_idx in llm_stage_positions if stage_idx in started_llm_stages],
            )
            if self._omni_master_server is not None:
                try:
                    self._omni_master_server.stop()
                except Exception:
                    logger.exception("[AsyncOmniEngine] Failed to stop OmniMasterServer during stage-init cleanup")
            raise

        self.stage_clients = initialized_stage_clients
        self.output_processors = output_processors
        self.stage_vllm_configs = stage_vllm_configs
        self.input_processor = input_processor
        self.prompt_expand_func = prompt_expand_func
        # TODO(Peiqi): Hack here
        supported_tasks: set[str] = set()
        if any(getattr(stage_client, "is_comprehension", False) for stage_client in initialized_stage_clients):
            supported_tasks.add("generate")
        if any(metadata.get("final_output_type") == "audio" for metadata in stage_metadata):
            supported_tasks.add("speech")
        self.supported_tasks = tuple(supported_tasks) if supported_tasks else ("generate",)

        self.default_sampling_params_list = default_sampling_params_list
        self.stage_metadata = stage_metadata

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
            orchestrator = Orchestrator(
                request_async_queue=self.request_queue.async_q,
                output_async_queue=self.output_queue.async_q,
                rpc_async_queue=self.rpc_output_queue.async_q,
                async_chunk=self.async_chunk,
                stage_clients=self.stage_clients,
                output_processors=self.output_processors,
                stage_vllm_configs=self.stage_vllm_configs,
            )
            if not startup_future.done():
                startup_future.set_result(asyncio.get_running_loop())
            await orchestrator.run()

        try:
            loop.run_until_complete(_run_orchestrator())
        except Exception as e:
            if not startup_future.done():
                startup_future.set_exception(RuntimeError(f"Orchestrator initialization failed: {e}"))
            logger.exception("[AsyncOmniEngine] Orchestrator thread crashed")
            try:
                if self.output_queue is not None:
                    self.output_queue.sync_q.put_nowait({"type": "error", "error": "Orchestrator thread crashed"})
                if self.rpc_output_queue is not None:
                    self.rpc_output_queue.sync_q.put_nowait({"type": "error", "error": "Orchestrator thread crashed"})
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
        if stage_type != "diffusion" and not isinstance(prompt, EngineCoreRequest):
            # Inject global_request_id into the raw prompt.
            if isinstance(prompt, dict):
                _inject_global_id(prompt, request_id)
            elif isinstance(prompt, list):
                for item in prompt:
                    _inject_global_id(item, request_id)

            # Full input processing (tokenization, multimodal, etc.)
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

            # Register with stage 0's output processor.
            output_prompt_text = prompt_text
            if output_prompt_text is None and isinstance(original_prompt, dict):
                output_prompt_text = original_prompt.get("prompt")
            self.output_processors[0].add_request(
                request=request,
                prompt=output_prompt_text,
                parent_req=None,
                request_index=0,
                queue=None,
            )
            prompt = request

        return {
            "type": message_type,
            "request_id": request_id,
            "prompt": prompt,
            "original_prompt": original_prompt,
            "sampling_params_list": effective_sampling_params_list,
            "final_stage_id": final_stage_id,
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
            request = _upgrade_to_omni_request(request, companion_prompt)
            request.external_req_id = cid

            self.output_processors[0].add_request(
                request=request,
                prompt=companion_prompt,
                parent_req=None,
                request_index=0,
                queue=None,
            )

            self.request_queue.sync_q.put_nowait(
                {
                    "type": "add_companion_request",
                    "companion_id": cid,
                    "parent_id": parent_id,
                    "role": ep.role,
                    "prompt": request,
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

    @staticmethod
    def _create_default_diffusion_stage_cfg(kwargs: dict[str, Any]) -> list:
        """Create a default single-stage diffusion config from kwargs."""
        # We temporally create a default config for diffusion stage.
        # In the future, we should merge the default config with the user-provided config.
        normalized_kwargs = dict(kwargs)

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
            tensor_parallel_size = normalized_kwargs.get("tensor_parallel_size") or 1
            cfg_parallel_size = normalized_kwargs.get("cfg_parallel_size") or 1
            vae_patch_parallel_size = normalized_kwargs.get("vae_patch_parallel_size") or 1
            use_hsdp = normalized_kwargs.get("use_hsdp", False)
            hsdp_shard_size = normalized_kwargs.get("hsdp_shard_size", -1)
            hsdp_replicate_size = normalized_kwargs.get("hsdp_replicate_size", 1)
            if sequence_parallel_size is None:
                sequence_parallel_size = ulysses_degree * ring_degree

            parallel_config = DiffusionParallelConfig(
                pipeline_parallel_size=1,
                data_parallel_size=1,
                tensor_parallel_size=tensor_parallel_size,
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

        stage_engine_args = {
            "max_num_seqs": 1,
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
            "enforce_eager": kwargs.get("enforce_eager", False),
            "boundary_ratio": kwargs.get("boundary_ratio", None),
            "flow_shift": kwargs.get("flow_shift", None),
            "diffusion_load_format": kwargs.get("diffusion_load_format", "default"),
            "custom_pipeline_args": kwargs.get("custom_pipeline_args", None),
            "worker_extension_cls": kwargs.get("worker_extension_cls", None),
            "enable_sleep_mode": kwargs.get("enable_sleep_mode", False),
            "enable_multithread_weight_load": kwargs.get("enable_multithread_weight_load", True),
            "num_weight_load_threads": kwargs.get("num_weight_load_threads", 4),
            "quantization": kwargs.get("quantization", None),
            "enable_diffusion_pipeline_profiler": kwargs.get("enable_diffusion_pipeline_profiler", False),
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

        default_stage_cfg = [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {
                    "process": True,
                    "devices": devices,
                },
                "engine_args": stage_engine_args,
                "final_output": True,
                "final_output_type": "image",
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
        """
        # worker_extension_cls is a parent field but must pass through to
        # diffusion stages for colocate worker setup.
        _keep = {"worker_extension_cls"}
        # Orchestrator-level OmniEngineArgs fields that are consumed by
        # _resolve_stage_configs and must not leak into per-stage configs
        # (stage_configs_path would trigger the create_model_config guard).
        _strip_omni = {"stage_configs_path"}
        # Fields that are always set by callers (via from_cli_args / asdict)
        # and would always appear as overridden — suppress from the warning
        # so it only surfaces genuinely surprising overrides.
        _no_warn = {"model"}

        parent_fields: dict[str, dataclasses.Field] = {f.name: f for f in dataclasses.fields(EngineArgs)}
        overridden: list[str] = []
        result: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in _strip_omni:
                continue
            if k not in parent_fields or k in _keep:
                result[k] = v
                continue
            # Detect explicitly-set values that differ from the default.
            # Values may have been through asdict() which converts dataclass
            # defaults to dicts, so normalise before comparing.
            field = parent_fields[k]
            if field.default is not dataclasses.MISSING:
                default = field.default
            elif field.default_factory is not dataclasses.MISSING:
                default = field.default_factory()
            else:
                default = dataclasses.MISSING
            if default is dataclasses.MISSING or v is None:
                continue
            # Normalise dataclass defaults to dicts for comparison
            if dataclasses.is_dataclass(default) and not isinstance(default, type):
                default = dataclasses.asdict(default)
            if v != default and k not in _no_warn:
                overridden.append(k)

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

        # Use the legacy config loading path (load_and_resolve_stage_configs).
        # StageConfigFactory wiring will be done in config refactor [2/N].
        config_path, stage_configs = load_and_resolve_stage_configs(
            model,
            stage_configs_path,
            base_kwargs,
            default_stage_cfg_factory=lambda: self._create_default_diffusion_stage_cfg(kwargs),
        )

        # Inject diffusion LoRA-related knobs from kwargs if not present in the stage config.
        for cfg in stage_configs:
            try:
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
