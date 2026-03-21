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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.engine.arg_utils import OmniEngineArgs

import janus
import torch
from omegaconf import OmegaConf
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.utils import get_engine_zmq_addresses, launch_core_engines

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    resolve_omni_kv_config_for_stage,
)
from vllm_omni.engine import (
    OmniEngineCoreRequest,
)
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.engine.stage_init_utils import (
    StartedLlmStage,
    acquire_device_locks,
    build_engine_args_dict,
    build_vllm_config,
    cleanup_failed_stage_initialization,
    close_started_llm_stage,
    extract_stage_metadata,
    finalize_initialized_stages,
    get_stage_connector_spec,
    initialize_diffusion_stage,
    load_omni_transfer_config_for_model,
    prepare_engine_environment,
    release_device_locks,
    setup_stage_devices,
)
from vllm_omni.entrypoints.utils import (
    load_and_resolve_stage_configs,
)

logger = init_logger(__name__)


def _inject_kv_stage_info(stage_cfg: Any, stage_id: int) -> None:
    """Inject stage_id and engine_input_source into omni_kv_config.

    OmniKVTransferManager needs stage_id to compute recv_stages for the
    receiving side. In the old Omni architecture, OmniDiffusion.__init__
    performed this injection; replicate it here for AsyncOmniEngine.
    """
    try:
        engine_args = stage_cfg.engine_args
        if hasattr(engine_args, "get"):
            omni_kv = engine_args.get("omni_kv_config", None)
        else:
            omni_kv = getattr(engine_args, "omni_kv_config", None)

        if omni_kv is None:
            return

        if hasattr(omni_kv, "setdefault"):
            omni_kv.setdefault("stage_id", stage_id)
        elif hasattr(omni_kv, "__setitem__"):
            if "stage_id" not in omni_kv:
                omni_kv["stage_id"] = stage_id

        engine_input_source = getattr(stage_cfg, "engine_input_source", None)
        if engine_input_source is not None:
            if hasattr(omni_kv, "setdefault"):
                omni_kv.setdefault("engine_input_source", list(engine_input_source))
            elif hasattr(omni_kv, "__setitem__") and "engine_input_source" not in omni_kv:
                omni_kv["engine_input_source"] = list(engine_input_source)
    except Exception as e:
        logger.debug("Failed to inject stage info into omni_kv_config: %s", e)


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
        **kwargs: Any,
    ) -> None:
        self.model = model
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
        from vllm_omni.platforms import current_omni_platform

        started_stage: StartedLlmStage | None = None
        lock_fds: list[int] = []
        device_control_env = current_omni_platform.device_control_env_var

        try:
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
                    addresses = get_engine_zmq_addresses(vllm_config)
                    launch_cm = launch_core_engines(
                        vllm_config=vllm_config,
                        executor_class=executor_class,
                        log_stats=False,
                        addresses=addresses,
                    )
                    engine_manager, coordinator, addresses = launch_cm.__enter__()
                    started_stage = StartedLlmStage(
                        stage_id=metadata.stage_id,
                        metadata=metadata,
                        vllm_config=vllm_config,
                        executor_class=executor_class,
                        engine_manager=engine_manager,
                        coordinator=coordinator,
                        addresses=addresses,
                    )
                finally:
                    if previous_visible_devices is None:
                        current_omni_platform.unset_device_control_env_var()
                    else:
                        current_omni_platform.set_device_control_env_var(previous_visible_devices)

            logger.info("[AsyncOmniEngine] Stage %s engine launch started", metadata.stage_id)
            launch_cm.__exit__(None, None, None)
            logger.info("[AsyncOmniEngine] Stage %s engine startup completed", metadata.stage_id)
            assert started_stage is not None
            return started_stage
        except Exception:
            if started_stage is not None:
                close_started_llm_stage(started_stage)
            raise
        finally:
            if lock_fds:
                release_device_locks(lock_fds)

    def _attach_llm_stage(
        self,
        started: StartedLlmStage,
    ) -> tuple[Any, Any, Any, InputProcessor | None]:
        """Attach a READY LLM stage to the orchestrator event loop."""

        client_addresses = {
            "input_address": started.addresses.inputs[0],
            "output_address": started.addresses.outputs[0],
        }
        if started.addresses.frontend_stats_publish_address is not None:
            client_addresses["stats_update_address"] = started.addresses.frontend_stats_publish_address

        try:
            stage_client = StageEngineCoreClient(
                vllm_config=started.vllm_config,
                executor_class=started.executor_class,
                metadata=started.metadata,
                client_addresses=client_addresses,
                engine_manager=started.engine_manager,
                coordinator=started.coordinator,
            )
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
                input_processor = InputProcessor(vllm_config=started.vllm_config)
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

        num_stages = self.num_stages
        stage_clients: list[Any | None] = [None] * num_stages
        output_processors: list[Any | None] = [None] * num_stages
        stage_vllm_configs: list[Any | None] = [None] * num_stages
        input_processor: InputProcessor | None = None
        llm_stage_ids: list[int] = []
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

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, llm_stage_count),
                thread_name_prefix="llm-stage-launch",
            ) as launch_executor:
                for stage_id, stage_cfg in enumerate(self.stage_configs):
                    logger.info("[AsyncOmniEngine] Initializing stage %s", stage_id)
                    metadata = extract_stage_metadata(stage_cfg)
                    if metadata.prompt_expand_func is not None:
                        prompt_expand_func = metadata.prompt_expand_func

                    stage_connector_spec = get_stage_connector_spec(
                        omni_transfer_config=omni_transfer_config,
                        stage_id=stage_id,
                        async_chunk=async_chunk,
                    )

                    omni_kv_connector = resolve_omni_kv_config_for_stage(omni_transfer_config, stage_id)

                    if metadata.stage_type == "diffusion":
                        setup_stage_devices(stage_id, metadata.runtime_cfg)
                        omni_conn_cfg, omni_from, omni_to = omni_kv_connector
                        if omni_conn_cfg:
                            from vllm_omni.entrypoints.utils import inject_omni_kv_config

                            inject_omni_kv_config(stage_cfg, omni_conn_cfg, omni_from, omni_to)
                        _inject_kv_stage_info(stage_cfg, stage_id)
                        stage_clients[stage_id] = initialize_diffusion_stage(self.model, stage_cfg, metadata)
                        logger.info("[AsyncOmniEngine] Stage %s initialized (diffusion)", stage_id)
                        continue

                    llm_stage_ids.append(stage_id)
                    llm_launch_futures[stage_id] = launch_executor.submit(
                        self._launch_llm_stage,
                        stage_cfg,
                        metadata,
                        stage_connector_spec,
                        stage_init_timeout,
                        llm_stage_launch_lock,
                        omni_kv_connector,
                    )

                concurrent.futures.wait(list(llm_launch_futures.values()))

                for stage_id in llm_stage_ids:
                    started_llm_stages[stage_id] = llm_launch_futures[stage_id].result()

            for stage_id in llm_stage_ids:
                started = started_llm_stages[stage_id]
                stage_client, output_processor, vllm_config, stage0_input_processor = self._attach_llm_stage(started)
                stage_clients[stage_id] = stage_client
                output_processors[stage_id] = output_processor
                stage_vllm_configs[stage_id] = vllm_config
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
                [started_llm_stages[stage_id] for stage_id in llm_stage_ids if stage_id in started_llm_stages],
            )
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
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
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
            )
            # TODO (Peiqi): add this for Qwen3-TTS only. Other models don't have
            # additional_information field in the prompt.
            request = _upgrade_to_omni_request(request, prompt)

            # Restore external_req_id to the original user-facing request_id.
            # InputProcessor.process_inputs() renames request_id to an internal
            # UUID (saving the original in external_req_id), but then overwrites
            # external_req_id with the new internal ID. We need external_req_id
            # to match the key used in Orchestrator.request_states so that
            # output routing (output.request_id lookup) can find the req_state.
            request.external_req_id = request_id

            # Register with stage 0's output processor.
            self.output_processors[0].add_request(
                request=request,
                prompt=prompt,
                parent_req=None,
                request_index=0,
                queue=None,
            )
            prompt = request

        return {
            "type": "add_request",
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

            # Run through same input processing as the main prompt
            if isinstance(companion_prompt, dict):
                _inject_global_id(companion_prompt, cid)

            request = self.input_processor.process_inputs(
                request_id=cid,
                prompt=companion_prompt,
                params=stage0_params,
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
                    "sampling_params_list": sampling_params_list,
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
                cfg_parallel_size=cfg_parallel_size,
                vae_patch_parallel_size=vae_patch_parallel_size,
                use_hsdp=use_hsdp,
                hsdp_shard_size=hsdp_shard_size,
                hsdp_replicate_size=hsdp_replicate_size,
            )

        num_devices = max(1, int(parallel_config.world_size))
        devices = ",".join(str(i) for i in range(num_devices))

        default_stage_cfg = [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {
                    "process": True,
                    "devices": devices,
                },
                "engine_args": {
                    "max_num_seqs": 1,
                    "parallel_config": parallel_config,
                    "model_class_name": kwargs.get("model_class_name", None),
                    "vae_use_slicing": kwargs.get("vae_use_slicing", False),
                    "vae_use_tiling": kwargs.get("vae_use_tiling", False),
                    "cache_backend": cache_backend,
                    "cache_config": cache_config,
                    "enable_cache_dit_summary": kwargs.get("enable_cache_dit_summary", False),
                    "enable_cpu_offload": kwargs.get("enable_cpu_offload", False),
                    "enable_layerwise_offload": kwargs.get("enable_layerwise_offload", False),
                    "enforce_eager": kwargs.get("enforce_eager", False),
                    "diffusion_load_format": kwargs.get("diffusion_load_format", "default"),
                    "custom_pipeline_args": kwargs.get("custom_pipeline_args", None),
                    "worker_extension_cls": kwargs.get("worker_extension_cls", None),
                    "enable_sleep_mode": kwargs.get("enable_sleep_mode", False),
                    "enable_multithread_weight_load": kwargs.get("enable_multithread_weight_load", True),
                    "num_weight_load_threads": kwargs.get("num_weight_load_threads", 4),
                    "quantization": kwargs.get("quantization", None),
                    "enable_diffusion_pipeline_profiler": kwargs.get("enable_diffusion_pipeline_profiler", False),
                },
                "final_output": True,
                "final_output_type": "image",
            }
        ]
        default_stage_cfg[0]["engine_args"]["model_stage"] = "diffusion"
        return default_stage_cfg

    def _resolve_stage_configs(self, model: str, kwargs: dict[str, Any]) -> tuple[str, list[Any]]:
        """Resolve stage configs and inject defaults shared by orchestrator/headless."""

        stage_configs_path = kwargs.get("stage_configs_path", None)
        explicit_stage_configs = kwargs.pop("stage_configs", None)
        if explicit_stage_configs is not None:
            logger.warning(
                "`stage_configs` is not part of the public API. "
                "Ignoring it and resolving stages from stage_configs_path/model factory."
            )

        # Use the legacy config loading path (load_and_resolve_stage_configs).
        # StageConfigFactory wiring will be done in config refactor [2/N].
        config_path, stage_configs = load_and_resolve_stage_configs(
            model,
            stage_configs_path,
            kwargs,
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
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
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
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            arrival_time=arrival_time,
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
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
    ) -> None:
        """Async add_request API."""
        self.add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            arrival_time=arrival_time,
        )

    def try_get_output(self, timeout: float = 0.001) -> dict[str, Any] | None:
        """Read one output message from the Orchestrator output queue."""
        if self.output_queue is None:
            return None
        try:
            return self.output_queue.sync_q.get(timeout=timeout)
        except queue.Empty:
            return None

    async def try_get_output_async(self) -> dict[str, Any] | None:
        """Async read from the Orchestrator output queue."""
        if self.output_queue is None:
            return None
        try:
            return self.output_queue.sync_q.get_nowait()
        except queue.Empty:
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
