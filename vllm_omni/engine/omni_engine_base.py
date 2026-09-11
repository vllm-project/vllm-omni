# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniEngineBase: process/thread ownership and transport shared by every Omni engine."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import copy
import queue
import threading
import time
import uuid
import weakref
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import janus
from vllm import envs as vllm_envs
from vllm.logger import init_logger
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.config.config_factory import StageConfigFactory, with_trust_remote_code_override
from vllm_omni.config.stage_config import _DEPLOY_DIR, load_deploy_config
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    parse_attention_config,
    resolve_model_class_name,
)
from vllm_omni.diffusion.io_support import get_diffusion_output_type
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.resolver import OmniConfigResolution, resolve_omni_config
from vllm_omni.config.stage_config import (
    DuplexSessionRuntimeConfig,
    PipelineConfig,
    load_deploy_config,
)
from vllm_omni.data_entry_keys import REQUEST_ARTIFACT_DIRS_KEY, TRANSFORM_OWNED_META_KEYS
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_engine_utils import (
    SHUTDOWN_ENQUEUE_TIMEOUT_S,
    SHUTDOWN_JOIN_TIMEOUT_S,
    enqueue_orchestrator_shutdown,
    is_abort_transport_shutdown,
    is_janus_sync_queue_shutdown,
    shutdown_runtime_after_orchestrator,
    weak_shutdown_async_omni_engine,
)
from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AbortResultMessage,
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    EngineQueueMessage,
    ErrorMessage,
    OutputMessage,
)
from vllm_omni.engine.orchestrator import OrchestratorBase
from vllm_omni.engine.rpc_result_router import CorrelatedRpcClient
from vllm_omni.engine.stage_client import StageClient
from vllm_omni.engine.stage_init_utils import build_stage0_input_processor
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.engine.stage_runtime import (
    StageRuntimeInfo,
    create_stage_runtime,
)
from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin
from vllm_omni.entrypoints.utils import parse_stage_overrides
from vllm_omni.inputs.data import OmniInteractionPrompt, OmniSamplingParams
from vllm_omni.metrics.prometheus import OmniRequestCounter

logger = init_logger(__name__)

_STARTUP_POLL_INTERVAL_S = 1.0
_REQUEST_QUEUE_MAXSIZE = 256
_ConfigResolutionResult = OmniConfigResolution | tuple[str | None, list[Any], str | None]


def load_and_resolve_stage_configs(
    model: str,
    kwargs: dict[str, Any],
    *,
    trust_remote_code: bool | None,
    deploy_config_path: str | None,
    stage_overrides: Mapping[str, Mapping[str, Any]] | None,
    strategy_config_path: str | None,
) -> OmniConfigResolution:
    """Compatibility seam delegating to the single config resolver."""
    return resolve_omni_config(
        model,
        trust_remote_code=trust_remote_code,
        cli_overrides=kwargs,
        deploy_config_path=deploy_config_path,
        stage_overrides=stage_overrides,
        strategy_config_path=strategy_config_path,
    )


class OmniEngineBase:
    """Generic engine: launches an orchestrator in a background thread.

    Shared by ``AsyncOmniEngine`` (turn-based requests) and ``DuplexOmniEngine``
    (duplex sessions). Subclasses construct their orchestrator in
    ``_create_orchestrator``.

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

    # Class-level defaults so tests that bypass __init__ via object.__new__
    # don't AttributeError when stage-init / forward paths touch these attrs.
    _log_stats: bool = False
    _coordinator_runtime: Any = None
    _transfer_emitter: Any = None
    _prom_metrics: Any = None
    _enable_orch_monitor: bool = False
    # Lazily created by get_output_blocking_async().
    _output_drain_executor: concurrent.futures.ThreadPoolExecutor | None = None

    def __init__(
        self,
        model: str,
        stage_init_timeout: int = 300,
        init_timeout: int = 600,
        single_stage_mode: bool = False,
        transfer_emitter: Any = None,
        prom_metrics: Any = None,
        log_stats: bool = False,
        tokenizer: str | None = None,
        trust_remote_code: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        # Cached by get_diffusion_od_config().
        self._diffusion_od_config_view: Any = None
        startup_timeout = int(init_timeout)
        # Forwarded into Orchestrator so its _forward_to_next_stage path can
        # emit per-edge transfer_tx_s / transfer_size_bytes histograms.
        # Optional: when None, Orchestrator silently skips TX emit (existing
        # RX path still works via OrchestratorAggregator).
        self._transfer_emitter = transfer_emitter
        self._prom_metrics = prom_metrics
        # Drives upstream EngineCore + scheduler stats production. When False
        # the engine skips SchedulerStats / IterationStats; the per-(stage,
        # replica) vllm:* wrap stays registered but reads zero. Respects the
        # --log-stats CLI flag set by the user via OmniBase.
        self._log_stats = log_stats
        self._enable_orch_monitor = bool(kwargs.pop("enable_orch_monitor", False))

        logger.info(f"[OmniEngine] Initializing with model {model}")

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

        # New omni-coordinator flags. Consumed only in single_stage_mode.
        # ``omni_dp_size_local`` is process-local: each invocation (head and
        # every headless) launches that many replicas for its own stage.
        self._omni_dp_size_local: int = int(kwargs.get("omni_dp_size_local") or 1)
        if self._omni_dp_size_local < 1:
            raise ValueError(f"--omni-dp-size-local must be >= 1, got {self._omni_dp_size_local}")
        self._omni_lb_policy: str = str(kwargs.get("omni_lb_policy") or "random")
        self._omni_heartbeat_timeout: float = float(kwargs.get("omni_heartbeat_timeout") or 30.0)
        if self._omni_heartbeat_timeout <= 0:
            raise ValueError(f"--omni-heartbeat-timeout must be > 0, got {self._omni_heartbeat_timeout}")
        # Concurrent same-device stage init (admission + SH/EX phase locks).
        # Sourced from the parallel_stage_init orchestrator/CLI arg (config,
        # not an env var); default False preserves serial init.
        self._parallel_stage_init: bool = bool(kwargs.get("parallel_stage_init") or False)

        if single_stage_mode:
            logger.info(
                "[OmniEngine] Single-stage mode enabled (stage_id_filter=%s, master=%s:%s)",
                self._single_stage_id_filter,
                self._omni_master_address,
                self._omni_master_port,
            )

        # Keep the historical tuple return from _resolve_stage_configs while
        # retaining the richer resolver result for pipeline-wide settings.
        # Overrides of that private seam fall back to the factory below.
        deploy_config_path = kwargs.get("deploy_config")
        # ``trust_remote_code`` is tri-state (bool | None): ``None`` means "not
        # specified" so stage-config resolution can defer to the deploy yaml's
        # per-stage value (see ``with_trust_remote_code_override``). The
        # restriction path below loads the top-level HF config via vLLM's
        # ``get_config``, which needs a real bool, so collapse ``None`` to the
        # default ``False`` here (#5495).
        pipeline_config = StageConfigFactory.get_pipeline_config(
            model=model,
            trust_remote_code=bool(trust_remote_code),
            deploy_config_path=deploy_config_path,
        )
        self.pipeline_config = pipeline_config
        self.endpoint_restrictions = pipeline_config.endpoint_restrictions if pipeline_config is not None else ()
        # The resolved deploy profile (explicit --deploy-config or the pipeline
        # default). Duplex engines read ``deploy_config.duplex_session`` and
        # ``session_mode`` from it; the turn-based engine only keeps it for
        # introspection.
        self.deploy_config = None
        deploy_config_source: str | Path | None = deploy_config_path
        if deploy_config_source is None and pipeline_config is not None:
            default_name = getattr(pipeline_config, "default_deploy_config_name", None)
            if default_name:
                deploy_config_source = _DEPLOY_DIR / default_name
        elif isinstance(deploy_config_source, str) and not Path(deploy_config_source).exists():
            candidate = _DEPLOY_DIR / deploy_config_source
            if candidate.exists():
                deploy_config_source = candidate
        if deploy_config_source is not None:
            try:
                self.deploy_config = load_deploy_config(deploy_config_source)
            except Exception:
                if deploy_config_path is not None:
                    # The caller named this file: failing to load it is their error.
                    raise
                logger.warning(
                    "[OmniEngine] Could not load the pipeline's default deploy config %s",
                    deploy_config_source,
                    exc_info=True,
                )

        # Subclasses check deployment invariants here, before any stage
        # process is launched (a failure is cheap at this point).
        self._validate_deployment()

        # Tri-state: None means "not specified" — the deploy yaml's per-stage
        # trust_remote_code stays in effect. An explicit True/False here is a
        # global override (precedence: caller > deploy yaml > default False);
        # the merge rule lives in with_trust_remote_code_override.
        kwargs = with_trust_remote_code_override(kwargs, trust_remote_code)
        self._config_resolution: OmniConfigResolution | None = None
        self.config_path, self.stage_configs = self._resolve_stage_configs(
            model,
            kwargs,
            trust_remote_code=trust_remote_code,
        )
        if self._config_resolution is None:
            pipeline_config = StageConfigFactory.get_pipeline_config(
                model=model,
                trust_remote_code=bool(trust_remote_code),
                deploy_config_path=deploy_config_path,
            )
            self._set_pipeline_runtime_config(pipeline_config, self.config_path)
        else:
            self._set_pipeline_runtime_config(
                self._config_resolution.pipeline_config,
                self._config_resolution.config_path,
            )

        self.num_stages = len(self.stage_configs)
        stage0_args = getattr(self.stage_configs[0], "engine_args", None) if self.num_stages > 0 else None
        self.async_chunk = bool(getattr(stage0_args, "async_chunk", False))
        self.stage_pools: list[StagePool] = []
        self.stage_clients: list[StageClient] = []  # logical-stage view for external readers
        self.input_processor: InputProcessor | None = None
        self.prompt_transform_func: Any | None = None
        self.prompt_expand_func: Any | None = None
        self.supported_tasks: tuple[str, ...] = ("generate",)
        self.default_sampling_params_list: list[OmniSamplingParams] = []
        self.stage_metadata: list[StageRuntimeInfo] = []
        # Janus queues are constructed eagerly here (not deferred to the
        # orchestrator thread) so the master server's ROUTER thread always
        # sees a non-None ``self.request_queue`` when on_register fires.
        # ``async_q`` lazily binds to whatever event loop first awaits on
        # it (the orchestrator loop), so cross-thread use stays correct.
        self.request_queue: janus.Queue[EngineQueueMessage] = janus.Queue(maxsize=_REQUEST_QUEUE_MAXSIZE)
        self.output_queue: janus.Queue[EngineQueueMessage] = janus.Queue()
        self.rpc_output_queue: janus.Queue[EngineQueueMessage] = janus.Queue()
        self._shutdown_called = False
        self._weak_finalizer: weakref.finalize | None = None
        self._correlated_rpc_client: CorrelatedRpcClient | None = None
        self._running_counter = OmniRequestCounter()
        self._engines_waiting_counter = OmniRequestCounter()

        logger.info(f"[OmniEngine] Launching Orchestrator thread with {self.num_stages} stages")

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
        self._correlated_rpc_client = CorrelatedRpcClient(
            self.request_queue.sync_q,
            self.rpc_output_queue.sync_q,
        )

        # Stage runtime fields are assigned directly on self by the bootstrap thread.
        self._weak_finalizer = weakref.finalize(
            self,
            weak_shutdown_async_omni_engine,
            self.orchestrator_thread,
            self.request_queue,
            self.output_queue,
            self.rpc_output_queue,
            self._correlated_rpc_client,
        )

        logger.info(f"[OmniEngine] Orchestrator ready with {self.num_stages} stages")

    def get_diffusion_od_config(self) -> Any:
        """Expose the diffusion ``model_class_name`` to client-side model-extras.

        The worker holds the full config; here we just resolve the pipeline class
        name from the model config (cached). ``model_class_name`` may be ``None``.
        """
        if self._diffusion_od_config_view is None:
            from types import SimpleNamespace

            from vllm_omni.diffusion.data import resolve_model_class_name
            from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

            model_class_name = resolve_model_class_name(self.model)
            metadata = get_diffusion_model_metadata(model_class_name)
            self._diffusion_od_config_view = SimpleNamespace(
                model_class_name=model_class_name,
                supports_multimodal_inputs=metadata.supports_multimodal_inputs,
                max_multimodal_image_inputs=metadata.max_multimodal_image_inputs,
                supports_mixed_reference_inputs=metadata.supports_mixed_reference_inputs,
            )
        return self._diffusion_od_config_view

    def _initialize_stages(self, stage_init_timeout: int) -> None:
        """Initialize stage clients/processors via StageRuntime and assign to self."""
        self._runtime = create_stage_runtime(
            stage_configs=self.stage_configs,
            model=self.model,
            config_path=self.config_path,
            single_stage_mode=self.single_stage_mode,
            stage_init_timeout=stage_init_timeout,
            async_chunk=self.async_chunk,
            tokenizer=self.tokenizer,
            parallel_stage_init=self._parallel_stage_init,
            single_stage_id_filter=self._single_stage_id_filter,
            omni_master_address=self._omni_master_address,
            omni_master_port=self._omni_master_port,
            omni_dp_size_local=self._omni_dp_size_local,
            omni_heartbeat_timeout=self._omni_heartbeat_timeout,
            omni_lb_policy=self._omni_lb_policy,
            request_queue=self.request_queue,
            log_stats=self._log_stats,
        )
        self._runtime.initialize()

        self.num_stages = len(self.stage_configs)
        self.stage_pools = self._runtime.stage_pools
        self.stage_clients = [
            cast(StageClient, pool.stage_client) for pool in self.stage_pools if pool.stage_client is not None
        ]
        self.stage_vllm_configs = [pool.stage_vllm_config for pool in self.stage_pools]
        self.output_processors = [pool.output_processor for pool in self.stage_pools]
        self.input_processor = (
            build_stage0_input_processor(self.stage_vllm_configs[0])
            if self.stage_vllm_configs and self.stage_vllm_configs[0] is not None
            else None
        )
        self.prompt_transform_func = (
            getattr(self.stage_clients[0], "prompt_transform_func", None) if self.stage_clients else None
        )
        self.prompt_expand_func = next(
            (
                getattr(client, "prompt_expand_func", None)
                for client in self.stage_clients
                if getattr(client, "prompt_expand_func", None) is not None
            ),
            None,
        )
        self.default_sampling_params_list = [client.default_sampling_params for client in self.stage_clients]
        self.stage_metadata = [
            StageRuntimeInfo(
                final_output=client.final_output,
                final_output_type=client.final_output_type,
                stage_type=client.stage_type,
                model_stage=getattr(client, "model_stage", None),
            )
            for client in self.stage_clients
        ]
        supported_tasks: set[str] = set()
        if any(getattr(client, "is_comprehension", False) for client in self.stage_clients):
            supported_tasks.add("generate")
        if any(meta.final_output_type == "audio" for meta in self.stage_metadata):
            supported_tasks.add("speech")
        self.supported_tasks = tuple(supported_tasks) if supported_tasks else ("generate",)

    def _bootstrap_orchestrator(
        self,
        stage_init_timeout: int,
        startup_future: concurrent.futures.Future,
    ) -> None:
        """Create loop, initialize stages, then run Orchestrator."""

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run_orchestrator() -> None:
            self._initialize_stages(stage_init_timeout)

            pd_config = self._detect_pd_config()

            membership_controller = self._runtime.create_membership_controller()

            orchestrator = self._create_orchestrator(
                request_async_queue=self.request_queue.async_q,
                output_async_queue=self.output_queue.async_q,
                rpc_async_queue=self.rpc_output_queue.async_q,
                stage_pools=self.stage_pools,
                async_chunk=self.async_chunk,
                pd_config=pd_config,
                membership_controller=membership_controller,
                running_counter=self._running_counter,
                engines_waiting_counter=self._engines_waiting_counter,
                transfer_emitter=self._transfer_emitter,
                prom_metrics=self._prom_metrics,
                log_stats=self._log_stats,
                enable_orch_monitor=self._enable_orch_monitor,
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
            logger.exception("[OmniEngine] Orchestrator thread crashed")
            error_text = str(e) or "Orchestrator thread crashed"
            try:
                error_msg = ErrorMessage(error=error_text, fatal=True)
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
                logger.exception("[OmniEngine] Failed during orchestrator loop cleanup")
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def _validate_deployment(self) -> None:
        """Seam: check ``pipeline_config`` / ``deploy_config`` before stages start (default: nothing)."""

    def _create_orchestrator(self, **orchestrator_kwargs: Any) -> OrchestratorBase:
        """Construct this engine's orchestrator (runs on the orchestrator thread after stage init)."""
        raise NotImplementedError

    def _wait_for_orchestrator_init(self, startup_future: concurrent.futures.Future, startup_timeout: int) -> None:
        """
        Wait for orchestrator startup future to return ready. Raises exception on any failures to the init process.
        """
        deadline = time.monotonic() + startup_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "[OmniEngine] Orchestrator startup timed out after %ss. "
                    "Multi-stage deployments that initialize stages sequentially on one device "
                    "or load checkpoints from slow storage may need larger --init-timeout and "
                    "--stage-init-timeout values.",
                    startup_timeout,
                )
                self._try_shutdown("[OmniEngine] Failed to cleanup after orchestrator startup timeout")
                raise TimeoutError(f"Orchestrator did not become ready within {startup_timeout}s")
            try:
                startup_future.result(
                    timeout=min(remaining, _STARTUP_POLL_INTERVAL_S),
                )
                break
            except concurrent.futures.TimeoutError:
                if not self.orchestrator_thread.is_alive():
                    self._try_shutdown("[OmniEngine] Failed to cleanup after orchestrator startup failure")
                    if startup_future.done():
                        startup_future.result()  # re-raises the real exception
                    raise RuntimeError("Orchestrator thread died during startup")
            except Exception:
                self._try_shutdown("[OmniEngine] Failed to cleanup after orchestrator startup failure")
                raise

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
        if cache_backend == "mag_cache":
            return {
                "mag_threshold": 0.24,
                "mag_max_skip_steps": 5,
                "mag_retention_ratio": 0.1,
            }
        if cache_backend == "step_cache":
            return {
                "step_cache_dit_enabled": True,
                "velocity_sim_thresholds": [0.95, 0.93],
                "velocity_skip_countdowns": [4, 2],
                "step_cache_dit_min_history": 2,
                "step_cache_dit_max_history": 2,
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
            cache_config = OmniEngineBase._get_default_cache_config(cache_backend)
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
            logger.warning("[OmniEngine] Could not extract PD bootstrap address: %s", exc)

        logger.info(
            "[OmniEngine] PD disaggregation detected: prefill=stage-%d, decode=stage-%d, bootstrap=%s",
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
            logger.warning("[OmniEngine] Could not extract prefill engine_id: %s", exc)

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
        cache_config = OmniEngineBase._normalize_cache_config(
            cache_backend,
            normalized_kwargs.get("cache_config", None),
        )

        parallel_config = normalized_kwargs.get("parallel_config")
        if isinstance(parallel_config, dict):
            parallel_config = DiffusionParallelConfig.from_dict(parallel_config)
        if parallel_config is None:
            ulysses_degree = normalized_kwargs.get("ulysses_degree") or 1
            ring_degree = normalized_kwargs.get("ring_degree") or 1
            allgather_degree = normalized_kwargs.get("allgather_degree") or 1
            ulysses_mode = normalized_kwargs.get("ulysses_mode") or "strict"
            ulysses_a2a_permute = bool(normalized_kwargs.get("ulysses_a2a_permute", False))
            sequence_parallel_size = normalized_kwargs.get("sequence_parallel_size")
            pipeline_parallel_size = normalized_kwargs.get("pipeline_parallel_size") or 1
            data_parallel_size = normalized_kwargs.get("data_parallel_size")
            tensor_parallel_size = normalized_kwargs.get("tensor_parallel_size") or 1
            cfg_parallel_size = normalized_kwargs.get("cfg_parallel_size") or 1
            pipeline_parallel_size = normalized_kwargs.get("pipeline_parallel_size") or 1
            vae_patch_parallel_size = normalized_kwargs.get("vae_patch_parallel_size") or 1
            vae_parallel_mode = normalized_kwargs.get("vae_parallel_mode") or "tile"
            text_encoder_tp_size = normalized_kwargs.get("text_encoder_tp_size") or 1
            enable_expert_parallel = normalized_kwargs.get("enable_expert_parallel") or False
            use_hsdp = normalized_kwargs.get("use_hsdp", False)
            hsdp_shard_size = normalized_kwargs.get("hsdp_shard_size", -1)
            hsdp_replicate_size = normalized_kwargs.get("hsdp_replicate_size", 1)
            if sequence_parallel_size is None:
                sequence_parallel_size = allgather_degree if allgather_degree > 1 else ulysses_degree * ring_degree

            parallel_config = DiffusionParallelConfig(
                pipeline_parallel_size=pipeline_parallel_size,
                data_parallel_size=data_parallel_size,
                tensor_parallel_size=tensor_parallel_size,
                enable_expert_parallel=enable_expert_parallel,
                sequence_parallel_size=sequence_parallel_size,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                allgather_degree=allgather_degree,
                ulysses_mode=ulysses_mode,
                ulysses_a2a_permute=ulysses_a2a_permute,
                cfg_parallel_size=cfg_parallel_size,
                vae_patch_parallel_size=vae_patch_parallel_size,
                vae_parallel_mode=vae_parallel_mode,
                text_encoder_tp_size=text_encoder_tp_size,
                use_hsdp=use_hsdp,
                hsdp_shard_size=hsdp_shard_size,
                hsdp_replicate_size=hsdp_replicate_size,
            )

        num_gpus = normalized_kwargs.get("num_gpus")
        if num_gpus is not None:
            num_gpus = int(num_gpus)
            parallel_config.resolve_data_parallel_size(num_gpus)

        num_devices = max(1, int(parallel_config.world_size))
        devices = ",".join(str(i) for i in range(num_devices))
        model_class_name = kwargs.get("model_class_name", None)
        final_output_type = get_diffusion_output_type(model_class_name)

        attention_config = None
        if (
            kwargs.get("diffusion_attention_config") is not None
            or kwargs.get("diffusion_attention_backend") is not None
            or kwargs.get("fastvideo_vsa_topk") is not None
        ):
            attention_config = parse_attention_config(
                kwargs.get("diffusion_attention_config"),
                attention_backend=kwargs.get("diffusion_attention_backend"),
                fastvideo_vsa_topk=kwargs.get("fastvideo_vsa_topk"),
            )

        extras = dict(kwargs.get("extras") or {})
        for key, default in (
            ("auxiliary_text_encoder", None),
            ("default_llama_model_id", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        ):
            top_level_value = kwargs.get(key)
            if top_level_value is not None:
                extras[key] = top_level_value
            else:
                extras.setdefault(key, default)

        stage_engine_args = {
            "max_num_seqs": kwargs.get("max_num_seqs") or 1,
            "parallel_config": parallel_config,
            # Default-stage construction bypasses the structured projection.
            # Runner selection remains owned by the selected engine/platform.
            "engine_backend": kwargs.get("engine_backend", "default"),
            "model_class_name": kwargs.get("model_class_name", None),
            "task_type": kwargs.get("task_type", None),
            "model_config": kwargs.get("model_config", None),
            "additional_config": kwargs.get("additional_config", None),
            "step_execution": kwargs.get("step_execution", False),
            "request_batch_max_wait_ms": kwargs.get("request_batch_max_wait_ms", 0.0),
            "vae_use_slicing": kwargs.get("vae_use_slicing", False),
            "vae_use_tiling": kwargs.get("vae_use_tiling", False),
            "cache_backend": cache_backend,
            "cache_config": cache_config,
            "enable_cache_dit_summary": kwargs.get("enable_cache_dit_summary", False),
            "diffusion_offload_config": kwargs.get("diffusion_offload_config", None),
            "enable_cpu_offload": kwargs.get("enable_cpu_offload", False),
            "enable_layerwise_offload": kwargs.get("enable_layerwise_offload", False),
            "enable_distributed_layerwise_offload": kwargs.get("enable_distributed_layerwise_offload", False),
            "dlo_use_allgather": kwargs.get("dlo_use_allgather", True),
            "dlo_resident_layers": kwargs.get("dlo_resident_layers", 0),
            "host_weight_runtime_mode": kwargs.get("host_weight_runtime_mode", "disabled"),
            "host_weight_runtime_root": kwargs.get("host_weight_runtime_root"),
            "dlo_host_registration_limit_gib": kwargs.get("dlo_host_registration_limit_gib", 0.0),
            "enforce_eager": False if kwargs.get("enforce_eager") is None else kwargs.get("enforce_eager"),
            "diffusion_compile_granularity": (
                "regional"
                if kwargs.get("diffusion_compile_granularity") is None
                else kwargs["diffusion_compile_granularity"]
            ),
            "diffusion_compile_dynamic": (
                True if kwargs.get("diffusion_compile_dynamic") is None else kwargs["diffusion_compile_dynamic"]
            ),
            "fa_deterministic": bool(kwargs.get("fa_deterministic", False)),
            "boundary_ratio": kwargs.get("boundary_ratio", None),
            "flow_shift": kwargs.get("flow_shift", None),
            "diffusion_load_format": kwargs.get("diffusion_load_format", "default"),
            "lora_path": kwargs.get("lora_path", None),
            "lora_scale": kwargs.get("lora_scale", 1.0),
            "lora_backend": kwargs.get("lora_backend", "peft"),
            "custom_pipeline_args": kwargs.get("custom_pipeline_args", None),
            "worker_extension_cls": kwargs.get("worker_extension_cls", None),
            "trust_remote_code": (False if kwargs.get("trust_remote_code") is None else kwargs["trust_remote_code"]),
            "distributed_executor_backend": kwargs.get("distributed_executor_backend"),
            "enable_sleep_mode": kwargs.get("enable_sleep_mode", False),
            "enable_prompt_embed_cache": kwargs.get("enable_prompt_embed_cache", False),
            "prompt_embed_cache_size": kwargs.get("prompt_embed_cache_size", 32),
            "enable_multithread_weight_load": kwargs.get("enable_multithread_weight_load", True),
            "num_weight_load_threads": kwargs.get("num_weight_load_threads", 4),
            "quantization": kwargs.get("quantization", None),
            "quantization_config": kwargs.get("quantization_config", None),
            "diffusion_kv_cache_dtype": kwargs.get("diffusion_kv_cache_dtype", None),
            "diffusion_kv_cache_skip_steps": kwargs.get("diffusion_kv_cache_skip_steps", None),
            "diffusion_kv_cache_skip_layers": kwargs.get("diffusion_kv_cache_skip_layers", None),
            **({"diffusion_attention_config": attention_config} if attention_config is not None else {}),
            "force_cutlass_fp8": bool(kwargs.get("force_cutlass_fp8", False)),
            "enable_diffusion_pipeline_profiler": kwargs.get("enable_diffusion_pipeline_profiler", False),
            "streaming_output": kwargs.get("diffusion_streaming_output", False),
            "enable_ar_profiler": kwargs.get("enable_ar_profiler", False),
            "extras": extras,
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
        if num_gpus is not None:
            stage_engine_args["num_gpus"] = num_gpus
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
                "engine_input_source": [],
                "default_sampling_params": stage_default_sampling_params,
                "final_output": True,
                "final_output_type": final_output_type,
            }
        ]
        default_stage_cfg[0]["engine_args"]["model_stage"] = "diffusion"
        return default_stage_cfg

    def _apply_strategy_lb_policy(self, derived: str | None, kwargs: dict[str, Any]) -> None:
        """Apply a strategy-derived ``omni_lb_policy`` to the engine.

        Precedence: an explicit ``--omni-lb-policy`` always wins. ``"random"`` is
        the engine default and is treated as "unset" (indistinguishable from no
        flag), so a strategy value overrides it. If the user explicitly passed a
        non-default policy that conflicts with the strategy-derived one, raise so
        the mismatch is not silently ignored.
        """
        if not derived:
            return
        explicit = kwargs.get("omni_lb_policy")
        user_set = explicit is not None and str(explicit) != "random"
        if user_set:
            if str(explicit) != str(derived):
                raise ValueError(
                    f"Conflicting load-balancer policy: --omni-lb-policy={explicit!r} was given "
                    f"but the composable-parallel strategy derived omni_lb_policy={derived!r}. "
                    "Drop --omni-lb-policy to use the strategy value, or make them match."
                )
            return
        if self._omni_lb_policy != str(derived):
            logger.info(
                "[composable_parallel] applying strategy-derived omni_lb_policy=%r (was %r).",
                derived,
                self._omni_lb_policy,
            )
            self._omni_lb_policy = str(derived)

    @staticmethod
    def _create_default_diffusion_stage_cfg(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Compatibility seam for the factory-owned diffusion fallback."""
        return StageConfigFactory.create_default_diffusion(kwargs)

    def _set_pipeline_runtime_config(
        self,
        pipeline_config: PipelineConfig | None,
        config_path: str | None,
    ) -> None:
        """Initialize engine-wide settings resolved from pipeline metadata."""
        self.endpoint_restrictions = pipeline_config.endpoint_restrictions if pipeline_config is not None else ()
        self._duplex_runtime_extension_path = (
            pipeline_config.duplex_runtime_extension if pipeline_config is not None else None
        )
        self.duplex_serving_adapter_path = (
            pipeline_config.duplex_serving_adapter if pipeline_config is not None else None
        )
        self._duplex_control_enabled = bool(pipeline_config and pipeline_config.duplex_control_enabled)
        self.duplex_session_config = DuplexSessionRuntimeConfig()
        if config_path is not None:
            self.duplex_session_config = load_deploy_config(config_path).duplex_session

    def _resolve_stage_configs(
        self,
        model: str,
        kwargs: dict[str, Any],
        *,
        trust_remote_code: bool | None,
    ) -> tuple[str, list[Any]]:
        """Resolve stage configs and inject defaults shared by orchestrator/headless."""

        for legacy_arg in ("stage_configs_path", "stage_configs"):
            if legacy_arg in kwargs:
                raise ValueError(f"`{legacy_arg}` is no longer supported; use `deploy_config` instead.")

        # log_stats is captured by __init__; its CLI-only negative alias must
        # not cross into per-stage structured config ownership validation.
        kwargs.pop("disable_log_stats", None)
        deploy_config_path = kwargs.pop("deploy_config", None)
        strategy_config_path = kwargs.pop("strategy_config", None)
        # CLI callers arrive pre-parsed; offline Python callers may use the
        # JSON-string form documented in recipes.
        stage_overrides = parse_stage_overrides(kwargs.pop("stage_overrides", None))

        # ``diffusion_streaming_output`` is the public AsyncOmni/serve kwarg;
        # stage configs know the field as ``streaming_output``. Mirror only a
        # truthy value so the CLI's False default does not override deploy YAML.
        if kwargs.get("diffusion_streaming_output") and kwargs.get("streaming_output") is None:
            kwargs["streaming_output"] = True

        resolution = cast(
            _ConfigResolutionResult,
            load_and_resolve_stage_configs(
                model,
                kwargs,
                trust_remote_code=trust_remote_code,
                deploy_config_path=deploy_config_path,
                stage_overrides=stage_overrides,
                strategy_config_path=strategy_config_path,
            ),
        )
        if isinstance(resolution, OmniConfigResolution):
            self._config_resolution = resolution
            config_path = resolution.config_path
            stage_configs = list(resolution.stage_configs)
            strategy_lb_policy = resolution.omni_lb_policy
        else:
            # Compatibility for overrides of the historical tuple-returning
            # seam. Production always receives OmniConfigResolution above.
            config_path, stage_configs, strategy_lb_policy = resolution

        # A strategy.yaml may derive a pipeline-wide load-balancer policy. It is
        # an orchestrator-level knob (read once at construction), so apply it here
        # rather than as a per-stage config field.
        self._apply_strategy_lb_policy(strategy_lb_policy, kwargs)

        return cast(str, config_path), stage_configs

    # ==================== Public API ====================

    def try_get_output(self, timeout: float = 0.001) -> EngineQueueMessage | None:
        """Read one output message from the Orchestrator output queue."""
        try:
            return self.output_queue.sync_q.get(timeout=timeout)
        except queue.Empty:
            if not self.is_alive():
                raise RuntimeError("Orchestrator died unexpectedly. See logs above.")
            return None

    async def try_get_output_async(self) -> EngineQueueMessage | None:
        """Async read from the Orchestrator output queue."""
        try:
            return self.output_queue.sync_q.get_nowait()
        except queue.Empty:
            if not self.is_alive():
                raise RuntimeError("Orchestrator died unexpectedly. See logs above.")
            return None

    async def get_output_blocking_async(self, timeout: float = 1.0) -> EngineQueueMessage | None:
        """Blocking-wait read from the Orchestrator output queue.

        Waits up to ``timeout`` seconds in a dedicated drain thread for the
        next message (condition-variable wakeup instead of a poll cadence);
        returns ``None`` on timeout so the caller keeps its liveness check,
        mirroring ``try_get_output_async``'s contract. Used by the serving
        final-output drain when ``VLLM_OMNI_EVENT_DRIVEN_ORCH`` is on.
        """
        executor = self._output_drain_executor
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="omni-output-drain",
            )
            self._output_drain_executor = executor

        sync_q = self.output_queue.sync_q

        def _drain_get() -> EngineQueueMessage | None:
            # Exceptions are swallowed to a None sentinel: the queue may be
            # closed mid-shutdown, and an exception left on an executor future
            # after task cancellation would warn as never-retrieved.
            try:
                return sync_q.get(timeout=timeout)
            except queue.Empty:
                return None
            except Exception:
                return None

        loop = asyncio.get_running_loop()
        msg = await loop.run_in_executor(executor, _drain_get)
        if msg is None and not self.is_alive():
            raise RuntimeError("Orchestrator died unexpectedly. See logs above.")
        return msg

    def get_stage_metadata(self, stage_id: int) -> StageRuntimeInfo:
        """Get cached metadata for a stage."""
        return self.stage_metadata[stage_id]

    def abort(self, request_ids: list[str]) -> None:
        """Fire-and-forget abort: enqueue and return without waiting.

        Prefer :meth:`abort_async` when the caller needs acknowledgment that
        stage aborts, binding release, and orchestrator request cleanup finished.
        """
        if not request_ids or getattr(self, "_shutdown_called", False):
            return
        if self.request_queue is None:
            raise RuntimeError("request_queue is not initialized")
        try:
            self.request_queue.sync_q.put(AbortRequestMessage(request_ids=request_ids))
        except Exception as exc:
            if getattr(self, "_shutdown_called", False) and is_janus_sync_queue_shutdown(exc):
                return
            raise

    async def abort_async(
        self,
        request_ids: list[str],
        timeout: float | None = None,
    ) -> list[OutputMessage]:
        """Abort requests and wait for orchestrator acknowledgment.

        Unlike :meth:`abort`, this generates an ``rpc_id``, correlates the
        :class:`AbortResultMessage` via :class:`CorrelatedRpcClient`, and
        raises if the orchestrator reports failure or times out.

        Returns:
            Final-stage AR abort ``OutputMessage`` list carrying partial
            tokens generated before abort (empty for diffusion / no OP state).
        """
        if not request_ids or getattr(self, "_shutdown_called", False):
            return []
        if self.request_queue is None:
            raise RuntimeError("request_queue is not initialized")
        transport = self._correlated_rpc_client
        if transport is None:
            raise RuntimeError("correlated RPC client is not initialized")

        rpc_id = uuid.uuid4().hex
        msg = AbortRequestMessage(request_ids=request_ids, rpc_id=rpc_id)

        def _wait() -> AbortResultMessage:
            result_msg = transport.execute(
                ("abort", rpc_id),
                msg,
                timeout=timeout,
                timeout_message=f"abort timed out after {timeout} seconds",
                block_on_submit=True,
            )
            if not isinstance(result_msg, AbortResultMessage):
                raise RuntimeError(f"unexpected abort result type: {type(result_msg).__name__}")
            return result_msg

        loop = asyncio.get_running_loop()
        try:
            result_msg = await loop.run_in_executor(None, _wait)
        except Exception as exc:
            if getattr(self, "_shutdown_called", False) and is_abort_transport_shutdown(exc):
                return []
            raise
        if not result_msg.success:
            raise RuntimeError(result_msg.error or "abort failed")
        return list(result_msg.abort_outputs or [])

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
        rpc_id = uuid.uuid4().hex
        msg = CollectiveRPCRequestMessage(
            rpc_id=rpc_id,
            method=method,
            timeout=timeout,
            args=tuple(args),
            kwargs=kwargs or {},
            stage_ids=stage_ids,
        )

        transport = self._correlated_rpc_client
        if transport is None:
            raise RuntimeError("correlated RPC client is not initialized")
        result_msg = transport.execute(
            ("collective", rpc_id),
            msg,
            timeout=timeout,
            timeout_message=f"collective_rpc timed out after {timeout} seconds",
            block_on_submit=True,
        )
        if not isinstance(result_msg, CollectiveRPCResultMessage):
            raise RuntimeError(f"unexpected collective RPC result type: {type(result_msg).__name__}")
        return list(result_msg.results)

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

    @property
    def rpc_client(self) -> CorrelatedRpcClient:
        """The correlated RPC transport (available once the orchestrator is up)."""
        if self._correlated_rpc_client is None:
            raise RuntimeError("engine RPC transport is not initialized")
        return self._correlated_rpc_client

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

        logger.info("[OmniEngine] Shutting down Orchestrator")
        request_queue_closed = False
        shutdown_enqueued = enqueue_orchestrator_shutdown(
            self.request_queue,
            timeout=SHUTDOWN_ENQUEUE_TIMEOUT_S,
        )
        if self.request_queue is not None and not shutdown_enqueued:
            logger.error(
                "[OmniEngine] Failed to enqueue orchestrator shutdown; "
                "closing the request queue to wake the request handler"
            )
            try:
                self.request_queue.close()
                request_queue_closed = True
            except Exception:
                logger.exception("[OmniEngine] Failed to close the request queue")

        if self._correlated_rpc_client is not None:
            try:
                self._correlated_rpc_client.close()
            except Exception:
                logger.exception("[OmniEngine] Failed to close correlated RPC client")

        orchestrator_stopped = False
        try:
            if self.is_alive():
                self.orchestrator_thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
            orchestrator_stopped = not self.is_alive()
            if not orchestrator_stopped:
                logger.error(
                    "[OmniEngine] Orchestrator did not stop within %.1f seconds; continuing cleanup",
                    SHUTDOWN_JOIN_TIMEOUT_S,
                )
        except Exception:
            logger.exception("[OmniEngine] Failed to join Orchestrator thread")

        for q in (self.request_queue, self.output_queue, self.rpc_output_queue):
            try:
                if not (q is self.request_queue and request_queue_closed):
                    q.close()
            except Exception:
                pass

        if self._output_drain_executor is not None:
            # Any in-flight blocking get bails out within its ≤1 s timeout
            # (or immediately via the queue close above), so don't wait.
            self._output_drain_executor.shutdown(wait=False)
            self._output_drain_executor = None

        if hasattr(self, "_runtime") and self._runtime is not None and orchestrator_stopped:
            try:
                self._runtime.shutdown()
            except Exception:
                logger.exception("[OmniEngine] Failed to shutdown StageRuntime")
        elif hasattr(self, "_runtime") and self._runtime is not None:
            logger.warning("[OmniEngine] Deferring StageRuntime shutdown until the Orchestrator exits")
            threading.Thread(
                target=shutdown_runtime_after_orchestrator,
                args=(self.orchestrator_thread, self._runtime),
                daemon=True,
                name="omni-stage-runtime-shutdown",
            ).start()

        # ── Release CuMem allocator memory pool ──────────────────────────────
        # When enable_sleep_mode is in use, the CuMem (CUDA Virtual Memory
        # Management) allocator holds model weights in a singleton memory pool
        # that lives in the parent process.  Killing the engine-core subprocess
        # does NOT release this pool — the weights stay resident on the GPU
        # and can cause CUDA OOM for subsequent engine instances (especially
        # large models like BAGEL-7B-MoT whose weights alone consume ~134 GiB).
        #
        # CuMemAllocator.sleep() is NOT idempotent — calling it on already-
        # slept entries causes CUDA_ERROR_INVALID_VALUE at cumem_allocator
        # cuMemRelease (double-free of the memory handle).  Use release_pools()
        # instead, which is the designed cleanup path: it drops MemPool refs
        # and lets the destructor/free path handle asleep entries correctly
        # (returns a null handle so the C extension skips unmap/release).
        try:
            from vllm.device_allocator.cumem import CuMemAllocator, cumem_available

            if cumem_available:
                allocator = CuMemAllocator.get_instance()
                allocator.release_pools()
                logger.debug("[OmniEngine] Released CuMem memory pool during shutdown")
        except Exception:
            pass

    def _try_shutdown(self, *args, **kwargs) -> None:
        try:
            self.shutdown()
        except Exception:
            logger.exception(*args, **kwargs)
