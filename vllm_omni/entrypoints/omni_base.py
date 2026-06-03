from __future__ import annotations

import os
import time
import weakref
from collections.abc import Sequence
from typing import Any, Literal

import huggingface_hub
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import (
    EngineQueueMessage,
    ErrorMessage,
    OutputMessage,
    StageMetricsMessage,
)
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin
from vllm_omni.entrypoints.utils import coerce_param_message_types, get_final_stage_id_for_e2e
from vllm_omni.metrics.modality import OmniModalityMetrics, observe_modality_at_finalize
from vllm_omni.metrics.prometheus import OmniPrometheusMetrics
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.metrics.transfer import OmniTransferMetrics
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.tracking_parser import TrackingNamespace

logger = init_logger(__name__)


class OmniEngineDeadError(EngineDeadError):
    _DEFAULT_MESSAGE = EngineDeadError().args[0]
    error_stage_id: int | None

    def __init__(
        self,
        message: str | None = None,
        *,
        error_stage_id: int | None = None,
        suppress_context: bool = False,
    ) -> None:
        resolved_message = message or self._DEFAULT_MESSAGE
        Exception.__init__(self, resolved_message)
        self.__suppress_context__ = suppress_context
        self.error_stage_id = error_stage_id


def _weak_shutdown_engine(engine: AsyncOmniEngine) -> None:
    """Best-effort engine cleanup for GC finalization."""
    try:
        engine.shutdown()
    except Exception:
        pass


def omni_snapshot_download(model_id: str) -> str:
    if os.path.exists(model_id):
        return model_id

    # TODO: this is just a workaround for quickly use modelscope, we should support
    # modelscope in weight loading feature instead of using `snapshot_download`
    if os.environ.get("VLLM_USE_MODELSCOPE", False):
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(model_id)

    try:
        download_weights_from_hf_specific(
            model_name_or_path=model_id,
            cache_dir=None,
            allow_patterns=["*"],
            require_all=True,
        )
    except huggingface_hub.errors.GatedRepoError:
        raise ValueError(
            f"Access to model '{model_id}' is restricted. "
            f"Visit https://huggingface.co/{model_id} to accept "
            f"the license and request access."
        )
    except huggingface_hub.errors.RepositoryNotFoundError:
        raise ValueError(f"Repository not found for '{model_id}'. Please check the model name or path.")
    except PermissionError:
        logger.warning(
            "Permission denied when downloading '%s'. Assuming the model is already cached locally.",
            model_id,
        )

    return model_id


OutputMessageHandleResult = tuple[Literal[True], None, None, None] | tuple[Literal[False], str, int, ClientRequestState]


class OmniBase(PDDisaggregationMixin):
    """Shared runtime foundation for AsyncOmni and Omni."""

    @classmethod
    def from_cli_args(
        cls,
        args: TrackingNamespace,
        model: str | None = None,
    ) -> OmniBase:
        """Build from a TrackingNamespace parsed by TrackingArgumentParser.
        Only args that are explicitly passed to parse_args are forwarded.
        """
        if not isinstance(args, TrackingNamespace):
            raise TypeError(
                f"expected args to be of type TrackingNamespace, got {type(args)}. "
                "Hint: did you parse your args with TrackingArgumentParser?"
            )

        explicit_kwargs = args.get_explicit_kwargs_dict()
        args_model = explicit_kwargs.pop("model", None) or args.model
        if model is not None and args_model is not None and model != args_model:
            raise ValueError(
                f"explicit model kwarg and args.model were both provided, but do not match [{model} != {args_model}]"
            )

        if model is None and args_model is None:
            raise ValueError(
                "model must be explicitly passed as a parsed arg in the TrackingNamespace or directly provided."
            )

        resolved_model = model or args_model
        return cls(model=resolved_model, **explicit_kwargs)

    def __init__(
        self,
        model: str,
        **kwargs: Any,
    ) -> None:
        if "engine_args" in kwargs:
            logger.warning(
                "engine_args were passed as a kwarg to an Omni instance; this is not supported. "
                "You should instead, pass the keyword arguments used to initialize the engine args "
                "directly to this object's initializer."
            )
        stage_init_timeout = kwargs.pop("stage_init_timeout", 300)
        init_timeout = kwargs.pop("init_timeout", 600)
        log_stats = kwargs.pop("log_stats", False)
        self._enable_ar_profiler = kwargs.pop("enable_ar_profiler", False)
        # NOTE: read-only lookup — must NOT pop. Popping here drops the key
        # before it reaches ``StageConfigFactory._create_from_registry``, so
        # ``--no-async-chunk`` (``async_chunk=False``) silently fails to
        # override the deploy YAML's ``async_chunk: true`` default.
        async_chunk = kwargs.get("async_chunk")
        output_modalities = kwargs.pop("output_modalities", None)
        diffusion_batch_size: int = kwargs.pop("diffusion_batch_size", 1)

        if "log_requests" in kwargs:
            raise TypeError("`log_requests` has been removed in Omni/AsyncOmni. Use `log_stats`.")
        model = omni_snapshot_download(model)
        self.__dict__["_name"] = self.__class__.__name__
        self.model = model
        self.log_stats = log_stats
        # Provisional value (mirrors the CLI/caller kwarg); the engine resolves
        # pipeline + deploy YAML + CLI precedence below and the final value is
        # re-assigned from ``self.engine.async_chunk`` after init.
        self.async_chunk = bool(async_chunk) if async_chunk is not None else False
        self.output_modalities = output_modalities or []
        self.tts_batch_max_items: int = kwargs.pop("tts_batch_max_items", 32)

        logger.info("[%s] Initializing with model %s", self.__class__.__name__, model)
        # Construct transfer_metrics first so we can hand it to AsyncOmniEngine
        # (which forwards it to the Orchestrator background thread for
        # TX-side emit; see Orchestrator._forward_to_next_stage).
        self.transfer_metrics = OmniTransferMetrics(model_name=model, log_stats=log_stats)
        st = time.time()
        self.engine = AsyncOmniEngine(
            model=model,
            init_timeout=init_timeout,
            stage_init_timeout=stage_init_timeout,
            diffusion_batch_size=diffusion_batch_size,
            transfer_emitter=self.transfer_metrics,
            log_stats=log_stats,
            **kwargs,
        )
        self._shutdown_called = False
        self._weak_finalizer = weakref.finalize(self, _weak_shutdown_engine, self.engine)
        et = time.time()
        logger.info("[%s] AsyncOmniEngine initialized in %.2f seconds", self.__class__.__name__, et - st)
        # Authoritative: ``AsyncOmniEngine`` resolves (pipeline + deploy YAML +
        # CLI overrides) through ``StageConfigFactory`` and stores the final
        # value on ``engine.async_chunk``; mirror it here so ``--no-async-chunk``
        # (explicit ``False``) is not fallen-back-through by ``or``.
        self.async_chunk = bool(getattr(self.engine, "async_chunk", False))

        self.request_states: dict[str, ClientRequestState] = {}
        self.prom_metrics = OmniPrometheusMetrics(model_name=model, log_stats=log_stats)
        self.mod_metrics = OmniModalityMetrics(model_name=model, log_stats=log_stats)

        self.default_sampling_params_list = self.engine.default_sampling_params_list
        if not self.output_modalities:
            self.output_modalities = [
                self.engine.get_stage_metadata(i).final_output_type for i in range(self.engine.num_stages)
            ]

        self._stage_meta_list = [self.engine.get_stage_metadata(i) for i in range(self.engine.num_stages)]

        logger.info(
            "[%s] Initialized with %s stages for model %s",
            self.__class__.__name__,
            self.engine.num_stages,
            model,
        )

        # PD disaggregation state (detects if a prefill/decode stage pair is configured)
        self._init_pd_state()

    @property
    def num_stages(self) -> int:
        return self.engine.num_stages

    @property
    def stage_configs(self) -> list:
        """Expose engine stage configs for PD disaggregation detection and validation."""
        return self.engine.stage_configs

    def _has_dead_stage(self) -> bool:
        for stage_client in self.engine.stage_clients:
            if getattr(stage_client, "_engine_dead", False):
                return True
            resources = getattr(stage_client, "resources", None)
            if resources is not None and getattr(resources, "engine_dead", False):
                return True
        return False

    @property
    def is_running(self) -> bool:
        return self.engine.is_alive() and not self._has_dead_stage()

    @property
    def errored(self) -> bool:
        """Whether the engine is in a non-recoverable error state.

        True when the orchestrator thread is dead **or** any stage client
        has been marked dead (e.g. diffusion worker OOM / process death).

        Checks both ``_engine_dead`` (StageDiffusionClient) and
        ``resources.engine_dead`` (StageEngineCoreClient / AsyncMPClient)
        since the two client types store the flag differently.
        """
        return not self.engine.is_alive() or self._has_dead_stage()

    def check_health(self) -> None:
        if not self.engine.is_alive():
            raise EngineDeadError("Orchestrator process is not alive")
        for stage_client in self.engine.stage_clients:
            if hasattr(stage_client, "check_health"):
                stage_client.check_health()

    def resolve_sampling_params_list(
        self,
        sampling_params_list: Sequence[Any] | Any | None,
        allow_delta_coercion: bool = False,
    ) -> Sequence[Any]:
        if sampling_params_list is None:
            normalized = self.default_sampling_params_list
            # Set the output kind to delta since no params were specified
            if allow_delta_coercion:
                normalized = coerce_param_message_types(list(normalized), is_streaming=True)

        elif isinstance(sampling_params_list, Sequence) and not isinstance(sampling_params_list, (str, bytes)):
            normalized = sampling_params_list
        elif self.num_stages == 1:
            normalized = [sampling_params_list]
        else:
            raise ValueError(f"Expected {self.num_stages} sampling params, got a single sampling params object")
        if len(normalized) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} sampling params, got {len(normalized)}")
        return normalized

    def _fire_failure_counter_if_alive(self, request_id: str) -> None:
        """Fire the abort/exception bucket of requests_success_total.

        Called from cancel / exception paths in async_omni.generate() BEFORE
        _abort_internal_requests pops request_states — that method resolves
        the internal id by dict lookup, so popping first would no-op it. We
        keep this counter fire separate from _log_summary_and_cleanup (which
        pops) so the abort path can still find the state to clean up.
        """
        req_state = self.request_states.get(request_id)
        prom = getattr(self, "prom_metrics", None)
        if req_state is None or req_state.metrics is None or prom is None:
            return
        if str(request_id) not in req_state.metrics.e2e_done:
            prom.request_failed()

    def _log_summary_and_cleanup(self, request_id: str) -> None:
        req_state = self.request_states.get(request_id)
        try:
            if req_state is None or req_state.metrics is None:
                return
            if str(request_id) not in req_state.metrics.e2e_done:
                self.prom_metrics.request_failed()
            if self.log_stats:
                # Emit per-request orchestrator timing (including e2e_total_ms)
                # before dropping request state.
                req_state.metrics.build_and_log_summary()
        except Exception:
            logger.exception(
                "[%s] Failed to build/log summary for req=%s",
                self.__class__.__name__,
                request_id,
            )
        finally:
            self.request_states.pop(request_id, None)
            # Republish gauges so any stale value left by the per-stage
            # publish in _process_single_result (which runs while the request
            # is still in self.request_states) is corrected after the pop.
            prom = getattr(self, "prom_metrics", None)
            counter = getattr(getattr(self, "engine", None), "_running_counter", None)
            if prom is not None:
                total = len(self.request_states)
                running = counter.value if counter is not None else total
                prom.set_running(running)
                prom.set_waiting(max(0, total - running))

    def _compute_final_stage_id(self, output_modalities: list[str] | None) -> int:
        return get_final_stage_id_for_e2e(
            output_modalities,
            self.output_modalities,
            self._stage_meta_list,
        )

    def _process_stage_metrics_message(self, msg: StageMetricsMessage) -> None:
        req_id = msg.request_id
        req_state = self.request_states.get(req_id)
        if req_state is None or req_state.metrics is None:
            return
        _m = msg.metrics
        stage_id = msg.stage_id
        req_state.metrics.on_stage_metrics(stage_id, req_id, _m)
        submit_ts = msg.stage_submit_ts
        now = time.time()
        if req_state.metrics.stage_first_ts[stage_id] is None:
            req_state.metrics.stage_first_ts[stage_id] = submit_ts if submit_ts is not None else now
        req_state.metrics.stage_last_ts[stage_id] = max(req_state.metrics.stage_last_ts[stage_id] or 0.0, now)

    def _handle_output_message(
        self,
        msg: EngineQueueMessage | None,
    ) -> OutputMessageHandleResult:
        """Handle one Orchestrator output-queue message."""
        if msg is None:
            return True, None, None, None

        if isinstance(msg, StageMetricsMessage):
            self._process_stage_metrics_message(msg)
            return True, None, None, None

        if isinstance(msg, ErrorMessage):
            if msg.fatal:
                raise OmniEngineDeadError(
                    msg.error,
                    error_stage_id=msg.stage_id,
                )
            raise RuntimeError(msg.error)

        if not isinstance(msg, OutputMessage):
            logger.warning("[%s] got unexpected msg type: %s", self.__class__.__name__, msg.type)
            return True, None, None, None

        req_id = msg.request_id
        stage_id = msg.stage_id

        req_state = self.request_states.get(req_id)
        if req_state is None:
            logger.debug(
                "[%s] dropping output for unknown req %s",
                self.__class__.__name__,
                req_id,
            )
            return True, None, None, None

        req_state.stage_id = stage_id

        return False, req_id, stage_id, req_state

    def _check_engine_output_error(
        self,
        result: OutputMessage,
        request_id: str,
        stage_id: int,
    ) -> None:
        """Raise if ``engine_outputs`` carries an error field.

        Raises :class:`EngineDeadError` when ``self.errored`` indicates the
        engine is unrecoverable, otherwise raises :class:`EngineGenerateError`
        (recoverable, single-request failure).
        """
        engine_outputs = result.engine_outputs
        error_text = getattr(engine_outputs, "error", None)
        if error_text is None:
            return
        logger.error(
            "[%s] Stage error for req=%s stage-%s: %s",
            self.__class__.__name__,
            request_id,
            stage_id,
            error_text,
        )
        # NOTE: O(n_stages) check for every error.
        if self.errored:
            raise OmniEngineDeadError(
                error_text,
                error_stage_id=stage_id,
            )
        raise EngineGenerateError(error_text)

    def _process_single_result(
        self,
        result: OutputMessage,
        stage_id: int,
        metrics: OrchestratorMetrics,
        req_start_ts: dict[str, float],
        wall_start_ts: float,
        final_stage_id_for_e2e: int,
    ) -> OmniRequestOutput | None:
        req_id = result.request_id
        engine_outputs = result.engine_outputs
        stage_durations = getattr(engine_outputs, "stage_durations", {})
        peak_memory_mb = getattr(engine_outputs, "peak_memory_mb", 0.0)

        # Merge AR stage timing from OrchestratorAggregator.stage_events
        if self._enable_ar_profiler:
            ar_events = metrics.stage_events.get(str(req_id), [])
            for evt in ar_events:
                if evt.stage_id != stage_id:
                    stage_durations[f"ar_stage_{evt.stage_id}"] = evt.stage_gen_time_ms / 1000.0

        # Merge pipeline timings from Orchestrator into stage_durations
        _m = result.metrics
        if _m is not None and hasattr(_m, "pipeline_timings") and _m.pipeline_timings:
            for key, value in _m.pipeline_timings.items():
                if key not in stage_durations:
                    stage_durations[key] = value

        # Merge per-stage gen times into stage_durations
        for evt in metrics.stage_events.get(str(req_id), []):
            key = f"stage_{evt.stage_id}_gen_ms"
            if key not in stage_durations:
                stage_durations[key] = evt.stage_gen_time_ms
        # Current stage gen time (not yet in stage_events at this point)
        if _m is not None:
            stage_durations.setdefault(f"stage_{stage_id}_gen_ms", _m.stage_gen_time_ms)

        finished = engine_outputs.finished

        submit_ts = result.stage_submit_ts
        now = time.time()
        if metrics.stage_first_ts[stage_id] is None:
            metrics.stage_first_ts[stage_id] = submit_ts if submit_ts is not None else now
        metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, now)

        _m = result.metrics
        if finished and _m is not None:
            metrics.on_stage_metrics(stage_id, req_id, _m)

        stage_meta = self.engine.get_stage_metadata(stage_id)
        if not stage_meta.final_output:
            return None

        output_type = getattr(engine_outputs, "final_output_type", stage_meta.final_output_type)

        try:
            rid_key = str(req_id)
            if stage_id == final_stage_id_for_e2e and rid_key not in metrics.e2e_done and finished:
                metrics.on_finalize_request(
                    stage_id,
                    req_id,
                    req_start_ts.get(req_id, wall_start_ts),
                )
                e2e_seconds = now - req_start_ts.get(req_id, wall_start_ts)
                # Extract finished_reason from upstream CompletionOutput so
                # the per-reason completion Counter is labelled correctly.
                completion_outputs = getattr(engine_outputs, "outputs", None) or []
                fr = (getattr(completion_outputs[0], "finish_reason", None) if completion_outputs else None) or "stop"
                self.prom_metrics.request_succeeded(
                    e2e_seconds,
                    finished_reason=fr,
                )

                # Modality observe inside the same finalize guard so it fires
                # once per request and inherits the try/except isolation.
                observe_modality_at_finalize(
                    self.mod_metrics,
                    output_type=output_type,
                    stage_id=stage_id,
                    replica_id=result.replica_id,
                    stage_metrics=_m,
                    engine_outputs=engine_outputs,
                )
        except Exception:
            logger.exception("[%s] Finalize request handling error", self.__class__.__name__)

        # When this result finalizes the request, the orchestrator has
        # already decremented _running_counter but _log_summary_and_cleanup
        # hasn't popped self.request_states yet — exclude the finalizing
        # request from `total` so waiting doesn't read 1 and stay stuck
        # there until the next request arrives.
        counter = getattr(self.engine, "_running_counter", None)
        is_finalizing = finished and stage_id == final_stage_id_for_e2e
        total = max(0, len(self.request_states) - (1 if is_finalizing else 0))
        running = counter.value if counter is not None else total
        self.prom_metrics.set_running(running)
        self.prom_metrics.set_waiting(max(0, total - running))

        images = getattr(engine_outputs, "images", []) if output_type == "image" else []
        return OmniRequestOutput(
            request_id=req_id or "",
            stage_id=stage_id,
            final_output_type=output_type,
            request_output=engine_outputs,
            images=images,
            trajectory_latents=getattr(engine_outputs, "trajectory_latents", None),
            trajectory_timesteps=getattr(engine_outputs, "trajectory_timesteps", None),
            trajectory_log_probs=getattr(engine_outputs, "trajectory_log_probs", None),
            trajectory_decoded=getattr(engine_outputs, "trajectory_decoded", None),
            _custom_output=getattr(engine_outputs, "_custom_output", {}),
            stage_durations=stage_durations,
            peak_memory_mb=peak_memory_mb,
        )

    def shutdown(self, timeout: float | None = None) -> None:
        logger.info("[%s] Shutting down", self.__class__.__name__)
        self._shutdown_base()

    def close(self) -> None:
        self.shutdown()

    def start_profile(
        self,
        profile_prefix: str | None = None,
        stages: list[int] | None = None,
    ) -> list[Any]:
        """Start profiling specified stages.

        Uses vLLM-compatible profile(is_start=True, profile_prefix) interface.

        Args:
            profile_prefix: Optional prefix for the trace file names.
            stages: List of stage IDs to profile. If None, profiles all stages.

        Returns:
            List of results from each stage.
        """
        return self.engine.collective_rpc(method="profile", args=(True, profile_prefix), stage_ids=stages)

    def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        """Stop profiling specified stages.

        Uses vLLM-compatible profile(is_start=False) interface.

        Args:
            stages: List of stage IDs to profile. If None, stops all stages.

        Returns:
            List of results from each stage.
        """
        return self.engine.collective_rpc(method="profile", args=(False, None), stage_ids=stages)

    def _shutdown_base(self) -> None:
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True
        finalizer = getattr(self, "_weak_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        self.engine.shutdown()
