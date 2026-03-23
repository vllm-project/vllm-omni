from __future__ import annotations

import os
import time
import types
import weakref
from collections.abc import Sequence
from pprint import pformat
from typing import TYPE_CHECKING, Any, Literal

import huggingface_hub
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.utils import get_final_stage_id_for_e2e
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.engine.arg_utils import OmniEngineArgs

logger = init_logger(__name__)


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
    except huggingface_hub.errors.RepositoryNotFoundError:
        logger.warning("Repository not found for '%s'.", model_id)
    except PermissionError:
        logger.warning(
            "Permission denied when downloading '%s'. Assuming the model is already cached locally.",
            model_id,
        )

    return model_id


OutputMessageHandleResult = tuple[Literal[True], None, None, None] | tuple[Literal[False], str, int, ClientRequestState]


class OmniBase:
    """Shared runtime foundation for AsyncOmni and Omni."""

    def __init__(
        self,
        model: str,
        **kwargs: Any,
    ) -> None:
        engine_args: OmniEngineArgs | None = kwargs.pop("engine_args", None)
        stage_init_timeout = kwargs.pop("stage_init_timeout", 300)
        init_timeout = kwargs.pop("init_timeout", 600)
        log_stats = kwargs.pop("log_stats", False)
        async_chunk = kwargs.pop("async_chunk", False)
        output_modalities = kwargs.pop("output_modalities", None)

        if "log_requests" in kwargs:
            raise TypeError("`log_requests` has been removed in Omni/AsyncOmni. Use `log_stats`.")
        model = omni_snapshot_download(model)
        self.model = model
        self.log_stats = log_stats
        self.async_chunk = async_chunk
        self.output_modalities = output_modalities or []

        logger.info("[%s] Initializing with model %s", self.__class__.__name__, model)
        st = time.time()
        self.engine = AsyncOmniEngine(
            model=model,
            engine_args=engine_args,
            init_timeout=init_timeout,
            stage_init_timeout=stage_init_timeout,
            **kwargs,
        )
        self._shutdown_called = False
        self._weak_finalizer = weakref.finalize(self, _weak_shutdown_engine, self.engine)
        et = time.time()
        logger.info("[%s] AsyncOmniEngine initialized in %.2f seconds", self.__class__.__name__, et - st)
        self.async_chunk = bool(self.async_chunk or getattr(self.engine, "async_chunk", False))

        self.request_states: dict[str, ClientRequestState] = {}

        self.default_sampling_params_list = self.engine.default_sampling_params_list
        if not self.output_modalities:
            self.output_modalities = [
                self.engine.get_stage_metadata(i).get("final_output_type") for i in range(self.engine.num_stages)
            ]

        self._stage_meta_list = [
            types.SimpleNamespace(**self.engine.get_stage_metadata(i)) for i in range(self.engine.num_stages)
        ]

        logger.info(
            "[%s] Initialized with %s stages for model %s",
            self.__class__.__name__,
            self.engine.num_stages,
            model,
        )

    @property
    def num_stages(self) -> int:
        return self.engine.num_stages

    @property
    def is_running(self) -> bool:
        return self.engine.is_alive()

    def check_health(self) -> None:
        if not self.engine.is_alive():
            raise EngineDeadError("Orchestrator process is not alive")

    def resolve_sampling_params_list(
        self,
        sampling_params_list: Sequence[Any] | Any | None,
    ) -> Sequence[Any]:
        if sampling_params_list is None:
            normalized = self.default_sampling_params_list
        elif isinstance(sampling_params_list, Sequence) and not isinstance(sampling_params_list, (str, bytes)):
            normalized = sampling_params_list
        elif self.num_stages == 1:
            normalized = [sampling_params_list]
        else:
            raise ValueError(f"Expected {self.num_stages} sampling params, got a single sampling params object")
        if len(normalized) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} sampling params, got {len(normalized)}")
        return normalized

    def _log_summary_and_cleanup(self, request_id: str) -> None:
        req_state = self.request_states.get(request_id)
        try:
            if req_state is None or req_state.metrics is None:
                return
            summary = req_state.metrics.build_and_log_summary()
            logger.info("[Summary] %s", pformat(summary, sort_dicts=False))
        except Exception:
            logger.exception(
                "[%s] Failed to build/log summary for req=%s",
                self.__class__.__name__,
                request_id,
            )
        finally:
            self.request_states.pop(request_id, None)

    def _compute_final_stage_id(self, output_modalities: list[str] | None) -> int:
        return get_final_stage_id_for_e2e(
            output_modalities,
            self.output_modalities,
            self._stage_meta_list,
        )

    def _process_stage_metrics_message(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("request_id")
        req_state = self.request_states.get(req_id)
        if req_state is None or req_state.metrics is None:
            return
        _m = msg.get("metrics")
        if _m is None:
            return
        stage_id = msg.get("stage_id", 0)
        req_state.metrics.on_stage_metrics(stage_id, req_id, _m)
        submit_ts = msg.get("stage_submit_ts")
        now = time.time()
        if req_state.metrics.stage_first_ts[stage_id] is None:
            req_state.metrics.stage_first_ts[stage_id] = submit_ts if submit_ts is not None else now
        req_state.metrics.stage_last_ts[stage_id] = max(req_state.metrics.stage_last_ts[stage_id] or 0.0, now)

    def _handle_output_message(
        self,
        msg: dict[str, Any] | None,
    ) -> OutputMessageHandleResult:
        """Handle one Orchestrator output-queue message."""
        if msg is None:
            return True, None, None, None

        msg_type = msg.get("type")
        if msg_type == "stage_metrics":
            self._process_stage_metrics_message(msg)
            return True, None, None, None

        if msg_type == "error":
            raise RuntimeError(msg.get("error", "Orchestrator returned an error message"))

        if msg_type != "output":
            logger.warning("[%s] got unexpected msg type: %s", self.__class__.__name__, msg_type)
            return True, None, None, None

        req_id = msg.get("request_id")
        if req_id is None:
            logger.warning("[%s] got output message without request_id", self.__class__.__name__)
            return True, None, None, None

        stage_id = msg.get("stage_id")
        if stage_id is None:
            logger.warning("[%s] got output message without stage_id for req=%s", self.__class__.__name__, req_id)
            return True, None, None, None

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

    def _process_single_result(
        self,
        result: dict[str, Any],
        stage_id: int,
        metrics: OrchestratorMetrics,
        req_start_ts: dict[str, float],
        wall_start_ts: float,
        final_stage_id_for_e2e: int,
    ) -> OmniRequestOutput | None:
        req_id = result.get("request_id")
        engine_outputs = result.get("engine_outputs")
        stage_durations = getattr(result["engine_outputs"], "stage_durations", {})
        finished = engine_outputs.finished

        submit_ts = result.get("stage_submit_ts")
        now = time.time()
        if metrics.stage_first_ts[stage_id] is None:
            metrics.stage_first_ts[stage_id] = submit_ts if submit_ts is not None else now
        metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, now)

        _m = result.get("metrics")
        if finished and _m is not None:
            metrics.on_stage_metrics(stage_id, req_id, _m)

        stage_meta = self.engine.get_stage_metadata(stage_id)
        if not stage_meta["final_output"]:
            return None

        try:
            rid_key = str(req_id)
            if stage_id == final_stage_id_for_e2e and rid_key not in metrics.e2e_done and finished:
                metrics.on_finalize_request(
                    stage_id,
                    req_id,
                    req_start_ts.get(req_id, wall_start_ts),
                )
        except Exception:
            logger.exception("[%s] Finalize request handling error", self.__class__.__name__)

        images = getattr(engine_outputs, "images", []) if stage_meta["final_output_type"] == "image" else []
        return OmniRequestOutput(
            stage_id=stage_id,
            final_output_type=stage_meta["final_output_type"],
            request_output=engine_outputs,
            images=images,
            stage_durations=stage_durations,
        )

    def shutdown(self) -> None:
        logger.info("[%s] Shutting down", self.__class__.__name__)
        self._shutdown_base()

    def close(self) -> None:
        self.shutdown()

    def _shutdown_base(self) -> None:
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True
        finalizer = getattr(self, "_weak_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        self.engine.shutdown()
