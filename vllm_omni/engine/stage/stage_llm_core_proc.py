"""
Stage LLM Core Process for vLLM-Omni V1 architecture.

``StageLLMCoreProc`` inherits from vLLM's ``EngineCoreProc`` and runs the engine
core busy loop in a subprocess, communicating with ``StageLLMCoreClient`` via ZMQ.
"""

from __future__ import annotations

import contextlib
import os
import signal
from types import FrameType
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.system_utils import (
    decorate_logs,
    set_process_title,
)
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    EngineZmqAddresses,
    SignalCallback,
)

from vllm_omni.distributed.omni_coordinator import create_stage_coord_client
from vllm_omni.engine.stage_init_utils import (
    maybe_apply_audex_cfg_patches,
    set_death_signal,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request

    from vllm_omni.engine.stage.stage_core_types import StageLLMCoreRequest

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


class StageLLMCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni (LLM backend).

    Inherits from ``EngineCoreProc`` and provides its own ``run_stage_core``
    entry point for launching in a subprocess. Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    def preprocess_add_request(self, request: StageLLMCoreRequest) -> tuple[Request, int]:
        """Preserve omni payloads when vLLM builds its scheduler request."""
        scheduler_request, current_wave = super().preprocess_add_request(request)
        scheduler_request.additional_information = request.additional_information
        scheduler_request.external_req_id = getattr(request, "external_req_id", request.request_id)
        return scheduler_request, current_wave

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        omni_coord_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch the ``StageLLMCoreProc`` busy loop in a background process.

        Omni-specific kwargs:
          - ``omni_coord_address``: ROUTER address of the head-side
            :class:`OmniCoordinator`. When provided, this subprocess
            instantiates an :class:`OmniCoordClientForStage` after the
            HELLO/INIT/READY handshake completes and reports its status +
            queue length via heartbeats.
          - ``omni_stage_id``: logical stage id this replica belongs to.
            Required when ``omni_coord_address`` is provided.
          - ``omni_replica_id``: cluster-unique replica id within the stage
            (assigned by :class:`OmniMasterServer`). Used for logging /
            metrics only.
        """
        signal_callback: SignalCallback | None = None
        maybe_register_config_serialize_by_value()

        # Register vllm-omni reasoning parsers (e.g. step_audio) in this
        # subprocess so they are available when the engine core resolves
        # ``--reasoning-parser``. The forked subprocess starts with a fresh
        # ReasoningParserManager.
        try:
            import vllm_omni.reasoning  # noqa: F401
        except ImportError:
            logger.warning(
                "Failed to import vllm_omni.reasoning in subprocess; custom "
                "reasoning parsers (e.g. step_audio) will not be available."
            )

        engine_core: StageLLMCoreProc | None = None
        coord_client = None
        try:
            stage_label = f"stage{omni_stage_id}" if omni_stage_id is not None else "noid"
            set_death_signal(signal.SIGTERM)
            set_process_title(f"StageLLMCoreProc_{stage_label}_replica{omni_replica_id}_DP{dp_rank}")
            decorate_logs()
            # Workaround for flashinfer/jit-cache version mismatch in CI.
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            os.environ["VLLM_OMNI_REPLICA_ID"] = str(max(int(omni_replica_id), 0))

            # Audex CFG scheduler patches must land before EngineCore builds
            # its Scheduler; gated on the stage's logits_processors config.
            # (The decoder-type rebinding OmniEngineCoreRequest formerly patched
            # inline here is now handled globally by vllm_omni.patch, which binds
            # vLLM's EngineCoreRequest to StageLLMCoreRequest.)
            maybe_apply_audex_cfg_patches(kwargs.get("vllm_config"))

            engine_core = StageLLMCoreProc(
                *args,
                engine_index=dp_rank,
                **kwargs,
            )

            # Each subprocess corresponds to exactly one omni replica with its
            # own OmniMasterServer allocation, so the heartbeat client runs
            # unconditionally — there is no dp_rank-based gating.
            if omni_coord_address is not None:
                if omni_stage_id is None:
                    raise ValueError("omni_stage_id must be provided when omni_coord_address is set")
                addresses: EngineZmqAddresses = engine_core.addresses
                if not addresses.inputs or not addresses.outputs:
                    raise RuntimeError(
                        "EngineCore handshake did not populate input/output addresses; "
                        "cannot start OmniCoordClientForStage"
                    )
                scheduler = getattr(engine_core, "scheduler", None)
                if scheduler is None:
                    raise RuntimeError("EngineCore scheduler is not initialized")
                coord_client = create_stage_coord_client(
                    coord_zmq_addr=omni_coord_address,
                    input_addr=addresses.inputs[0],
                    output_addr=addresses.outputs[0],
                    stage_id=int(omni_stage_id),
                    queue_length_getter=scheduler.get_num_unfinished_requests,
                )

            def wakeup_engine() -> None:
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum: int, frame: FrameType | None) -> None:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()
                raise SystemExit(_signal_exit_code(signum))

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageLLMCoreProc exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("StageLLMCoreProc failed to start.")
            else:
                logger.exception("StageLLMCoreProc encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if coord_client is not None:
                with contextlib.suppress(RuntimeError):
                    coord_client.close()
            if engine_core is not None:
                engine_core.shutdown()
