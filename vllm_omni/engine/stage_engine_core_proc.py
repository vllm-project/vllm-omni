"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import contextlib
import os
import signal
from typing import Any

import vllm.v1.engine.core as _vllm_engine_core_module
from vllm.logger import init_logger
from vllm.tracing import maybe_init_worker_tracer
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
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.stage_init_utils import set_death_signal

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        omni_coordinator_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageEngineCoreProc busy loop in background process.

        Omni-specific kwargs:
          - ``omni_coordinator_address``: ROUTER address of the head-side
            :class:`OmniCoordinator`. When provided, this subprocess
            instantiates an :class:`OmniCoordClientForStage` after the
            HELLO/INIT/READY handshake completes and reports its status +
            queue length via heartbeats. The hook is wired so each
            heartbeat refreshes ``queue_length`` from the live scheduler.
          - ``omni_stage_id``: logical stage id this replica belongs to.
            Required when ``omni_coordinator_address`` is provided.
          - ``omni_replica_id``: cluster-unique replica id within the
            stage (assigned by :class:`OmniMasterServer`). Used for
            logging / metrics only.
        """
        signal_callback: SignalCallback | None = None
        maybe_register_config_serialize_by_value()

        # Register vllm-omni reasoning parsers (e.g. step_audio) in this
        # subprocess so they are available when the engine core resolves
        # ``--reasoning-parser``.  The main process already registered them
        # at import time, but the forked subprocess starts with a fresh
        # ReasoningParserManager.
        try:
            import vllm_omni.reasoning  # noqa: F401
        except ImportError:
            logger.warning(
                "Failed to import vllm_omni.reasoning in subprocess; "
                "custom reasoning parsers (e.g. step_audio) will not be "
                "available."
            )

        engine_core: StageEngineCoreProc | None = None
        coord_client = None
        try:
            # NOTE: previous revisions hardcoded data_parallel_size=1 here
            # (TODO referencing issue #984). The hardcoding has been removed
            # so the DP fields propagate through from the caller exactly
            # like upstream vLLM.

            stage_label = f"stage{omni_stage_id}" if omni_stage_id is not None else "noid"
            set_death_signal(signal.SIGTERM)
            set_process_title(f"StageEngineCoreProc_{stage_label}_replica{omni_replica_id}_DP{dp_rank}")
            maybe_init_worker_tracer(
                "vllm_omni",
                "stage_engine_core",
                stage_label,
            )
            decorate_logs()
            # Workaround for flashinfer/jit-cache version mismatch in CI.
            # The parent process handles this gracefully via ring_globals.py,
            # but the subprocess hits an unprotected import in TopKTopPSampler.
            # Setting this env var allows the same graceful fallback to work.
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            os.environ["VLLM_OMNI_REPLICA_ID"] = str(max(int(omni_replica_id), 0))

            # Patch the decoder type so process_input_sockets (started
            # during __init__) decodes OmniEngineCoreRequest (which
            # carries additional_information) instead of the base
            # EngineCoreRequest.  Must happen BEFORE __init__ because
            # the IO thread creates MsgpackDecoder(EngineCoreRequest)
            # during __init__.
            _vllm_engine_core_module.EngineCoreRequest = OmniEngineCoreRequest
            logger.debug(
                "[StageEngineCoreProc] Patched EngineCoreRequest -> OmniEngineCoreRequest: %s",
                _vllm_engine_core_module.EngineCoreRequest,
            )

            engine_core = StageEngineCoreProc(
                *args,
                engine_index=dp_rank,
                **kwargs,
            )

            # Each subprocess corresponds to exactly one omni replica with
            # its own OmniMasterServer allocation, so the heartbeat client
            # runs unconditionally — there is no dp_rank-based gating.
            if omni_coordinator_address is not None:
                if omni_stage_id is None:
                    raise ValueError("omni_stage_id must be provided when omni_coordinator_address is set")
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
                    coord_zmq_addr=omni_coordinator_address,
                    input_addr=addresses.inputs[0],
                    output_addr=addresses.outputs[0],
                    stage_id=int(omni_stage_id),
                    queue_length_getter=scheduler.get_num_unfinished_requests,
                )

            def wakeup_engine() -> None:
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum: int, frame: Any) -> None:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()
                raise SystemExit(_signal_exit_code(signum))

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageEngineCoreProc exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("StageEngineCoreProc failed to start.")
            else:
                logger.exception("StageEngineCoreProc encountered a fatal error.")
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
