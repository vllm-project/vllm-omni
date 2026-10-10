# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bound a stage's readiness handshake with ``stage_init_timeout``.

Upstream ``vllm.v1.engine.utils.wait_for_engine_startup`` returns when every
engine core reports READY and raises only when an engine process exits. A stage
that hangs while initializing therefore blocks its launcher forever, and the only
remaining bound is the engine-wide ``init_timeout`` on the main thread, which
cannot interrupt the launcher. This helper arms a timer for ``stage_init_timeout``
seconds; if it fires first, the stage's processes are terminated, upstream
observes the exit and raises, and the failure is re-raised as ``TimeoutError``
naming the stage.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine.utils import wait_for_engine_startup

logger = init_logger(__name__)


def wait_for_engine_startup_with_deadline(
    handshake_socket: Any,
    core_engines: Any,
    parallel_config: Any,
    coordinated_dp: bool,
    cache_config: Any,
    launch: Any,
    *,
    timeout: float | None,
    kill_processes: Callable[[], None],
    stage_label: str,
) -> None:
    """Run ``wait_for_engine_startup`` and fail with ``TimeoutError`` after ``timeout`` seconds.

    Args:
        timeout: ``stage_init_timeout`` in seconds. ``None`` or ``<= 0`` disables the deadline.
        kill_processes: Terminates the stage's processes when the deadline passes; the
            resulting process exit is what makes upstream's poll loop return.
        stage_label: Human-readable stage name for the error message and logs.
    """
    if timeout is None or timeout <= 0:
        wait_for_engine_startup(handshake_socket, core_engines, parallel_config, coordinated_dp, cache_config, launch)
        return

    timed_out = threading.Event()

    def _on_deadline() -> None:
        timed_out.set()
        logger.error(
            "[StageInit] %s not READY after %ss (stage_init_timeout); terminating its processes",
            stage_label,
            timeout,
        )
        try:
            kill_processes()
        except Exception:
            logger.exception("[StageInit] Failed to terminate %s after the readiness deadline", stage_label)

    timer = threading.Timer(timeout, _on_deadline)
    timer.daemon = True
    timer.start()
    try:
        wait_for_engine_startup(handshake_socket, core_engines, parallel_config, coordinated_dp, cache_config, launch)
    except Exception as exc:
        if timed_out.is_set():
            raise TimeoutError(
                f"{stage_label} did not become ready within {timeout}s (stage_init_timeout); "
                "its processes were terminated"
            ) from exc
        raise
    finally:
        timer.cancel()
    if timed_out.is_set():
        # READY arrived in the same instant the deadline fired and the processes were
        # already terminated: do not hand back a stage whose engine is gone.
        raise TimeoutError(
            f"{stage_label} did not become ready within {timeout}s (stage_init_timeout); its processes were terminated"
        )
