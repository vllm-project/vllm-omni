# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for bounded diffusion worker process shutdown."""

from unittest.mock import MagicMock, call

import pytest

from vllm_omni.diffusion.executor.multiproc_executor import (
    _WORKER_JOIN_TIMEOUT_S,
    _WORKER_KILL_JOIN_TIMEOUT_S,
    _ExecutorShutdownCleaner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_shutdown_cleaner_kills_worker_stuck_after_terminate():
    proc = MagicMock()
    proc.name = "stuck-worker"
    proc.is_alive.return_value = True
    cleaner = _ExecutorShutdownCleaner(processes=[proc])

    cleaner()

    proc.terminate.assert_called_once_with()
    proc.kill.assert_called_once_with()
    assert proc.join.call_args_list == [
        call(_WORKER_JOIN_TIMEOUT_S),
        call(_WORKER_JOIN_TIMEOUT_S),
        call(_WORKER_KILL_JOIN_TIMEOUT_S),
    ]


def test_shutdown_cleaner_leaves_gracefully_stopped_worker_alone():
    proc = MagicMock()
    proc.is_alive.side_effect = [True, False]
    cleaner = _ExecutorShutdownCleaner(processes=[proc])

    cleaner()

    proc.join.assert_called_once_with(_WORKER_JOIN_TIMEOUT_S)
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()
