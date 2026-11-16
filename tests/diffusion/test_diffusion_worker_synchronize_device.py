# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the device step of the diffusion pause barrier."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_worker(runner: object) -> DiffusionWorker:
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = runner
    worker.rank = 0
    return worker


def test_synchronize_device_waits_for_prefetch_before_device_sync(mocker):
    platform = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
    order: list = []
    manager = MagicMock()
    manager.wait_prefetch.side_effect = lambda timeout: order.append(("prefetch", timeout)) or True
    platform.synchronize.side_effect = lambda: order.append("sync")
    worker = _make_worker(SimpleNamespace(kv_transfer_manager=manager))

    worker.synchronize_device(timeout=2.5)

    assert order == [("prefetch", 2.5), "sync"]


def test_synchronize_device_fails_when_prefetch_is_still_outstanding(mocker):
    platform = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
    manager = MagicMock()
    manager.wait_prefetch.return_value = False
    worker = _make_worker(SimpleNamespace(kv_transfer_manager=manager))

    with pytest.raises(TimeoutError, match="prefetch"):
        worker.synchronize_device(timeout=0.01)

    platform.synchronize.assert_not_called()


def test_synchronize_device_without_transfer_manager_only_syncs(mocker):
    platform = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
    worker = _make_worker(SimpleNamespace(kv_transfer_manager=None))

    worker.synchronize_device(timeout=None)

    platform.synchronize.assert_called_once_with()
