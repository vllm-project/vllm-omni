# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.worker import diffusion_worker as worker_module

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("fails", [False, True])
def test_worker_shutdown_uses_terminal_offloader_cleanup(monkeypatch, caplog, fails):
    worker = object.__new__(worker_module.DiffusionWorker)
    events = []
    backend = Mock()

    def shutdown():
        events.append("offloader")
        if fails:
            raise RuntimeError("offloader shutdown failed")

    backend.shutdown.side_effect = shutdown
    manager = SimpleNamespace(close=lambda: events.append("kv"))
    worker.model_runner = SimpleNamespace(offload_backend=backend, kv_transfer_manager=manager)
    monkeypatch.setattr(worker_module, "shutdown_kv_connector", lambda: events.append("connector"))
    monkeypatch.setattr(worker_module, "destroy_distributed_env", lambda: events.append("distributed"))
    worker.shutdown()
    assert events == ["offloader", "kv", "connector", "distributed"]
    backend.disable.assert_not_called()
    assert ("Failed to shut down diffusion offload backend during shutdown" in caplog.text) is fails
    if fails:
        assert "offloader shutdown failed" in caplog.text
