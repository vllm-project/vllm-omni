# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("initialized", [False, True])
def test_runner_stops_payload_transport_before_releasing_model(monkeypatch, initialized):
    events = []
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner._omni_connector_initialized = initialized
    runner.shutdown_omni_connectors = lambda: events.append("connector")
    runner._shutdown_omni_prefix_cache = lambda: events.append("cache")
    monkeypatch.setattr(GPUModelRunner, "shutdown", lambda self: events.append("model"))

    runner.shutdown()

    assert events == (["connector", "cache", "model"] if initialized else ["cache", "model"])


def test_runner_without_payload_transport_can_shutdown(monkeypatch):
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner._shutdown_omni_prefix_cache = lambda: None
    shutdown = []
    monkeypatch.setattr(GPUModelRunner, "shutdown", lambda self: shutdown.append(True))

    runner.shutdown()

    assert shutdown == [True]
