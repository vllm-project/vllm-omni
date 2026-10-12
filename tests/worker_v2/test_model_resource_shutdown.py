# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest

from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("failing", [None, "sender", "worker", "plane", "model"])
def test_shutdown_drains_all_owners_and_model_resources_even_after_prior_failure(monkeypatch, failing):
    calls = []

    def close(name):
        def invoke():
            calls.append(name)
            if name == failing:
                raise RuntimeError(f"failed {name}")

        return invoke

    runner = object.__new__(OmniGPUModelRunner)
    runner.model_state = SimpleNamespace(_first_audio_sender=SimpleNamespace(close=close("sender")))
    runner._native_output_materializer = SimpleNamespace(close=close("worker"))
    runner._omni_data_plane = SimpleNamespace(close=close("plane"))
    runner.model = SimpleNamespace(release_worker_resources=close("model"))
    monkeypatch.setattr(OmniGPUModelRunner.__mro__[1], "shutdown", lambda self: calls.append("parent"))
    if failing is None:
        runner.shutdown()
    else:
        with pytest.raises(RuntimeError, match=f"failed {failing}"):
            runner.shutdown()
    assert calls == ["sender", "worker", "plane", "model", "parent"]
    assert runner._native_output_materializer is None
