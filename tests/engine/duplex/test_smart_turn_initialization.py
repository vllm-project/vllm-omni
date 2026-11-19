# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Provider failures have one error contract and disable the affected detector."""

import hashlib

import pytest

from tests.engine.duplex.test_session_runner import append_audio, close_harness, find, open_harness
from vllm_omni.engine.duplex import smart_turn
from vllm_omni.engine.duplex.turn_detection import SemanticTurnDetector, TurnDetectionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=["resolve", "read", "initialize", "warm_up"])
def broken_provider(request, monkeypatch, tmp_path):
    path = tmp_path / "smart-turn.onnx"
    path.write_bytes(b"model")
    monkeypatch.setattr(smart_turn, "SMART_TURN_SHA256", hashlib.sha256(b"model").hexdigest())
    failure = {
        "resolve": OSError("cache lookup failed"),
        "read": PermissionError("artifact unreadable"),
        "initialize": RuntimeError("ONNX initialization failed"),
        "warm_up": ValueError("warm-up failed"),
    }[request.param]
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(request.param)
        raise failure

    provider = smart_turn.SmartTurnBackendProvider(model_path=str(path))
    if request.param == "resolve":
        from vllm.transformers_utils import repo_utils

        provider.model_path = None
        monkeypatch.setattr(repo_utils, "try_get_local_file", fail)
    elif request.param == "read":
        original_read = type(path).read_bytes

        def read_bytes(candidate):
            return fail() if candidate == path else original_read(candidate)

        monkeypatch.setattr(type(path), "read_bytes", read_bytes)
    elif request.param == "initialize":
        import onnxruntime

        monkeypatch.setattr(onnxruntime, "InferenceSession", fail)
    else:
        import onnxruntime

        class _Input:
            name = "input_features"

        class _Session:
            def get_inputs(self):
                return [_Input()]

        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *args, **kwargs: _Session())
        monkeypatch.setattr(smart_turn.SmartTurnBackend, "_warm_up", fail)
    return provider, failure, attempts


def test_provider_normalizes_initialization_error_and_preserves_cause(broken_provider):
    provider, failure, attempts = broken_provider
    with pytest.raises(smart_turn.SemanticVADUnavailableError) as caught:
        provider.get()
    assert caught.value.__cause__ is failure
    assert provider._backend is None
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_initialization_failure_disables_detector_without_retry(broken_provider):
    provider, _, attempts = broken_provider
    h = await open_harness(auto_response=False)
    try:
        h.runner.control._detector = SemanticTurnDetector(
            TurnDetectionConfig.from_realtime({"type": "semantic_vad"}),
            smart_turn_provider=provider,
            frame_scorer=lambda _: 0.0,
        )
        events = await h.run(append_audio(is_speech=None))
        assert find(events, "error").code == "semantic_vad_unavailable"
        assert h.runner.control._detector is None
        assert provider._backend is None
        events = await h.run(append_audio(is_speech=None))
        assert not any(event.type == "error" for event in events)
        assert len(attempts) == 1
    finally:
        await close_harness(h)
