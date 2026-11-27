# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-DuplexEval against a text-only deployment.

A Thinker-only MiniCPM-o serves text alone: it rejects a session that asks for
audio output. The runner must take the deployment's default outputs, or every
sample fails on a server error that says nothing about the model.
"""

import pytest

from tests.benchmarks.duplex.test_omni_duplex_eval_session_stats import (
    _CompletedSessionClient,
    _install_deadlines,
    _sample,
)
from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import PreparedSample
from vllm_omni.benchmarks.duplex_session_inputs import TURN_DETECTION_SERVER_VAD

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


class _TextOnlyClient(_CompletedSessionClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events.events[0]["session"]["capabilities"] = {
            "supported_output_modalities": ["text"],
            "supports_playback_ack": False,
        }
        self.configure_kwargs: dict[str, object] = {}

    async def configure(self, *args, **kwargs) -> None:
        self.configure_kwargs = kwargs
        self.configured = True


async def _generate(monkeypatch, tmp_path):
    created: list[_TextOnlyClient] = []

    def client_factory(*args, **kwargs):
        created.append(_TextOnlyClient(*args, **kwargs))
        return created[-1]

    _install_deadlines(monkeypatch)
    monkeypatch.setattr(runner, "RealtimeDuplexClient", client_factory)
    result = await runner.generate_sample(
        _sample(),
        url="ws://localhost:8000/v1/realtime",
        model="test-model",
        ref_audio=None,
        output_root=tmp_path,
        turn_detection=TURN_DETECTION_SERVER_VAD,
        prepared=PreparedSample(pcm=b"", frames=(), duration=1.0, ref_audio=None, ref_audio_sha256=None),
    )
    return result, created[0]


async def test_text_only_deployment_gets_its_default_outputs(monkeypatch, tmp_path) -> None:
    result, client = await _generate(monkeypatch, tmp_path)
    assert client.configure_kwargs["modalities"] is None, "asking for audio fails the session"
    assert result.error == ""
    assert result.metadata["ack_timeout"] is None


def _incomplete_client(reason: str):
    class _IncompleteClient(_TextOnlyClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.events.events[-1]["response"] = {
                "id": "r1",
                "status": "incomplete",
                "status_details": {"type": "incomplete", "reason": reason},
            }

    return _IncompleteClient


async def _generate_with(monkeypatch, tmp_path, client_cls):
    _install_deadlines(monkeypatch)
    monkeypatch.setattr(runner, "RealtimeDuplexClient", client_cls)
    return await runner.generate_sample(
        _sample(),
        url="ws://localhost:8000/v1/realtime",
        model="test-model",
        ref_audio=None,
        output_root=tmp_path,
        turn_detection=TURN_DETECTION_SERVER_VAD,
        prepared=PreparedSample(pcm=b"", frames=(), duration=1.0, ref_audio=None, ref_audio_sha256=None),
    )


async def test_continuation_limit_response_is_not_a_sample_failure(monkeypatch, tmp_path) -> None:
    result = await _generate_with(monkeypatch, tmp_path, _incomplete_client("continuation_limit"))
    assert result.error == ""
    assert result.metadata["unfinished_response_ids"] == []


async def test_other_incomplete_reason_still_fails_the_sample(monkeypatch, tmp_path) -> None:
    result = await _generate_with(monkeypatch, tmp_path, _incomplete_client("max_output_tokens"))
    assert "Response did not complete" in result.error
