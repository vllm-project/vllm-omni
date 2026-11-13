# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-DuplexEval against a text-only deployment.

A Thinker-only MiniCPM-o serves text alone: it rejects a session that asks for
audio output and answers ``playback.ack`` with ``unsupported_operation``. The
runner must take the deployment's default outputs and send no ACKs, or every
sample fails on server errors that say nothing about the model.
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


class _SessionClient(_CompletedSessionClient):
    supports_playback_ack = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events.events[0]["session"]["capabilities"] = {
            "supported_output_modalities": ["audio", "text"] if self.supports_playback_ack else ["text"],
            "supports_playback_ack": self.supports_playback_ack,
        }
        self.configure_kwargs: dict[str, object] = {}
        self.ack_calls = 0

    async def configure(self, *args, **kwargs) -> None:
        self.configure_kwargs = kwargs
        self.configured = True

    async def acknowledge_playback(self) -> None:
        self.ack_calls += 1


class _TextOnlyClient(_SessionClient):
    supports_playback_ack = False


async def _generate(monkeypatch, tmp_path, client_cls):
    created: list[_SessionClient] = []

    def client_factory(*args, **kwargs):
        created.append(client_cls(*args, **kwargs))
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


async def test_text_only_deployment_gets_its_default_outputs_and_no_acks(monkeypatch, tmp_path) -> None:
    result, client = await _generate(monkeypatch, tmp_path, _TextOnlyClient)
    assert client.configure_kwargs["modalities"] is None, "asking for audio fails the session"
    assert client.ack_calls == 0
    assert result.error == ""
    assert result.metadata["ack_timeout"] is None


async def test_speech_deployment_still_acknowledges_playback(monkeypatch, tmp_path) -> None:
    result, client = await _generate(monkeypatch, tmp_path, _SessionClient)
    assert client.ack_calls >= 1
    assert result.error == ""
