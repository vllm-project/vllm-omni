# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Distinct concurrent speech requests retain their expected transcripts."""

import threading
from types import SimpleNamespace

import pytest

from tests.helpers import assertions
from tests.helpers.client import OmniResponse, OnlineOmniClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_distinct_concurrent_speech_requests_validate_corresponding_transcripts(monkeypatch):
    barrier = threading.Barrier(2, timeout=5)

    def create(**kwargs):
        barrier.wait()
        return kwargs["input"].encode()

    client = object.__new__(OnlineOmniClient)
    client.client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
    client.run_level = "advanced_model"
    client.log_stats = False
    monkeypatch.setattr(
        client,
        "_process_non_stream_audio_speech_response",
        lambda response, **kwargs: OmniResponse(success=True, audio_bytes=response, audio_format="audio/wav"),
    )
    monkeypatch.setattr(assertions, "convert_audio_bytes_to_text", lambda audio, **kwargs: audio.decode())
    texts = ["The train leaves tomorrow morning.", "Please put the notebook on the table."]
    responses = client.send_audio_speech_request(
        [
            {"model": "test-model", "input": text, "response_format": "wav", "transcript_expected_text": text}
            for text in texts
        ]
    )
    assert [response.audio_bytes for response in responses] == [text.encode() for text in texts]


@pytest.mark.parametrize(
    ("configs", "request_num", "message"),
    [
        ([], 1, "must not be empty"),
        ([{"model": "test-model", "input": "Hello"}], 2, "request_num is not supported"),
    ],
)
def test_speech_request_list_rejects_ambiguous_batches(configs, request_num, message):
    client = object.__new__(OnlineOmniClient)
    with pytest.raises(ValueError, match=message):
        client.send_audio_speech_request(configs, request_num=request_num)
