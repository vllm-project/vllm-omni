# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The shared speech helper must time audio, not count SSE framing as PCM."""

import base64
import json
import time

import pytest

from tests.helpers.client import OnlineOmniClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("attribute", ["response", "http_response"])
def test_sse_split_frames_are_decoded_before_audio_is_counted(mocker, attribute):
    pcm = b"\x01\x02\x03\x04"
    wire = (
        "event: speech.audio.delta\ndata: "
        + json.dumps({"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode()})
        + '\n\nevent: speech.audio.done\ndata: {"type":"speech.audio.done"}\n\n'
    ).encode()
    response = mocker.Mock(spec=[attribute, "iter_bytes"])
    getattr(response, attribute).headers = {"content-type": "text/event-stream"}
    response.iter_bytes.return_value = iter([wire[:17], wire[17:41], wire[41:]])
    client = object.__new__(OnlineOmniClient)
    result = client._process_stream_audio_speech_response(
        response, response_format="pcm", wall_start=time.perf_counter()
    )
    assert result.success and result.audio_bytes == pcm
    assert result.audio_chunk_sizes == [4]
    assert result.audio_chunk_arrivals_s is not None
    assert result.e2e_latency is not None
    assert len(result.audio_chunk_arrivals_s) == 1
    assert result.audio_chunk_arrivals_s[0] <= result.e2e_latency


def test_missing_sse_done_is_a_failure(mocker):
    response = mocker.Mock(spec=["response", "iter_bytes"])
    response.response.headers = {"content-type": "text/event-stream"}
    response.iter_bytes.return_value = iter([b'data: {"type":"speech.audio.delta","audio":"AQI="}\n\n'])
    client = object.__new__(OnlineOmniClient)
    assert not client._process_stream_audio_speech_response(response, wall_start=time.perf_counter()).success
