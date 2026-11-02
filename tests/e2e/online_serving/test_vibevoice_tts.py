# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Functional VibeVoice HTTP coverage; quality benchmarks are separate.

Uses the shared speech client and its SDK for stream-level assertions.
These are transport tests, not ASR / voice-similarity qualification.
"""

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("VIBEVOICE_MODEL", "vibevoice/VibeVoice-1.5B-hf")
REFERENCE = Path(__file__).resolve().parents[2] / "assets/qwen3_tts/clone_2.wav"
TEXT = "Hello, this is a short speech synthesis test."
pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.tts,
    *hardware_marks(res={"cuda": "H100"}, num_cards=1),
    pytest.mark.parametrize(
        "omni_server",
        [
            pytest.param(
                OmniServerParams(
                    model=MODEL,
                    stage_config_path=get_deploy_config_path("vibevoice.yaml"),
                    server_args=["--disable-log-stats"],
                ),
                id="vibevoice",
            )
        ],
        indirect=True,
    ),
]


@pytest.fixture
def ref_audio():
    return "data:audio/wav;base64," + base64.b64encode(REFERENCE.read_bytes()).decode()


def _assert_wav_non_silent(responses: list) -> None:
    """Decode the WAV body and reject silence; the shared assert skips HNR for WAV."""
    audio, sr = sf.read(io.BytesIO(responses[0].audio_bytes), dtype="int16")
    assert audio.ndim == 1 and audio.dtype == np.int16 and sr == 24000
    assert audio.size % 3200 == 0 and np.abs(audio.astype(np.int32)).max() > 32


def test_no_reference(omni_server, online_client):
    responses = online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": TEXT,
            "response_format": "wav",
            "sample_rate": 24000,
            "max_new_tokens": 128,
            "min_audio_bytes": 6400,
            "timeout": 300,
            "transcript_expected_text": "",
        }
    )
    _assert_wav_non_silent(responses)


@pytest.mark.parametrize("speakers", [2, 4])
def test_multispeaker_references(omni_server, online_client, ref_audio, speakers):
    responses = online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "\n".join(f"Speaker {s}: Hello, this is speaker number {s}." for s in range(speakers)),
            "ref_audio": [ref_audio] * speakers,
            "response_format": "wav",
            "sample_rate": 24000,
            "max_new_tokens": 256,
            "min_audio_bytes": 6400,
            "timeout": 300,
            "transcript_expected_text": "",
        }
    )
    _assert_wav_non_silent(responses)


@pytest.mark.parametrize("fmt", ["pcm", "wav", "sse"])
def test_audio_stream_contract(omni_server, online_client, ref_audio, fmt):
    # The collecting helper does not expose SSE events. Use its SDK directly
    # to check framing, terminal usage and PCM/WAV payloads under one contract.
    with online_client.client.audio.speech.with_streaming_response.create(
        model=omni_server.model,
        input=TEXT,
        voice=None,
        response_format="wav" if fmt == "wav" else "pcm",
        extra_body={"ref_audio": ref_audio, "stream_format": "sse" if fmt == "sse" else "audio", "max_new_tokens": 64},
        timeout=300,
    ) as response:
        if fmt == "sse":
            assert "text/event-stream" in response.headers["content-type"]
            events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
            assert len(events) > 1 and events[-1]["type"] == "speech.audio.done"
            deltas = events[:-1]
            assert all(event["type"] == "speech.audio.delta" for event in deltas)
            assert all(event["sample_rate"] == 24000 and event["channels"] == 1 for event in deltas)
            data = b"".join(base64.b64decode(event["audio"], validate=True) for event in deltas)
            usage = events[-1]["usage"]
            assert len(data) // 6400 <= usage["output_tokens"] <= 64
            assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
        else:
            assert fmt in response.headers["content-type"]
            data = b"".join(response.iter_bytes())
    if fmt == "wav":
        audio, sr = sf.read(io.BytesIO(data), dtype="int16")
        assert sr == 24000
    else:
        assert len(data) % 2 == 0
        audio = np.frombuffer(data, dtype="<i2")
    assert audio.ndim == 1 and audio.dtype == np.int16
    assert 0 < audio.size <= 64 * 3200 and audio.size % 3200 == 0
    assert np.abs(audio.astype(np.int32)).max() > 32
    assert audio.size / 24000 > 0.1


def test_queue_beyond_four(omni_server, online_client, ref_audio):
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": TEXT,
            "ref_audio": ref_audio,
            "response_format": "wav",
            "sample_rate": 24000,
            "max_new_tokens": 64,
            "min_audio_bytes": 6400,
            "timeout": 300,
            "transcript_expected_text": "",
        },
        request_num=6,
    )


def test_disconnect_then_reuse(omni_server, online_client, ref_audio):
    first_audio = Barrier(4, timeout=120)

    def disconnect():
        with online_client.client.audio.speech.with_streaming_response.create(
            model=omni_server.model,
            input=(TEXT + " ") * 40,
            voice=None,
            response_format="pcm",
            extra_body={"ref_audio": ref_audio, "stream_format": "audio", "max_new_tokens": 2048},
            timeout=180,
        ) as response:
            assert next(response.iter_bytes())
            first_audio.wait()  # keep all four streams open until each emits
        # Closing the response cancels this stream, rather than draining it.

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(disconnect) for _ in range(4)]
        for future in futures:
            future.result(timeout=180)
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": TEXT,
            "ref_audio": ref_audio,
            "response_format": "wav",
            "sample_rate": 24000,
            "max_new_tokens": 64,
            "min_audio_bytes": 6400,
            "timeout": 180,
            "transcript_expected_text": "",
        },
        request_num=4,
    )


def test_uploaded_voice(omni_server, online_client):
    name = f"vibevoice-test-{uuid4().hex}"
    online_client.send_audio_voices_create_http_request(
        {
            "data": {"name": name, "consent": "test-fixture", "ref_text": "Stored transcript is not conditioning."},
            "files": {"audio_sample": ("reference.wav", REFERENCE.read_bytes(), "audio/wav")},
        }
    )
    try:
        online_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": TEXT,
                "voice": name,
                "response_format": "wav",
                "sample_rate": 24000,
                "max_new_tokens": 128,
                "min_audio_bytes": 6400,
                "timeout": 300,
                "transcript_expected_text": "",
            }
        )
    finally:
        online_client.send_audio_voices_delete_http_request({"name": name})
