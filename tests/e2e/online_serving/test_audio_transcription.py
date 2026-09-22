# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E conformance tests for ``/v1/audio/transcriptions``.

Validates schema conformance, correctness, streaming, and parameter handling
for the OpenAI-compatible audio transcription endpoint served by vLLM.
The transcription route is an upstream vLLM passthrough — no ``--omni`` flag
is needed; a standard ASR model (Whisper) is sufficient.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_audio_transcription.py
"""

import concurrent.futures
import io
import json
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest
import requests
import soundfile as sf

from tests.helpers.mark import hardware_marks, hardware_test
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.slow, pytest.mark.omni]

MODEL = "openai/whisper-small"

_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=["--enforce-eager"],
            use_omni=False,
        ),
        id="whisper_small",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]


def _get_audio_path(name: str = "mary_had_lamb") -> str:
    """Return local path for a vLLM audio asset (downloads on first use)."""
    from vllm.assets.audio import AudioAsset

    return str(AudioAsset(name).get_local_path())


def _transcribe(openai_client, model: str, file, **kwargs):
    """Call ``/v1/audio/transcriptions`` via the OpenAI SDK and return parsed JSON."""
    defaults = {"language": "en", "response_format": "text", "temperature": 0.0}
    defaults.update(kwargs)
    result = openai_client.client.audio.transcriptions.create(
        model=model, file=file, **defaults
    )
    return json.loads(result)


# ---- Schema validation ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_transcription_response_schema(omni_server, openai_client) -> None:
    """Valid transcription returns JSON with ``text`` (str) and ``usage.seconds`` (int)."""
    with open(_get_audio_path("mary_had_lamb"), "rb") as f:
        out = _transcribe(openai_client, omni_server.model, f)

    assert "text" in out, f"Missing 'text' key in response: {out}"
    assert isinstance(out["text"], str), f"'text' is not a string: {type(out['text'])}"
    assert "usage" in out, f"Missing 'usage' key in response: {out}"
    assert "seconds" in out["usage"], f"Missing 'seconds' in usage: {out['usage']}"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_transcription_verbose_json(omni_server, openai_client) -> None:
    """``response_format='verbose_json'`` returns structured response with segments."""
    with open(_get_audio_path("mary_had_lamb"), "rb") as f:
        result = openai_client.client.audio.transcriptions.create(
            model=omni_server.model,
            file=f,
            language="en",
            response_format="verbose_json",
            temperature=0.0,
        )
    assert hasattr(result, "text"), f"Missing 'text' attribute: {result}"
    assert result.text, "Empty transcription text"


# ---- Correctness ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_transcription_basic_english(omni_server, openai_client) -> None:
    """Transcribe English audio — output contains expected phrase, usage reports 16s."""
    with open(_get_audio_path("mary_had_lamb"), "rb") as f:
        out = _transcribe(openai_client, omni_server.model, f)

    assert "mary had a little lamb" in out["text"].lower(), (
        f"Expected 'mary had a little lamb' in output, got: {out['text']!r}"
    )
    assert out["usage"]["seconds"] == 16, (
        f"Expected 16s of audio, got {out['usage']['seconds']}s"
    )


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_transcription_wav_format(omni_server, openai_client) -> None:
    """Transcription accepts WAV audio created in-memory."""
    audio_path = _get_audio_path("mary_had_lamb")
    data, sr = sf.read(audio_path)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, data, sr, format="WAV")
    wav_buf.seek(0)
    wav_buf.name = "test.wav"

    out = _transcribe(openai_client, omni_server.model, wav_buf)
    assert out["text"].strip(), "Empty transcription from WAV input"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_transcription_long_audio(omni_server, openai_client) -> None:
    """Tiled audio (10x) produces proportionally longer transcription and correct usage."""
    audio_path = _get_audio_path("mary_had_lamb")
    data, sr = sf.read(audio_path)
    tiled = np.tile(data, 10)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, tiled, sr, format="WAV")
    wav_buf.seek(0)
    wav_buf.name = "long.wav"

    out = _transcribe(openai_client, omni_server.model, wav_buf)
    text_lower = out["text"].lower()
    count = text_lower.count("mary")
    assert count >= 8, (
        f"Expected 'mary' at least 8 times in 10x tiled audio, got {count}. "
        f"Output: {out['text'][:200]!r}"
    )
    assert out["usage"]["seconds"] == 161, (
        f"Expected 161s usage for 10x tiled audio, got {out['usage']['seconds']}s"
    )


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_transcription_batched(omni_server, openai_client) -> None:
    """Two concurrent transcription requests both return correct text."""
    paths = [_get_audio_path("mary_had_lamb"), _get_audio_path("winning_call")]

    def _do_transcribe(audio_path):
        with open(audio_path, "rb") as f:
            return _transcribe(openai_client, omni_server.model, f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_do_transcribe, paths))

    assert results[0]["text"].strip(), "Empty transcription for mary_had_lamb"
    assert results[1]["text"].strip(), "Empty transcription for winning_call"
    assert "mary" in results[0]["text"].lower(), (
        f"mary_had_lamb transcription missing 'mary': {results[0]['text']!r}"
    )


# ---- Streaming ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_transcription_streaming(omni_server, openai_client) -> None:
    """Streaming transcription via SSE produces non-empty text."""
    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/transcriptions"
    with open(_get_audio_path("mary_had_lamb"), "rb") as f:
        resp = requests.post(
            url,
            files={"file": f},
            data={
                "model": omni_server.model,
                "language": "en",
                "stream": "true",
                "temperature": "0.0",
            },
            stream=True,
            timeout=300,
        )
    assert resp.status_code == 200, f"Streaming request failed: {resp.text[:200]}"

    streamed_text = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            line = line[len("data: "):]
        if line.strip() == "[DONE]":
            break
        chunk = json.loads(line)
        text = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if text:
            streamed_text += text

    assert streamed_text.strip(), "Streaming transcription returned empty text"
    assert "mary" in streamed_text.lower(), (
        f"Streaming output missing expected content: {streamed_text[:200]!r}"
    )


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_transcription_stream_usage(omni_server, openai_client) -> None:
    """Streaming with ``stream_include_usage=True`` emits a final usage chunk."""
    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/transcriptions"
    with open(_get_audio_path("mary_had_lamb"), "rb") as f:
        resp = requests.post(
            url,
            files={"file": f},
            data={
                "model": omni_server.model,
                "language": "en",
                "stream": "true",
                "stream_include_usage": "true",
                "temperature": "0.0",
            },
            stream=True,
            timeout=300,
        )
    assert resp.status_code == 200, f"Streaming request failed: {resp.text[:200]}"

    usage_found = False
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            usage_found = True

    assert usage_found, "No usage chunk found in streaming response"


# ---- Parameter handling ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_transcription_temperature_accepted(omni_server, openai_client) -> None:
    """The ``temperature`` parameter is accepted and produces non-empty output."""
    with open(_get_audio_path("mary_had_lamb"), "rb") as f:
        out = _transcribe(openai_client, omni_server.model, f, temperature=0.8)

    assert out["text"].strip(), "Empty transcription with temperature=0.8"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_transcription_max_tokens(omni_server, openai_client) -> None:
    """``max_completion_tokens=1`` produces strictly shorter output than uncapped."""
    with open(_get_audio_path("mary_had_lamb"), "rb") as f:
        full = _transcribe(openai_client, omni_server.model, f)

    with open(_get_audio_path("mary_had_lamb"), "rb") as f:
        capped = openai_client.client.audio.transcriptions.create(
            model=omni_server.model,
            file=f,
            language="en",
            response_format="text",
            temperature=0.0,
            extra_body={"max_completion_tokens": 1},
        )
    capped_out = json.loads(capped)
    assert len(capped_out["text"]) < len(full["text"]), (
        f"Capped output not shorter than full. "
        f"Capped: {capped_out['text']!r}, Full: {full['text'][:100]!r}"
    )
