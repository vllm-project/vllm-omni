# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E conformance tests for ``/v1/audio/translations``.

Validates schema conformance, correctness, streaming, and parameter handling
for the OpenAI-compatible audio translation endpoint served by vLLM.
The translation route is an upstream vLLM passthrough — no ``--omni`` flag
is needed; a standard ASR model (Whisper) is sufficient.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_audio_translation.py
"""

import io
import json
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import openai
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


def _get_audio_path(name: str = "azacinto_foscolo") -> str:
    """Return local path for a vLLM audio asset (downloads on first use)."""
    from vllm.assets.audio import AudioAsset

    return str(AudioAsset(name).get_local_path())


def _translate(openai_client, model: str, file, **kwargs):
    """Call ``/v1/audio/translations`` via the OpenAI SDK and return parsed JSON."""
    defaults = {
        "response_format": "text",
        "temperature": 0.0,
        "extra_body": {"language": "it", "to_language": "en"},
    }
    extra_body = defaults.pop("extra_body")
    if "extra_body" in kwargs:
        extra_body.update(kwargs.pop("extra_body"))
    defaults.update(kwargs)
    result = openai_client.client.audio.translations.create(model=model, file=file, extra_body=extra_body, **defaults)
    return json.loads(result)


# ---- Schema validation ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_translation_response_schema(omni_server, openai_client) -> None:
    """Valid translation returns JSON with ``text`` (str)."""
    with open(_get_audio_path(), "rb") as f:
        out = _translate(openai_client, omni_server.model, f)

    assert "text" in out, f"Missing 'text' key in response: {out}"
    assert isinstance(out["text"], str), f"'text' is not a string: {type(out['text'])}"
    assert out["text"].strip(), "Empty translation text"


# ---- Correctness ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_translation_italian_to_english(omni_server, openai_client) -> None:
    """Translate Italian (Foscolo) audio to English — output contains 'greek sea'."""
    with open(_get_audio_path(), "rb") as f:
        out = _translate(openai_client, omni_server.model, f)

    assert "greek sea" in out["text"].lower(), f"Expected 'greek sea' in translation, got: {out['text']!r}"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_translation_with_prompt(omni_server, openai_client) -> None:
    """Conditioning prompt influences translation output."""
    prompt = "Nor have I ever"
    with open(_get_audio_path(), "rb") as f:
        out = openai_client.client.audio.translations.create(
            model=omni_server.model,
            file=f,
            prompt=prompt,
            extra_body={"language": "it", "to_language": "en"},
            response_format="text",
            temperature=0.0,
        )
    text = json.loads(out)["text"]
    assert prompt not in text, f"Prompt should not appear verbatim in output: {text!r}"
    assert text.strip(), "Empty translation with prompt conditioning"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_translation_long_audio(omni_server, openai_client) -> None:
    """Tiled audio (2x) produces proportionally longer translation."""
    from vllm.multimodal.media.audio import load_audio

    audio_path = _get_audio_path()
    audio, sr = load_audio(audio_path)
    tiled = np.tile(audio, 2)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, tiled, sr, format="WAV")
    wav_buf.seek(0)

    out = _translate(openai_client, omni_server.model, wav_buf)
    text_lower = out["text"].lower()
    count = text_lower.count("greek sea")
    assert count == 2, f"Expected 'greek sea' twice in 2x tiled audio, got {count}. Output: {out['text'][:300]!r}"


# ---- Streaming ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_translation_streaming(omni_server, openai_client) -> None:
    """Streaming translation via SSE produces non-empty text."""
    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/translations"
    with open(_get_audio_path(), "rb") as f:
        resp = requests.post(
            url,
            files={"file": f},
            data={
                "model": omni_server.model,
                "language": "it",
                "to_language": "en",
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
            line = line[len("data: ") :]
        if line.strip() == "[DONE]":
            break
        chunk = json.loads(line)
        text = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if text:
            streamed_text += text

    assert streamed_text.strip(), "Streaming translation returned empty text"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_translation_stream_usage(omni_server, openai_client) -> None:
    """Streaming with ``stream_include_usage=True`` emits usage chunks."""
    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/translations"
    with open(_get_audio_path(), "rb") as f:
        resp = requests.post(
            url,
            files={"file": f},
            data={
                "model": omni_server.model,
                "language": "it",
                "to_language": "en",
                "stream": "true",
                "stream_include_usage": "true",
                "stream_continuous_usage_stats": "true",
                "temperature": "0.0",
            },
            stream=True,
            timeout=300,
        )
    assert resp.status_code == 200, f"Streaming request failed: {resp.text[:200]}"

    final_usage = False
    continuous_usage = True
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload.strip() == "[DONE]":
            break
        chunk = json.loads(payload)
        choices = chunk.get("choices", [])
        if not choices:
            final_usage = True
        else:
            continuous_usage = continuous_usage and ("usage" in chunk)

    assert final_usage, "No final usage chunk found in streaming response"
    assert continuous_usage, "Continuous usage stats missing from streaming chunks"


# ---- Parameter handling ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_translation_max_tokens(omni_server, openai_client) -> None:
    """``max_completion_tokens=1`` produces strictly shorter output than uncapped."""
    with open(_get_audio_path("mary_had_lamb"), "rb") as f:
        full = openai_client.client.audio.translations.create(
            model=omni_server.model,
            file=f,
            response_format="text",
            temperature=0.0,
        )
    full_out = json.loads(full)

    with open(_get_audio_path("mary_had_lamb"), "rb") as f:
        capped = openai_client.client.audio.translations.create(
            model=omni_server.model,
            file=f,
            response_format="text",
            temperature=0.0,
            extra_body={"max_completion_tokens": 1},
        )
    capped_out = json.loads(capped)
    assert len(capped_out["text"]) < len(full_out["text"]), (
        f"Capped output not shorter than full. Capped: {capped_out['text']!r}, Full: {full_out['text'][:100]!r}"
    )


# ---- Error handling ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_translation_invalid_audio(omni_server, openai_client) -> None:
    """Corrupted audio data must be rejected with an error."""
    invalid_audio = io.BytesIO(b"not a valid audio file")
    invalid_audio.name = "invalid.wav"
    with pytest.raises((openai.BadRequestError, openai.APIStatusError)):
        openai_client.client.audio.translations.create(
            model=omni_server.model,
            file=invalid_audio,
            temperature=0.0,
        )
