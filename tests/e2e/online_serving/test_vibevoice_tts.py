# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI speech HTTP coverage for VibeVoice's default TP=1 topology."""

from __future__ import annotations

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pybase64 as base64
import pytest
import soundfile as sf
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.tts,
    pytest.mark.skipif(
        not torch.accelerator.is_available() or torch.accelerator.device_count() < 1,
        reason="One CUDA device is required",
    ),
]

_MODEL = os.getenv("VIBEVOICE_TEST_MODEL", "microsoft/VibeVoice-1.5B")
_TOKENIZER = os.getenv("VIBEVOICE_TEST_TOKENIZER", "Qwen/Qwen2.5-1.5B")
_MODEL_REVISION = os.getenv("VIBEVOICE_TEST_MODEL_REVISION")
_TOKENIZER_REVISION = os.getenv("VIBEVOICE_TEST_TOKENIZER_REVISION")
_DEPLOY_CONFIG_OVERRIDE = os.getenv("VIBEVOICE_TEST_DEPLOY_CONFIG")
_STAGE_CONFIG_PATH = (
    str(Path(_DEPLOY_CONFIG_OVERRIDE).expanduser().resolve())
    if _DEPLOY_CONFIG_OVERRIDE
    else get_deploy_config_path("vibevoice.yaml")
)
_REFERENCE_DATA_URL = get_asset_path("cosyvoice3/zero_shot_prompt.wav", as_data_url=True)
_REFERENCE_FILE = get_asset_path("cosyvoice3/zero_shot_prompt.wav").resolve()
_FOUR_SPEAKER_REFERENCE_URLS = [
    get_asset_path("cosyvoice3/zero_shot_prompt.wav", as_data_url=True),
    get_asset_path("glm_tts/jiayan_zh.wav", as_data_url=True),
    get_asset_path("indextts2/ref_audio.wav", as_data_url=True),
    get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True),
]


def _data_url_for_wav(waveform: np.ndarray, sample_rate: int = 24_000) -> str:
    with io.BytesIO() as buffer:
        sf.write(buffer, waveform, sample_rate, format="WAV")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


_SERVER_ARGS = [
    "--tokenizer",
    _TOKENIZER,
    "--allowed-local-media-path",
    str(get_asset_path("").resolve()),
    "--disable-log-stats",
]
if _MODEL_REVISION:
    _SERVER_ARGS.extend(["--revision", _MODEL_REVISION])
if _TOKENIZER_REVISION:
    _SERVER_ARGS.extend(["--tokenizer-revision", _TOKENIZER_REVISION])

_SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_STAGE_CONFIG_PATH,
            server_args=_SERVER_ARGS,
            env_dict={
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                # Deterministic media-fetch errors: the server-side fetch must
                # bypass the host proxy for localhost URLs, otherwise the
                # rejection message depends on whether the proxy is running.
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            },
            init_timeout=900,
            stage_init_timeout=600,
        ),
        id=f"official-vibevoice-tp1-{Path(_STAGE_CONFIG_PATH).stem}",
    )
]


def _speech_url(omni_server) -> str:
    return f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"


def _batch_url(omni_server) -> str:
    return f"{_speech_url(omni_server)}/batch"


def _voices_url(omni_server) -> str:
    return f"http://{omni_server.host}:{omni_server.port}/v1/audio/voices"


def _post(omni_server, payload: dict, *, timeout: float = 300.0) -> httpx.Response:
    # Local E2E traffic must not inherit developer/CI SOCKS proxy settings.
    with httpx.Client(trust_env=False, timeout=timeout) as client:
        return client.post(_speech_url(omni_server), json=payload)


def _assert_error(response: httpx.Response, message: str) -> None:
    assert response.status_code in (400, 422), response.text
    assert message in response.text


@pytest.fixture(scope="module", autouse=True)
def _require_real_weights(run_level: str) -> None:
    if run_level not in {"advanced_model", "full_model"}:
        pytest.skip("VibeVoice HTTP E2E requires --run-level advanced_model (or full_model)")


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_http_wav_pcm_local_file_001(omni_server) -> None:
    base_payload = {
        "model": omni_server.model,
        "input": "Hello.",
        "ref_audio": _REFERENCE_DATA_URL,
        "stream": False,
    }

    wav_response = _post(
        omni_server,
        {**base_payload, "response_format": "wav"},
    )
    assert wav_response.status_code == 200, wav_response.text
    assert wav_response.headers["content-type"].startswith("audio/wav")
    waveform, sample_rate = sf.read(io.BytesIO(wav_response.content), dtype="float32")
    assert sample_rate == 24_000
    assert waveform.ndim == 1
    assert waveform.size > 0
    assert waveform.size % 3_200 == 0
    assert np.isfinite(waveform).all()

    pcm_response = _post(
        omni_server,
        {**base_payload, "response_format": "pcm"},
    )
    assert pcm_response.status_code == 200, pcm_response.text
    assert pcm_response.headers["content-type"].startswith("audio/pcm")
    assert len(pcm_response.content) > 0
    assert len(pcm_response.content) % (3_200 * 2) == 0

    truncated_response = _post(
        omni_server,
        {
            **base_payload,
            "response_format": "wav",
            "max_new_tokens": 2,
        },
    )
    assert truncated_response.status_code == 200, truncated_response.text
    assert truncated_response.headers.get("X-Finish-Reason") == "length"
    assert truncated_response.content[:4] == b"RIFF"

    file_response = _post(
        omni_server,
        {
            **base_payload,
            "ref_audio": _REFERENCE_FILE.as_uri(),
            "response_format": "wav",
        },
    )
    assert file_response.status_code == 200, file_response.text
    assert file_response.content[:4] == b"RIFF"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_http_bundled_default_voices_002(omni_server) -> None:
    requests = [
        {
            "model": omni_server.model,
            "input": "Hello from the bundled default voice.",
            "response_format": "wav",
            "max_new_tokens": 512,
        },
        {
            "model": omni_server.model,
            "input": "\n".join(
                [
                    "Speaker 8: Welcome.",
                    "Speaker 3: It is good to be here.",
                    "Speaker 5: Let us begin.",
                    "Speaker 1: Thank you.",
                ]
            ),
            "response_format": "wav",
            "max_new_tokens": 1_024,
        },
    ]

    for payload in requests:
        response = _post(omni_server, payload, timeout=900.0)
        assert response.status_code == 200, response.text
        assert response.headers.get("X-Finish-Reason") == "stop"
        waveform, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        assert sample_rate == 24_000
        assert waveform.ndim == 1
        assert waveform.size > 0
        assert waveform.size % 3_200 == 0
        assert np.isfinite(waveform).all()
        assert float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))) > 1e-5


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_http_uploaded_voice_lifecycle_003(omni_server) -> None:
    voice_name = "vibevoice-e2e-narrator"
    with httpx.Client(trust_env=False, timeout=300.0) as client:
        try:
            with _REFERENCE_FILE.open("rb") as audio_file:
                upload_response = client.post(
                    _voices_url(omni_server),
                    data={
                        "name": voice_name,
                        "consent": "vibevoice-e2e-consent",
                        # The registry may retain this for other models;
                        # VibeVoice must resolve only the uploaded audio.
                        "ref_text": "stored but unused transcript",
                    },
                    files={
                        "audio_sample": (
                            _REFERENCE_FILE.name,
                            audio_file,
                            "audio/wav",
                        )
                    },
                )
            assert upload_response.status_code == 200, upload_response.text

            voices_response = client.get(_voices_url(omni_server))
            assert voices_response.status_code == 200, voices_response.text
            assert voice_name in voices_response.json()["voices"]

            speech_response = client.post(
                _speech_url(omni_server),
                json={
                    "model": omni_server.model,
                    "input": "Hello from an uploaded voice.",
                    "voice": voice_name,
                    "response_format": "wav",
                },
            )
            assert speech_response.status_code == 200, speech_response.text
            assert speech_response.headers.get("X-Finish-Reason") == "stop"
            waveform, sample_rate = sf.read(io.BytesIO(speech_response.content), dtype="float32")
            assert sample_rate == 24_000
            assert waveform.ndim == 1
            assert waveform.size > 0
        finally:
            delete_response = client.delete(f"{_voices_url(omni_server)}/{voice_name}")
            assert delete_response.status_code in (200, 404), delete_response.text


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_http_four_speaker_natural_004(omni_server) -> None:
    response = _post(
        omni_server,
        {
            "model": omni_server.model,
            "input": "\n".join(
                [
                    "Speaker 0: Welcome.",
                    "Speaker 1: It is good to be here.",
                    "Speaker 2: Let us begin.",
                    "Speaker 3: Thank you.",
                ]
            ),
            "ref_audio": _FOUR_SPEAKER_REFERENCE_URLS,
            "response_format": "wav",
            "max_new_tokens": 1_024,
        },
        timeout=900.0,
    )

    assert response.status_code == 200, response.text
    finish_reason = response.headers.get("X-Finish-Reason")
    assert finish_reason == "stop"
    waveform, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
    assert sample_rate == 24_000
    assert waveform.ndim == 1
    assert waveform.size >= 4 * 3_200
    assert waveform.size % 3_200 == 0
    assert waveform.size < 180 * sample_rate
    assert np.isfinite(waveform).all()
    assert float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))) > 1e-5


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_http_batch_mixed_results_005(omni_server) -> None:
    with httpx.Client(trust_env=False, timeout=600.0) as client:
        response = client.post(
            _batch_url(omni_server),
            json={
                "model": omni_server.model,
                "items": [
                    {
                        "input": "Force a short length cap.",
                        "ref_audio": _REFERENCE_DATA_URL,
                        "response_format": "wav",
                        "max_new_tokens": 2,
                    },
                    {
                        "input": "This item must fail before generation.",
                        "ref_audio": _REFERENCE_DATA_URL,
                        "instructions": "unsupported",
                    },
                    {
                        # Batch items use the same adapter fallback as direct
                        # speech requests when no reference is supplied.
                        "input": "Hello from a bundled default voice.",
                        "response_format": "wav",
                        "max_new_tokens": 256,
                    },
                ],
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 3
    assert payload["succeeded"] == 2
    assert payload["failed"] == 1
    first, invalid, third = payload["results"]
    assert first["status"] == "success"
    assert base64.b64decode(first["audio_data"])[:4] == b"RIFF"
    assert invalid["status"] == "error"
    assert "does not support 'instructions'" in invalid["error"]
    assert third["status"] == "success"
    assert base64.b64decode(third["audio_data"])[:4] == b"RIFF"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_sse_terminal_length_006(omni_server) -> None:
    delta_chunks: list[bytes] = []
    event_types: list[str] = []
    done_events: list[dict] = []
    with httpx.Client(trust_env=False, timeout=600.0) as client:
        with client.stream(
            "POST",
            _speech_url(omni_server),
            json={
                "model": omni_server.model,
                "input": "Force a short SSE length cap.",
                "ref_audio": _REFERENCE_DATA_URL,
                "response_format": "pcm",
                "stream": True,
                "max_new_tokens": 2,
            },
        ) as response:
            assert response.status_code == 200, response.read().decode()
            assert response.headers["content-type"].startswith("text/event-stream")
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line.removeprefix("data: "))
                event_types.append(event["type"])
                if event["type"] == "speech.audio.delta":
                    delta_chunks.append(base64.b64decode(event["audio"]))
                elif event["type"] == "speech.audio.done":
                    done_events.append(event)
                elif event["type"] == "speech.audio.error":
                    pytest.fail(f"SSE generation failed: {event}")

    # SSE transport may coalesce adjacent model audio chunks into one delta;
    # event boundaries are not model-token boundaries. Validate ordering and
    # the complete PCM payload instead of requiring one event per token.
    assert delta_chunks
    assert event_types[:-1] == ["speech.audio.delta"] * len(delta_chunks)
    assert event_types[-1] == "speech.audio.done"
    assert len(b"".join(delta_chunks)) == 2 * 3_200 * 2
    assert len(done_events) == 1
    assert done_events[0]["finish_reason"] == "length"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_raw_pcm_streaming_007(omni_server) -> None:
    audio = bytearray()
    with httpx.Client(trust_env=False, timeout=600.0) as client:
        with client.stream(
            "POST",
            _speech_url(omni_server),
            json={
                "model": omni_server.model,
                "input": "Force a short raw PCM length cap.",
                "ref_audio": _REFERENCE_DATA_URL,
                "response_format": "pcm",
                "stream": True,
                "stream_format": "audio",
                "max_new_tokens": 2,
            },
        ) as response:
            assert response.status_code == 200, response.read().decode()
            assert response.headers["content-type"].startswith("audio/pcm")
            for chunk in response.iter_bytes():
                audio.extend(chunk)

    assert len(audio) == 2 * 3_200 * 2


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_http_rejects_invalid_requests_008(omni_server) -> None:
    valid = {
        "model": omni_server.model,
        "input": "Hello.",
        "ref_audio": _REFERENCE_DATA_URL,
        "response_format": "wav",
    }

    _assert_error(
        _post(
            omni_server,
            {**valid, "extra_params": {"num_diffusion_steps": 0}},
        ),
        "positive integer",
    )
    _assert_error(
        _post(
            omni_server,
            {**valid, "extra_params": {"num_diffusion_steps": 1.5}},
        ),
        "positive integer",
    )
    _assert_error(
        _post(
            omni_server,
            {**valid, "extra_params": {"num_diffusion_steps": 51}},
        ),
        "cannot exceed 50",
    )
    _assert_error(
        _post(
            omni_server,
            {**valid, "extra_params": {"guidance_scale": 20.1}},
        ),
        "between 0.0 and 20.0",
    )
    _assert_error(
        _post(
            omni_server,
            {**valid, "extra_params": {"guid" + "ence_scale": 1.3}},
        ),
        "Unsupported VibeVoice extra_params",
    )
    _assert_error(
        _post(omni_server, {**valid, "seed": 42}),
        "request-level seed",
    )
    _assert_error(
        _post(
            omni_server,
            {
                **valid,
                "input": "Speaker 0: hello\nSpeaker 1: world",
                "ref_audio": [_REFERENCE_DATA_URL],
            },
        ),
        "found 2 speakers but received 1 reference audios",
    )
    _assert_error(
        _post(omni_server, {**valid, "ref_audio": "not-a-url"}),
        "ref_audio must be a URL",
    )
    _assert_error(
        _post(
            omni_server,
            {
                **valid,
                "ref_audio": "https://127.0.0.1:1/missing.wav",
            },
        ),
        "Cannot connect",
    )

    too_long = _data_url_for_wav(np.zeros(60 * 24_000 + 1, dtype=np.float32))
    _assert_error(
        _post(omni_server, {**valid, "ref_audio": too_long}),
        "maximum is 60s",
    )

    five_speakers = "\n".join(f"Speaker {index}: hello" for index in range(5))
    _assert_error(
        _post(
            omni_server,
            {
                **valid,
                "input": five_speakers,
                "ref_audio": [_REFERENCE_DATA_URL] * 5,
            },
        ),
        "at most 4",
    )


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_concurrent_mixed_controls_009(omni_server) -> None:
    token_limits = [2, 3, 4, 5]
    payloads = [
        {
            "model": omni_server.model,
            "input": "Default controls, short request.",
            "ref_audio": _REFERENCE_DATA_URL,
            "response_format": "pcm",
            "max_new_tokens": token_limits[0],
        },
        {
            "model": omni_server.model,
            "input": "Custom eager controls, first request.",
            "ref_audio": _REFERENCE_DATA_URL,
            "response_format": "pcm",
            "max_new_tokens": token_limits[1],
            "extra_params": {
                "guidance_scale": 1.0,
                "num_diffusion_steps": 5,
            },
        },
        {
            "model": omni_server.model,
            "input": "Default controls, independently finishing request.",
            "ref_audio": _REFERENCE_DATA_URL,
            "response_format": "pcm",
            "max_new_tokens": token_limits[2],
        },
        {
            "model": omni_server.model,
            "input": "Custom eager controls, second request.",
            "ref_audio": _REFERENCE_DATA_URL,
            "response_format": "pcm",
            "max_new_tokens": token_limits[3],
            "extra_params": {
                "guidance_scale": 2.0,
                "num_diffusion_steps": 7,
            },
        },
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(lambda payload: _post(omni_server, payload, timeout=600.0), payloads))

    for response, token_limit in zip(responses, token_limits, strict=True):
        assert response.status_code == 200, response.text
        assert response.headers.get("X-Finish-Reason") == "length"
        assert response.headers["content-type"].startswith("audio/pcm")
        assert len(response.content) == token_limit * 3_200 * 2


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_client_abort_keeps_server_healthy_010(omni_server) -> None:
    saw_delta = False
    with httpx.Client(trust_env=False, timeout=600.0) as client:
        with client.stream(
            "POST",
            _speech_url(omni_server),
            json={
                "model": omni_server.model,
                "input": "Generate enough speech for the client to abort after the first audio delta.",
                "ref_audio": _REFERENCE_DATA_URL,
                "response_format": "pcm",
                "stream": True,
                "max_new_tokens": 512,
            },
        ) as response:
            assert response.status_code == 200, response.read().decode()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line.removeprefix("data: "))
                if event["type"] == "speech.audio.delta":
                    saw_delta = True
                    break
        assert saw_delta

        probe = client.post(
            _speech_url(omni_server),
            json={
                "model": omni_server.model,
                "input": "Probe after abort.",
                "ref_audio": _REFERENCE_DATA_URL,
                "response_format": "pcm",
                "max_new_tokens": 2,
            },
        )

    assert probe.status_code == 200, probe.text
    assert probe.headers.get("X-Finish-Reason") == "length"
    assert len(probe.content) == 2 * 3_200 * 2
