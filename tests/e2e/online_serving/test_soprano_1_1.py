# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Soprano WAV API smoke, including negative requests and worker recovery.

SOPRANO_TEST_BASE_URL attaches to an existing local server. Otherwise the
standard OmniServer harness launches the explicit Soprano deploy. Optional
SOPRANO_ONLINE_ARTIFACTS saves actual HTTP headers/bodies and WAVs.
"""

import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.tts]
_TIMEOUT = 180


@pytest.fixture(scope="module")
def soprano_endpoint():
    attached = os.environ.get("SOPRANO_TEST_BASE_URL")
    if attached:
        yield attached.rstrip("/")
        return
    from tests.helpers.runtime import OmniServer

    model = os.environ.get("SOPRANO_MODEL_PATH", "ekwek/Soprano-1.1-80M")
    args = [
        "--deploy-config",
        get_deploy_config_path("soprano.yaml"),
        "--served-model-name",
        "soprano",
        "--disable-log-stats",
    ]
    with OmniServer(model, args, env_dict={"TRITON_F32_DEFAULT": "ieee"}) as server:
        yield f"http://{server.host}:{server.port}"


def _save(label: str, response, payload=None):
    directory = os.environ.get("SOPRANO_ONLINE_ARTIFACTS")
    if directory is None:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "url": response.url,
        "status": response.status_code,
        "headers": dict(response.headers),
        "request": payload,
    }
    (root / f"{label}.json").write_text(json.dumps(record, indent=2))
    suffix = ".wav" if response.content.startswith(b"RIFF") else ".body"
    (root / f"{label}{suffix}").write_bytes(response.content)


def _post(base, label, **extra):
    payload = {
        "model": "soprano",
        "input": "Hello, this is a reproducible speech synthesis test.",
        "response_format": "wav",
        "seed": 44,
        "extra_params": {"temperature": 0.3, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.2},
        **extra,
    }
    response = requests.post(f"{base}/v1/audio/speech", json=payload, timeout=_TIMEOUT)
    _save(label, response, payload)
    return response


def _wav(response):
    assert response.status_code == 200, response.text[:1000]
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content[:4] == b"RIFF" and response.content[8:12] == b"WAVE"
    wave, rate = sf.read(BytesIO(response.content), dtype="float32")
    assert rate == 32000 and wave.ndim == 1 and wave.size > 0
    assert np.isfinite(wave).all() and np.abs(wave).max() > 0
    return wave


@pytest.mark.core_model
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_complete_wav_and_fixed_default_voice(soprano_endpoint):
    base = soprano_endpoint
    health = requests.get(f"{base}/health", timeout=30)
    _save("health", health)
    assert health.status_code == 200
    models = requests.get(f"{base}/v1/models", timeout=30)
    _save("models", models)
    assert models.status_code == 200
    assert "soprano" in {m["id"] for m in models.json()["data"]}
    omitted = _wav(_post(base, "voice-omitted"))
    explicit = _wav(_post(base, "voice-default", voice="default", language="English", sample_rate=32000))
    assert np.array_equal(omitted, explicit)


@pytest.mark.core_model
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_unsupported_options_are_request_errors(soprano_endpoint):
    cases = [
        ("voice", {"voice": "alloy"}),
        ("instructions", {"instructions": "Speak happily"}),
        ("reference", {"ref_audio": "file:///this-must-not-be-opened.wav"}),
        ("language", {"language": "Chinese"}),
        ("stream", {"stream": True}),
        ("speed", {"speed": 1.2}),
        ("format", {"response_format": "mp3"}),
        ("rate", {"sample_rate": 24000}),
        ("budget", {"max_new_tokens": 513}),
        ("extra", {"extra_params": {"ignore_eos": True}}),
        ("invalid-sampling", {"extra_params": {"top_p": 1.1}}),
        (
            "multiple-units",
            {
                "input": "This sentence is long enough to stand alone. The next sentence is also long enough to stand alone."
            },
        ),
    ]
    for label, kwargs in cases:
        response = _post(soprano_endpoint, f"reject-{label}", **kwargs)
        assert response.status_code == 400, (label, response.status_code, response.text[:500])
        assert "error" in response.json()
        assert not response.content.startswith(b"RIFF")


@pytest.mark.core_model
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_short_output_error_does_not_kill_server(soprano_endpoint):
    response = _post(soprano_endpoint, "short-output-error", max_new_tokens=1)
    assert response.status_code == 500 and "error" in response.json()
    health = requests.get(f"{soprano_endpoint}/health", timeout=30)
    _save("health-after-error", health)
    assert health.status_code == 200
    _wav(_post(soprano_endpoint, "recovered"))
